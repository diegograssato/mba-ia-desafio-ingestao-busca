"""Ingestão idempotente de um PDF no PostgreSQL com pgVector."""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import ENV_FILE, create_embedding, required_env

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def _resolve_pdf_path(pdf_path: str | Path | None = None) -> Path:
    """Resolve caminhos relativos a partir da raiz do projeto, onde fica o .env."""
    configured_path = Path(pdf_path or required_env("PDF_PATH"))
    if not configured_path.is_absolute():
        configured_path = ENV_FILE.parent / configured_path

    resolved_path = configured_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {resolved_path}")
    return resolved_path


def _document_ids(chunks: list, document_hash: str) -> list[str]:
    """Gera IDs estáveis para evitar duplicar chunks em novas execuções."""
    ids: list[str] = []
    for chunk in chunks:
        # O conteúdo participa da identidade para que uma alteração no texto
        # produza um novo vetor, mesmo quando página e posição forem iguais.
        identity = "|".join(
            (
                document_hash,
                str(chunk.metadata.get("page", "")),
                str(chunk.metadata.get("start_index", "")),
                chunk.page_content,
            )
        )
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    return ids


def ingest_pdf(pdf_path: str | Path | None = None) -> int:
    """Divide o PDF, gera embeddings Azure e faz upsert no PGVector."""
    resolved_pdf = _resolve_pdf_path(pdf_path)

    # O hash do arquivo permite rastrear de qual versão do PDF veio cada chunk.
    document_hash = hashlib.sha256(resolved_pdf.read_bytes()).hexdigest()

    pages = PyPDFLoader(str(resolved_pdf)).load()

    # O overlap preserva parte do contexto quando uma ideia cruza a fronteira
    # entre dois chunks; add_start_index mantém a posição original rastreável.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise RuntimeError(f"Nenhum texto foi extraído do PDF: {resolved_pdf}")

    # Metadados são armazenados em JSONB pelo PGVector e podem apoiar auditoria,
    # filtros ou citações por documento/página em evoluções futuras do RAG.
    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": resolved_pdf.name,
                "document_sha256": document_hash,
                "chunk_index": chunk_index,
            }
        )

    # O cliente de embeddings chama o deployment configurado no Azure OpenAI;
    # somente os vetores e metadados resultantes são persistidos no PostgreSQL.
    vector_store = PGVector(
        embeddings=create_embedding(),
        collection_name=required_env("PG_VECTOR_COLLECTION_NAME"),
        connection=required_env("DATABASE_URL"),
        use_jsonb=True,
    )
    vector_store.add_documents(
        documents=chunks,
        ids=_document_ids(chunks, document_hash),
    )
    return len(chunks)


if __name__ == "__main__":
    total = ingest_pdf()
    print(f"Ingestão concluída: {total} chunks armazenados.")
