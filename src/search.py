from langchain_core.documents import Document
from langchain_postgres import PGVector

from llm import create_chat_llm, create_embedding, required_env

NO_INFORMATION_RESPONSE = (
    "Não tenho informações necessárias para responder sua pergunta."
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def create_vector_store() -> PGVector:
    """Conecta à collection existente usando o embedding Azure."""
    return PGVector(
        embeddings=create_embedding(),
        collection_name=required_env("PG_VECTOR_COLLECTION_NAME"),
        connection=required_env("DATABASE_URL"),
        use_jsonb=True,
    )


def _join_context(results: list[tuple[Document, float]]) -> str:
    return "\n\n---\n\n".join(
        document.page_content.strip()
        for document, _score in results
        if document.page_content.strip()
    )


def search_prompt(question: str) -> str:
    """Busca os dez chunks mais próximos e responde segundo o prompt RAG."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("A pergunta não pode estar vazia.")

    results = create_vector_store().similarity_search_with_score(
        normalized_question,
        k=10,
    )
    context = _join_context(results)
    if not context:
        return NO_INFORMATION_RESPONSE

    prompt = PROMPT_TEMPLATE.format(
        contexto=context,
        pergunta=normalized_question,
    )
    response = create_chat_llm().invoke(prompt)
    return str(response.content).strip()


__all__ = ["create_vector_store", "search_prompt"]
