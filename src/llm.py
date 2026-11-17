"""Cliente LLM compartilhado para o Azure AI Foundry."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AsyncOpenAI, OpenAI

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)


def required_env(name: str) -> str:
    """Obtém uma configuração obrigatória carregada do ambiente ou do .env."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variável de ambiente obrigatória não configurada: {name}. "
            f"Verifique o arquivo {ENV_FILE}."
        )
    return value


LLM_DEPLOYMENT = required_env("AZURE_OPENAI_DEPLOYMENT")


def create_llm() -> OpenAI:
    """Cria um cliente compatível com a API v1 do Azure AI Foundry."""
    return OpenAI(
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        base_url=required_env("AZURE_OPENAI_BASE_URL"),
        timeout=30.0,
        max_retries=2,
    )


def create_chat_llm(
    *,
    temperature: float = 0.1,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> AzureChatOpenAI:
    """Cria um AzureChatOpenAI compatível com a API v1 do Foundry."""
    client_options = {
        "api_key": required_env("AZURE_OPENAI_API_KEY"),
        "base_url": required_env("AZURE_OPENAI_BASE_URL"),
        "timeout": timeout,
        "max_retries": max_retries,
    }
    sync_client = OpenAI(**client_options)
    async_client = AsyncOpenAI(**client_options)

    return AzureChatOpenAI(
        client=sync_client.chat.completions,
        async_client=async_client.chat.completions,
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=None,
        api_version="v1",
        model=LLM_DEPLOYMENT,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


def create_embedding(
    *,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> AzureOpenAIEmbeddings:
    """Cria embeddings Azure usando o endpoint v1 no escopo do recurso."""
    client_options = {
        "api_key": required_env("AZURE_OPENAI_API_KEY"),
        "base_url": required_env("AZURE_OPENAI_EMBEDDINGS_BASE_URL"),
        "timeout": timeout,
        "max_retries": max_retries,
    }
    sync_client = OpenAI(**client_options)
    async_client = AsyncOpenAI(**client_options)

    return AzureOpenAIEmbeddings(
        client=sync_client.embeddings,
        async_client=async_client.embeddings,
        api_key=required_env("AZURE_OPENAI_API_KEY"),
        azure_endpoint=None,
        api_version="v1",
        model=required_env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_MODEL"),
        azure_deployment=required_env(
            "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"
        ),
        timeout=timeout,
        max_retries=max_retries,
    )


# Instância compartilhada para uso com: from llm import llm
llm = create_llm()

__all__ = [
    "LLM_DEPLOYMENT",
    "create_chat_llm",
    "create_embedding",
    "create_llm",
    "llm",
    "required_env",
]
