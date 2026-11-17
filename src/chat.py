"""Interface CLI mínima para consultar o pipeline RAG."""

from search import search_prompt


def pretty_print(response: object) -> None:
    """Exibe respostas LangChain ou valores simples de forma uniforme."""
    # Objetos retornados pelo LangChain normalmente expõem `content`;
    # o fallback mantém a CLI útil caso outra implementação retorne string.
    content = getattr(response, "content", None)
    print(f"\n{'-' * 100}\n\n{content if content else response}")


def main() -> None:
    """Mantém o chat ativo até o usuário solicitar sua finalização."""
    print('Chat iniciado. Digite "sair" para encerrar.')

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat encerrado.")
            break

        if question.casefold() == "sair":
            print("Chat encerrado.")
            break

        if not question:
            print("Digite uma pergunta ou \"sair\" para encerrar.")
            continue

        # search_prompt concentra validação, busca vetorial e chamada ao deployment
        # de chat no Azure, mantendo esta camada restrita à interação no terminal.
        answer = search_prompt(question)
        pretty_print(answer)


if __name__ == "__main__":
    main()
