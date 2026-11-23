from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você responde perguntas sobre o manual de operações do porto. "
            "Use SOMENTE o contexto fornecido. Se a resposta não estiver no contexto, "
            "responda exatamente: Não encontrado no manual de operações.",
        ),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)
