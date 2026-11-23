import os

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model

load_dotenv()


def get_llm(temperature: float = 0):
    """Cria o chat model a partir das variáveis de ambiente configuradas."""
    return init_chat_model(
        model=os.environ["LLM_MODEL"],
        model_provider=os.environ["LLM_PROVIDER"],
        temperature=temperature,
    )


def get_embeddings():
    """Cria o modelo de embeddings a partir das variáveis de ambiente configuradas."""
    return init_embeddings(
        model=os.environ["EMBEDDINGS_MODEL"],
        provider=os.environ["EMBEDDINGS_PROVIDER"],
    )


def invocar_com_retry(llm, messages, ferramentas):
    """Invoca o LLM e, se ele inventar o nome de uma tool, repete com os nomes exatos.

    O Groq valida os nomes das tools chamadas e devolve 400 se o modelo errar
    (ex.: 'reservar_ok' em vez de 'reserva_ok'). Em vez de derrubar a execução,
    damos uma segunda chance com os nomes certos bem na frente do modelo.
    """
    nomes = ", ".join(getattr(t, "name", str(t)) for t in ferramentas)
    try:
        return llm.invoke(messages)
    except Exception as erro:
        if "not in request.tools" not in str(erro):
            raise
        from langchain_core.messages import HumanMessage

        aviso = HumanMessage(
            content=(
                "Lembrete: os nomes EXATOS das ferramentas disponíveis são: "
                f"{nomes}. Chame apenas esses nomes, sem alterações."
            )
        )
        return llm.invoke([*messages, aviso])
