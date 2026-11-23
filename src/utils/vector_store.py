from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.utils.llm_adapter import get_embeddings

DADOS = Path(__file__).resolve().parents[2] / "dados"
ARQUIVOS = ("manual_operacoes.txt", "relatorio_frota.txt", "avaliacao.txt")


def carregar_chunks(arquivo: Path) -> list[Document]:
    """Divide o arquivo por seção (##), guardando 'doc' e 'secao' nos metadados."""
    texto = arquivo.read_text(encoding="utf-8")
    divisor = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "secao")], strip_headers=False
    )
    chunks = divisor.split_text(texto)
    for chunk in chunks:
        chunk.metadata["doc"] = arquivo.stem
    return chunks


@lru_cache #decorador do Python que guarda o resultado da função
def get_vector_store(arquivos: tuple[str, ...] = ARQUIVOS) -> InMemoryVectorStore: #indexa uma vez e reaproveita
    """Indexa os documentos uma única vez e reaproveita o índice."""
    store = InMemoryVectorStore(get_embeddings())
    for nome in arquivos:
        store.add_documents(carregar_chunks(DADOS / nome))
    return store


def buscar(pergunta: str, doc: str, k: int) -> list[Document]:
    """Busca por similaridade filtrando por metadado 'doc'."""
    return get_vector_store().similarity_search(
        pergunta, k=k, filter=lambda d: d.metadata.get("doc") == doc
    )


def fonte(d: Document) -> str:
    return f"{d.metadata['doc']}#{d.metadata.get('secao', '-')}"

