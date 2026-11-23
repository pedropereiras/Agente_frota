import sys

from src.graph.builder import build_graph
from src.utils.langfuse_client import flush, run_config


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(errors="replace")
    graph = build_graph()
    pergunta = input("Pergunta: ").strip()
    try:
        saida = graph.invoke(
            {"pergunta": pergunta},
            config=run_config(
                trace_name="porto-alfa-rag",
                tags=("porto-alfa", "rag"),
            ),
        )
        print(saida["resultado"])
    finally:
        flush()


if __name__ == "__main__":
    main()
