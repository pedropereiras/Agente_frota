from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.graph.nodes.analyze import analyze
from src.graph.nodes.classify import classify
from src.graph.nodes.extract_frota import extract_frota
from src.graph.nodes.format_frota import format_frota
from src.graph.nodes.format_output import format_output
from src.graph.nodes.generate import generate
from src.graph.nodes.grade import grade
from src.graph.nodes.nao_encontrado import nao_encontrado
from src.graph.nodes.prepare import prepare
from src.graph.nodes.refuse import refuse
from src.graph.nodes.retrieve_frota import retrieve_frota
from src.graph.nodes.retrieve_regras import retrieve_regras
from src.graph.nodes.retrieve_regras_calculo import retrieve_regras_calculo
from src.graph.nodes.verify_grounding import verify_grounding
from src.graph.routing import route_after_grade, route_by_intent
from src.graph.state import State
from src.graph.tools import TOOLS


def build_graph():
    """Grafo completo do exercício 2: roteamento + RAG + tools."""
    graph = StateGraph(State)
    for nome, funcao in {
        "classify": classify,
        "retrieve_regras": retrieve_regras,
        "grade": grade,
        "generate": generate,
        "verify_grounding": verify_grounding,
        "format_output": format_output,
        "nao_encontrado": nao_encontrado,
        "retrieve_frota": retrieve_frota,
        "retrieve_regras_calculo": retrieve_regras_calculo,
        "extract_frota": extract_frota,
        "prepare": prepare,
        "analyze": analyze,
        "format_frota": format_frota,
        "refuse": refuse,
    }.items():
        graph.add_node(nome, funcao)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {"regra": "retrieve_regras", "frota": "retrieve_frota", "fora_escopo": "refuse"},
    )
    graph.add_edge("retrieve_regras", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"suficiente": "generate", "insuficiente": "nao_encontrado"}
    )
    graph.add_edge("generate", "verify_grounding")
    graph.add_edge("verify_grounding", "format_output")
    graph.add_edge("retrieve_frota", "retrieve_regras_calculo")
    graph.add_edge("retrieve_regras_calculo", "extract_frota")
    graph.add_edge("extract_frota", "prepare")
    graph.add_edge("prepare", "analyze")
    graph.add_conditional_edges(
        "analyze", tools_condition, {"tools": "tools", END: "format_frota"}
    )
    graph.add_edge("tools", "analyze")
    for terminal in ("format_output", "nao_encontrado", "format_frota", "refuse"):
        graph.add_edge(terminal, END)
    return graph.compile()
