from langchain_core.tools import tool


@tool
def calcular_eta_horas(distancia_nm: float, velocidade_kn: float) -> float:
    """Calcula o ETA em horas: distancia_nm dividida por velocidade_kn."""
    return round(distancia_nm / velocidade_kn, 2)


@tool
def velocidade_efetiva(
    velocidade_kn: float, em_tempestade: bool, limite_tempestade_kn: float
) -> float:
    """Velocidade efetiva. Em tempestade, é no máximo o limite de tempestade do manual."""
    return min(velocidade_kn, limite_tempestade_kn) if em_tempestade else velocidade_kn


@tool
def reserva_ok(combustivel_t: float, reserva_t: float) -> bool:
    """True se o combustível a bordo cobre a reserva mínima (navio pode zarpar)."""
    return combustivel_t >= reserva_t


TOOLS = [calcular_eta_horas, velocidade_efetiva, reserva_ok]
