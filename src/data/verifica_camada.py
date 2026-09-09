"""Trava de arquitetura: garante que a modelagem só consuma a camada Gold.

O enunciado da Fase 3 determina que os dados venham da camada Gold, e a arquitetura
medalhão diz a mesma coisa por outro caminho: o consumidor analítico lê a camada de
consumo, não volta à intermediária. É fácil furar essa regra sem perceber — basta
alguém precisar de uma coluna que só existe na Silver e apontar para lá.

Este teste falha se isso acontecer. Dois módulos podem ler a Silver, e só eles:

  · build_gold.py        — transforma Silver em Gold; é o seu trabalho
  · verifica_premissas.py — valida a matéria-prima antes de a Gold existir

Roda em segundos, sem tocar em dado. `bash run.sh camada`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import PROJ_DIR, get_logger

log = get_logger("camada")

PODEM_LER_SILVER = {"build_gold.py", "verifica_premissas.py"}
MARCADORES = ("SILVER_ALUNOS", "SILVER_INDICADORES", "/silver/")


def main() -> int:
    infratores: list[tuple[Path, int, str]] = []
    analisados = 0

    for arquivo in sorted((PROJ_DIR / "src").rglob("*.py")):
        if arquivo.name == Path(__file__).name:
            continue
        analisados += 1
        # config.py declara os caminhos da Silver para os dois módulos autorizados;
        # declarar não é consumir.
        if arquivo.name in PODEM_LER_SILVER or arquivo.name == "config.py":
            continue
        for n, linha in enumerate(arquivo.read_text(encoding="utf-8").splitlines(), 1):
            if any(m in linha for m in MARCADORES):
                infratores.append((arquivo.relative_to(PROJ_DIR), n, linha.strip()))

    log.info("=" * 66)
    log.info(f"TRAVA DE CAMADA — {analisados} modulos analisados")
    log.info(f"   autorizados a ler a Silver: {', '.join(sorted(PODEM_LER_SILVER))}")

    if infratores:
        log.error(f"   {len(infratores)} leitura(s) indevida(s) da Silver:")
        for caminho, n, linha in infratores:
            log.error(f"     {caminho}:{n}  {linha[:70]}")
        log.error("   a modelagem deve consumir a Gold; se falta uma coluna la,")
        log.error("   acrescente-a em build_gold.py em vez de voltar a Silver.")
        return 1

    log.info("   OK — nenhum modulo de modelagem le a Silver")
    log.info("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
