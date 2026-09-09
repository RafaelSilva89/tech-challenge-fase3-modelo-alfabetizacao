#!/usr/bin/env bash
# Orquestrador da pipeline da Fase 3 (executar dentro do WSL Ubuntu).
#
#   bash run.sh setup      instala o uv, cria o venv e as dependencias
#   bash run.sh camada     verifica que a modelagem so consome a Gold
#   bash run.sh base       constroi a base analitica lendo so da Gold
#   bash run.sh aluno      treina o modelo supervisionado por aluno
#   bash run.sh municipio  treina o modelo de risco municipal
#   bash run.sh notebooks  executa os tres notebooks de ponta a ponta
#   bash run.sh all        premissas -> gold -> camada -> base -> aluno -> municipio
#
# A camada Gold ja vem embarcada em data/gold/, entao o projeto roda sozinho: num
# clone limpo o caminho e `setup` -> `base` -> `aluno` -> `municipio`.
#
# Os dois alvos abaixo so fazem sentido com o data lake da Fase 2 disponivel, e se
# autopulam quando ele nao esta (por isso `all` funciona de qualquer forma):
#
#   bash run.sh premissas  revalida a materia-prima da Silver
#   bash run.sh gold       regenera as visoes Gold a partir da Silver
#
# Para apontar o data lake da Fase 2:
#   export FASE3_DATA_LAKE=/caminho/para/data_lake
#
# O venv fica no filesystem do Linux (~/.venvs/fase3) de proposito: o projeto vive
# em /mnt/c sob o OneDrive, onde milhares de arquivos de venv seriam lentos e
# entrariam na sincronizacao.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${FASE3_VENV:-$HOME/.venvs/fase3}"
PY="$VENV/bin/python"

# O LightGBM precisa de libgomp.so.1 (runtime OpenMP), que normalmente viria de
# "apt install libgomp1" -- indisponivel aqui porque o sudo pede senha. O wheel do
# scikit-learn ja distribui exatamente essa biblioteca, entao apontamos o loader
# para ela por symlink + LD_LIBRARY_PATH em vez de exigir privilegio de root.
link_libgomp() {
    local origem
    origem="$(find "$VENV/lib" -name 'libgomp*.so.1*' 2>/dev/null | head -1)"
    if [ -n "$origem" ]; then
        mkdir -p "$VENV/lib/gomp"
        ln -sf "$origem" "$VENV/lib/gomp/libgomp.so.1"
        echo ">> libgomp linkado a partir de $(basename "$origem")"
    else
        echo ">> AVISO: libgomp nao encontrado; o LightGBM sera ignorado na comparacao"
    fi
}
export LD_LIBRARY_PATH="$VENV/lib/gomp:${LD_LIBRARY_PATH:-}"

setup() {
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
        echo ">> instalando uv (binario standalone, sem sudo)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    echo ">> criando venv em $VENV"
    uv venv "$VENV" --python 3.12 --allow-existing
    echo ">> instalando dependencias"
    uv pip install --python "$PY" -r "$PROJ/requirements.txt"
    link_libgomp
    "$PY" -c "import duckdb, pandas, sklearn, lightgbm, shap; print('deps OK')"
}

run_module() { cd "$PROJ" && "$PY" -m "$1"; }

notebooks() {
    cd "$PROJ"
    "$PY" -m ipykernel install --user --name fase3 --display-name "Fase 3" >/dev/null 2>&1 || true
    for nb in 01_eda 02_modelagem_aluno 03_risco_municipal; do
        echo ">> executando notebooks/$nb.ipynb"
        "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=7200 "notebooks/$nb.ipynb"
    done
}

case "${1:-all}" in
    setup)      setup ;;
    premissas)  run_module src.data.verifica_premissas ;;
    gold)       run_module src.data.build_gold ;;
    camada)     run_module src.data.verifica_camada ;;
    base)       run_module src.data.build_base ;;
    aluno)      run_module src.models.train_aluno ;;
    municipio)  run_module src.models.train_municipio ;;
    notebooks)  notebooks ;;
    all)        run_module src.data.verifica_premissas
                run_module src.data.build_gold
                run_module src.data.verifica_camada
                run_module src.data.build_base
                run_module src.models.train_aluno
                run_module src.models.train_municipio ;;
    *)          echo "alvo desconhecido: $1"; exit 1 ;;
esac
