#!/usr/bin/env bash
# Todos los tests del proyecto, sin dependencias.
#   ./tests/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
run() {
  printf '\n\033[1m▸ %s\033[0m\n' "$1"
  shift
  if "$@"; then :; else fail=1; printf '  \033[31m✗ fallo\033[0m\n'; fi
}

run "motor JS (backprop, datasets, ias-mlp/1)"      node tests/test_core.js
run "UI real sobre un DOM falso"                     node tests/test_ui_smoke.js
run "paridad Python ↔ JavaScript"                    node tests/test_parity.js
run "verificación de gradiente en Python"            python3 python/ias_net.py --check
if command -v python3 >/dev/null 2>&1; then
  run "motor Python (unittest)"                      python3 -m unittest discover -s tests -p "test_net.py" -q
fi

printf '\n'
if [ "$fail" -eq 0 ]; then printf '\033[32m✔ todo el suite en verde\033[0m\n\n'; else printf '\033[31m✗ hay tests fallando\033[0m\n\n'; fi
exit $fail
