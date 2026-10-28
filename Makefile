# Atajos del laboratorio (todo opcional: el proyecto funciona sin Make).
PY ?= python3

serve:            ## sirve el laboratorio en http://localhost:8000
	$(PY) -m http.server 8000 --bind 0.0.0.0

demo:             ## entrenamiento animado en la terminal
	$(PY) python/ias_net.py --data spiral -H 8,6 --epochs 600 --watch

check:            ## comprueba el backprop contra gradientes numéricos
	$(PY) python/ias_net.py --check

test:             ## todos los tests (JS, UI, paridad, Python)
	./tests/run_all.sh

clean:
	rm -rf __pycache__ python/__pycache__ tests/__pycache__
	find . -name '*.pyc' -delete

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

.PHONY: serve demo check test clean help
