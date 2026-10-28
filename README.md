# IAS · laboratorio de redes neuronales

> Sorpresa: un playground donde **entrenas una red neuronal de verdad** dibujando puntos con el ratón, y ves la frontera de decisión formarse fotograma a fotograma. Sin TensorFlow, sin numpy, sin `npm install`, sin build.

Tres piezas, mismo motor:

| | qué es | cómo se abre |
|---|---|---|
| 🌐 **`index.html`** | el laboratorio interactivo (canvas + red en vivo) | GitHub Pages, o `make serve` |
| ⌨️ **`python/ias_net.py`** | el mismo MLP corriendo en tu terminal, con mapa ASCII y animación | `make demo` |
| 🧪 **`tests/`** | 77 comprobaciones: gradiente analítico vs numérico, UI pilotada en Node, paridad entre lenguajes | `make test` |

---

## Abrirlo

```bash
git clone https://github.com/sceneeditz67-maker/Ias && cd Ias
python3 -m http.server 8000        # → http://localhost:8000
```

O sin terminal: activa **Settings → Pages → Deploy from a branch → `main` / `root`** y abre la URL que GitHub te dé (tarda un minuto la primera vez). El `index.html` también funciona con doble clic desde el disco: no hay módulos ES ni `fetch`, así que `file://` va igual.

## Jugar 60 segundos

1. Arranca con la **espiral** entrenando a tope: verás el mapa de colores (la probabilidad que la red asigna a cada punto del plano) y la línea blanca, que es su frontera de decisión.
2. Pulsa **⏸ pausa**, pasa el cursor por el plano → el diagrama de la derecha te muestra **qué neurona se activa** para ese punto exacto.
3. Elige **a mano** y pinta: clic = clase A, `2` cambia al pincel B, `3` o Alt+clic borra. Reanuda y mira cómo la red se adapta a *tus* datos.
4. Quita capas ocultas con la ✕ de los chips hasta dejar la red **sin capa oculta** y carga **XOR**: se queda clavada en ~50 %. No es un bug, es un teorema — un modelo lineal no puede separar el XOR.
5. Sube la **tasa de aprendizaje** al máximo: la pérdida revienta y la frontera tiembla. Baja el **minibatch** y verás el ruido del SGD.
6. `d` cambia de dataset, `.` avanza una época, `⎘ enlace del estado` copia una URL que reproduce exactamente lo que estás viendo.

Lo que hay debajo del capó, en una línea: `p = σ(w₂·tanh(w₁·x + b₁) + b₂)`, minimizando `L = −[y·log p + (1−y)·log(1−p)]` con **Adam** (β₁ 0.9, β₂ 0.999, peso recortado por norma ∞ a 5.0).

## La terminal también tiene su versión

```bash
python3 python/ias_net.py --data circle -H 10,8 --epochs 420
```

```
  época   420/420   pérdida 0.0018   exactitud 100.0%
  [███████████████████████████ ▏]

    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@*-:..:-*%@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@:             .+%@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@.                 .*@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@-                    :#@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@%:                     .#@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@+.                     .%@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@#:                     -@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@%=.                  :%@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@*.               -%@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@+           .#@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@+.    .+%@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    ▂▂
    ··▂
    ···
    ···▂
    ····▂
    ·····▂
    ──────────────────────────────────────────────
    pérdida (mín 0.0018) · eje vertical = época →
  ──────────────────────────────────────────────────────────────────
  ✔ listo en 3.2s · 420 actualizaciones · pérdida final 0.0018 · exactitud 100.0%
```

La red aprendió que dentro del anillo hay una clase y fuera otra, y lo dibujó con caracteres.

```bash
python3 python/ias_net.py --data spiral -H 8,6 --watch    # animación en vivo
python3 python/ias_net.py --check                          # ¿es correcto el backprop?
python3 python/ias_net.py --help
```

## El puente entre los dos mundos

Ambos motores hablan un formato de pesos llamado `ias-mlp/1`:

```bash
python3 python/ias_net.py --data xor -H 6,4 --epochs 260 --quiet --export modelo.json
```

…y en el navegador, panel **visualización → ↑ importar .json**: la red del navegador carga los pesos de Python y *sigue entrenando desde ahí*, con los mismos puntos. Hay un test que verifica que las dos implementaciones devuelven literalmente la misma probabilidad (Δ < 1,3·10⁻⁸ en 35 puntos de control, solo el ruido del float y del JSON con 8 decimales).

## Qué hay en el repo

```
index.html              304 líneas · estructura y textos
assets/core.js          550 · el motor: forward, backprop, Adam, datasets, marching squares
assets/app.js          1089 · bucle de fotogramas, canvas, controles, pintura, import/export
assets/styles.css        401 · tema oscuro, fondo ambiental, responsive
python/ias_net.py        645 · el mismo motor en Python puro + CLI + render ASCII
tests/                    77 comprobaciones en 4 suites (ver abajo)
Makefile                  atajos: make serve | demo | check | test | help
```

## Los tests

```bash
make test          # o ./tests/run_all.sh
```

| suite | qué demuestra |
|---|---|
| `tests/test_core.js` (24) | el gradiente de cada capa coincide con el numérico (tanh, relu, leaky, elu); XOR se aprende con capa oculta y **no** sin ella; los datasets son deterministas; el JSON `ias-mlp/1` viaja de vuelta sin perder precisión |
| `tests/test_ui_smoke.js` (30) | la interfaz real arranca en un DOM falso, entrena, cada botón/slider/tecla responde sin excepciones, pintar añade y borra puntos, y el estado sobrevive a `localStorage` y al enlace compartible |
| `tests/test_parity.js` (8) | Python entrena → exporta → el navegador importa → mismo resultado, y también a la inversa |
| `tests/test_net.py` (15) | lo mismo sobre el motor Python, más `--check`, el render ASCII y la CLI |

`tests/mini-dom.js` es un DOM en miniatura (355 líneas) que parsea tu `index.html` real, simula el burbujeo de eventos y sustituye el contexto 2D por un sustituto que registra las llamadas. Ya ha merecido la pena: detectó una colisión de identificadores entre `core.js` y `app.js` que habría roto la página en el navegador, y un redondeo en el formato de exportación que descuadraba las predicciones en 1·10⁻⁴.

## Detalles con los que me lo pasé

- **La frontera no es un truco de dibujo**: `isoSegments()` hace *marching squares* de verdad sobre la rejilla de 104×N donde la red se evalúa cada 2 fotogramas, con `p = 0.5` (y opcionales 0.1 / 0.9).
- **Los puntos mal clasificados se marcan** con un halo rojo: es la señal más rápida de que algo va mal.
- **Train/test de verdad**: un % de los puntos se reserva y nunca entra en el `fit`; si la línea naranja punteada se despega de la verde, estás sobrentrenando.
- **Flujo de señal animado** por las aristas de la arquitectura, con grosor y color según el signo y la magnitud del peso.
- **Pinta y aprende a la vez**: el bucle puede cortar una época a mitad si el fotograma amenaza con pasarse de 11 ms, así el ratón nunca se queda colgado aunque entrenes 800 puntos con [16,16,16].
- **Todo cabe en la URL**: `#ds=checker&hidden=12.10&lr=0.01&…` reconstruye dataset, arquitectura e hiperparámetros.
- El `localStorage` guarda tus puntos dibujados, así que recargar no borra tu obra.

## Notas

- Sin dependencias, sin analítica, sin fuentes externas: se puede copiar la carpeta a un USB y funciona.
- Probado en Chrome/Firefox/Safari modernos (canvas 2D, Pointer Events, `Float64Array`). Python ≥ 3.9 para el módulo de terminal.
- Nada de esto es «inteligencia artificial» de ciencia ficción: es una red densa de dos entradas y una salida, perfecta para **ver** qué significan el gradiente, la capacidad y la generalización.

Licencia MIT — haz lo que quieras con él.
