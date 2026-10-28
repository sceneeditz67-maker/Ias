#!/usr/bin/env python3
"""
ias_net.py — un multilayer perceptron completo en Python puro.

Cero dependencias: no necesita numpy, ni torch, ni `pip install` de nada.
Es el mismo motor (arquitectura, activaciones, Adam y BCE loss) que la
versión del navegador en `assets/app.js`, así que puedes entrenar la misma
red en la terminal y en el navegador y comparar los resultados.

Uso rápido
----------
    python3 python/ias_net.py                        # demo por defecto
    python3 python/ias_net.py --data spiral -H 8,6   # 2 capas ocultas
    python3 python/ias_net.py --watch                # entrenamiento animado
    python3 python/ias_net.py --check                # test de gradiente numérico
    python3 python/ias_net.py --export model.json    # para abrirlo en el navegador

Licencia MIT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from typing import Callable, Sequence

# --------------------------------------------------------------------------- #
# Activaciones
# --------------------------------------------------------------------------- #

_CLIP = 30.0  # evita overflow en exp()


def _c(v: float) -> float:
    return -_CLIP if v < -_CLIP else _CLIP if v > _CLIP else v


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-_c(z)))


#: nombre -> (f(z), f'(z) en función de z)
ACTIVATIONS: dict[str, tuple[Callable[[float], float], Callable[[float], float]]] = {
    "tanh": (lambda z: math.tanh(_c(z)), lambda z: 1.0 - math.tanh(_c(z)) ** 2),
    "relu": (lambda z: z if z > 0.0 else 0.0, lambda z: 1.0 if z > 0.0 else 0.0),
    "leaky": (lambda z: z if z > 0.0 else 0.01 * z, lambda z: 1.0 if z > 0.0 else 0.01),
    "elu": (lambda z: z if z > 0.0 else math.exp(_c(z)) - 1.0,
            lambda z: 1.0 if z > 0.0 else math.exp(_c(z))),
}

HIDDEN_ACTS = ("tanh", "relu", "leaky", "elu")


# --------------------------------------------------------------------------- #
# Capa densa
# --------------------------------------------------------------------------- #

class Dense:
    """W aplanada [n_out * n_in], sesgo [n_out], y los momentos de Adam."""

    __slots__ = ("n_in", "n_out", "act", "is_out", "w", "b",
                 "gw", "gb", "mw", "vw", "mb", "vb")

    def __init__(self, n_in: int, n_out: int, act: str = "tanh",
                 is_out: bool = False, rng: random.Random | None = None) -> None:
        self.n_in, self.n_out, self.act, self.is_out = n_in, n_out, act, is_out
        rng = rng or random.Random()
        scale = (math.sqrt(2.0 / n_in) if act in ("relu", "leaky", "elu")
                 else math.sqrt(1.0 / n_in))
        self.w = [rng.gauss(0.0, scale) for _ in range(n_out * n_in)]
        self.b = [0.0] * n_out
        self.gw = [0.0] * (n_out * n_in)
        self.gb = [0.0] * n_out
        self.mw = [0.0] * (n_out * n_in)
        self.vw = [0.0] * (n_out * n_in)
        self.mb = [0.0] * n_out
        self.vb = [0.0] * n_out

    @property
    def n_params(self) -> int:
        return len(self.w) + len(self.b)

    def forward(self, a_in: Sequence[float], z: list[float], a: list[float]) -> None:
        """Escribe z (pre-activación) y a (activación) en listas preasignadas."""
        n, w, b = self.n_in, self.w, self.b
        for o in range(self.n_out):
            acc = b[o]
            base = o * n
            for i in range(n):
                acc += w[base + i] * a_in[i]
            z[o] = acc
        if self.is_out:                       # salida: sigmoid (probabilidad)
            for o in range(self.n_out):
                a[o] = sigmoid(z[o])
        else:
            f = ACTIVATIONS[self.act][0]
            for o in range(self.n_out):
                a[o] = f(z[o])

    def step(self, lr: float, t: int, b1: float, b2: float, wd: float) -> None:
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
        for arr, g, m, v, decay in ((self.w, self.gw, self.mw, self.vw, wd),
                                    (self.b, self.gb, self.mb, self.vb, 0.0)):
            for i in range(len(arr)):
                gi = g[i] + (decay * arr[i] if decay else 0.0)
                m[i] = b1 * m[i] + (1.0 - b1) * gi
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi
                arr[i] -= lr * (m[i] / bc1) / (math.sqrt(v[i] / bc2) + 1e-8)

    def to_dict(self) -> dict:
        return {"nIn": self.n_in, "nOut": self.n_out,
                "w": [round(x, 8) for x in self.w],
                "b": [round(x, 8) for x in self.b]}


# --------------------------------------------------------------------------- #
# Red
# --------------------------------------------------------------------------- #

class MLP:
    """Red densa para clasificación binaria: 2 -> h1 -> ... -> 1."""

    def __init__(self, sizes: Sequence[int], hidden_act: str = "tanh",
                 seed: int | None = None, lr: float = 0.03,
                 weight_decay: float = 0.0) -> None:
        if len(sizes) < 2:
            raise ValueError("la red necesita al menos [n_in, n_out]")
        if sizes[0] != 2 or sizes[-1] != 1:
            raise ValueError("para esta demo: entrada 2 y salida 1")
        rng = random.Random(seed)
        self.rng = rng
        self.sizes = list(sizes)
        self.hidden_act = hidden_act
        self.lr = lr
        self.weight_decay = weight_decay
        last = len(sizes) - 2
        self.layers = [Dense(sizes[i], sizes[i + 1], hidden_act,
                             is_out=(i == last), rng=rng)
                       for i in range(last + 1)]
        self.t = 0
        self.history: list[tuple[float, float]] = []  # (loss, accuracy)

    @property
    def n_params(self) -> int:
        return sum(l.n_params for l in self.layers)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(self, x: Sequence[float], cache: list | None = None) -> float:
        """Devuelve p(y=1|x). Si pasas una lista, rellena el cache de backprop."""
        a_cur = list(x)
        for layer in self.layers:
            z = [0.0] * layer.n_out
            a_next = [0.0] * layer.n_out
            layer.forward(a_cur, z, a_next)
            if cache is not None:
                cache.append((a_cur, z, a_next))
            a_cur = a_next
        return a_cur[0]

    def predict(self, x: Sequence[float]) -> float:
        return self.forward(x)

    def predict_many(self, xs: Sequence[Sequence[float]]) -> list[float]:
        return [self.forward(x) for x in xs]

    # ------------------------------------------------------------------ #
    # backward + Adam
    # ------------------------------------------------------------------ #

    def gradient(self, x: Sequence[float], y: float) -> tuple[list[list[float]],
                                                               list[list[float]],
                                                               float, bool]:
        """
        Acumula el gradiente de BCE de UN ejemplo en layer.gw/gb.
        Devuelve (gw_por_capa, gb_por_capa, loss, correcto).
        """
        cache: list[tuple[list[float], list[float], list[float]]] = []
        p = self.forward(x, cache)
        pc = min(max(p, 1e-7), 1.0 - 1e-7)
        loss = -(y * math.log(pc) + (1.0 - y) * math.log(1.0 - pc))
        # Con sigmoid + BCE la derivada se colapsa a (p - y):
        delta = [p - y]
        gws, gbs = [], []
        for li in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[li]
            a_in, _z, _a_out = cache[li]
            n_in, n_out = layer.n_in, layer.n_out
            gw = [0.0] * (n_out * n_in)
            gb = [0.0] * n_out
            for o in range(n_out):
                d = delta[o]
                gb[o] += d
                base = o * n_in
                for i in range(n_in):
                    gw[base + i] += d * a_in[i]
            if li > 0:
                # dL/dz_{li-1} = (W_liᵀ · delta) ⊙ f'(z_{li-1})
                prev = self.layers[li - 1]
                z_prev = cache[li - 1][1]
                d_act = ACTIVATIONS[prev.act][1]
                back = [0.0] * n_in
                for o in range(n_out):
                    d = delta[o]
                    base = o * n_in
                    for i in range(n_in):
                        back[i] += d * layer.w[base + i]
                delta = [back[i] * d_act(z_prev[i]) for i in range(n_in)]
            gws.append(gw)
            gbs.append(gb)
        gws.reverse()
        gbs.reverse()
        for li, layer in enumerate(self.layers):
            gw, gb = gws[li], gbs[li]
            for i in range(len(gw)):
                layer.gw[i] += gw[i]
            for i in range(len(gb)):
                layer.gb[i] += gb[i]
        return gws, gbs, loss, (p >= 0.5) == (y >= 0.5)

    def clip_and_step(self, n_samples: int, clip: float = 5.0) -> None:
        """Escala el gradiente acumulado, lo recorta y aplica Adam."""
        k = 1.0 / max(n_samples, 1)
        gmax = 0.0
        for layer in self.layers:
            gw, gb = layer.gw, layer.gb
            for i in range(len(gw)):
                gw[i] *= k
                v = -gw[i] if gw[i] < 0.0 else gw[i]
                if v > gmax:
                    gmax = v
            for i in range(len(gb)):
                gb[i] *= k
                v = -gb[i] if gb[i] < 0.0 else gb[i]
                if v > gmax:
                    gmax = v
        if clip and gmax > clip:
            s = clip / gmax
            for layer in self.layers:
                for i in range(len(layer.gw)):
                    layer.gw[i] *= s
                for i in range(len(layer.gb)):
                    layer.gb[i] *= s
        self.t += 1
        for layer in self.layers:
            layer.step(self.lr, self.t, 0.9, 0.999, self.weight_decay)
        for layer in self.layers:
            for i in range(len(layer.gw)):
                layer.gw[i] = 0.0
            for i in range(len(layer.gb)):
                layer.gb[i] = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float],
            epochs: int = 1, batch: int | None = None, clip: float = 5.0,
            on_epoch: Callable[[int, float, float], None] | None = None) -> tuple[float, float]:
        """Entrena `epochs` épocas (minibatch SGD-Adam). Devuelve (loss, acc)."""
        n = len(X)
        if n == 0:
            raise ValueError("sin datos")
        if n != len(y):
            raise ValueError("X e y de tamaños distintos")
        idx = list(range(n))
        size = batch or n
        loss, acc = float("nan"), float("nan")
        for ep in range(1, epochs + 1):
            self.rng.shuffle(idx)
            correct = 0
            loss_sum = 0.0
            for start in range(0, n, size):
                chunk = idx[start:start + size]
                for j in chunk:
                    _, _, l, ok = self.gradient(X[j], y[j])
                    loss_sum += l
                    correct += 1 if ok else 0
                self.clip_and_step(len(chunk), clip)
            loss, acc = loss_sum / n, correct / n
            self.history.append((loss, acc))
            if on_epoch:
                on_epoch(len(self.history), loss, acc)
        return loss, acc

    # ------------------------------------------------------------------ #
    # (de)serialización
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {"format": "ias-mlp/1", "sizes": self.sizes,
                "hiddenAct": self.hidden_act, "lr": self.lr,
                "layers": [l.to_dict() for l in self.layers]}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        net = cls(d["sizes"], d.get("hiddenAct", "tanh"))
        for layer, src in zip(net.layers, d["layers"]):
            if (layer.n_in, layer.n_out) != (src["nIn"], src["nOut"]):
                raise ValueError("forma de capa incompatible")
            layer.w = [float(v) for v in src["w"]]
            layer.b = [float(v) for v in src["b"]]
        net.lr = float(d.get("lr", net.lr))
        return net

    # ------------------------------------------------------------------ #
    # utilidad: gradiente de la entrada (mapa de sensibilidad)
    # ------------------------------------------------------------------ #

    def grad_input(self, x: Sequence[float]) -> tuple[float, list[float]]:
        """∂p/∂x por diferenciación central (útil y barato con 2 entradas)."""
        h = 1e-4
        p0 = self.forward(x)
        gx = []
        for i in range(len(x)):
            xp = list(x)
            xp[i] += h
            xm = list(x)
            xm[i] -= h
            gx.append((self.forward(xp) - self.forward(xm)) / (2 * h))
        return p0, gx


# --------------------------------------------------------------------------- #
# Conjuntos de datos sintéticos
# --------------------------------------------------------------------------- #

DATASETS = ("blobs", "xor", "circle", "moons", "spiral", "checker", "sine", "rings3")


def _gauss(rng: random.Random) -> tuple[float, float]:
    u = max(rng.random(), 1e-12)
    r = math.sqrt(-2.0 * math.log(u))
    t = 2.0 * math.pi * rng.random()
    return r * math.cos(t), r * math.sin(t)


def make_dataset(name: str, n: int = 240, noise: float = 0.07,
                 seed: int = 7) -> tuple[list[list[float]], list[float]]:
    """Genera puntos en [-1, 1]² con etiqueta 0/1. Mismo algoritmo que el JS."""
    rng = random.Random(seed)
    n2 = n // 2
    X: list[list[float]] = []
    y: list[float] = []

    def push(px: float, py: float, label: float) -> None:
        gx, gy = _gauss(rng)
        X.append([max(-1.35, min(1.35, px + gx * noise)),
                  max(-1.35, min(1.35, py + gy * noise))])
        y.append(label)

    if name == "blobs":
        for k in range(2):
            cx = -0.45 if k == 0 else 0.45
            for _ in range(n2):
                gx, gy = _gauss(rng)
                push(cx + gx * 0.25, gy * 0.25, float(k))
    elif name == "xor":
        for sx in (-1, 1):
            for sy in (-1, 1):
                lbl = 1.0 if sx * sy > 0 else 0.0
                for _ in range(n // 4):
                    gx, gy = _gauss(rng)
                    push(sx * 0.45 + gx * 0.2, sy * 0.45 + gy * 0.2, lbl)
    elif name in ("circle", "rings3"):
        radii = (0.28, 0.72) if name == "circle" else (0.2, 0.52, 0.85)
        per = n // len(radii)
        for k, r in enumerate(radii):
            for i in range(per):
                t = 2 * math.pi * i / per + rng.random() * 0.35
                push(r * math.cos(t), r * math.sin(t), float(k % 2))
    elif name == "moons":
        for i in range(n2):
            t = math.pi * i / n2
            push(math.cos(t) * 0.7, math.sin(t) * 0.55 - 0.22, 0.0)
        for i in range(n2):
            t = math.pi * i / n2
            push(0.7 - math.cos(t) * 0.7, 0.22 - math.sin(t) * 0.55, 1.0)
    elif name == "spiral":
        arms = 2
        per = n // arms
        for k in range(arms):
            for i in range(per):
                t = i / per
                ang = 3.4 * math.pi * t + k * math.pi / arms
                r = 0.14 + 0.78 * t
                push(r * math.cos(ang), r * math.sin(ang), float(k % 2))
    elif name == "checker":
        cells = 4
        for k in range(cells * cells):
            cx = (k % cells) / cells * 2 - 1 + 1 / cells
            cy = (k // cells) / cells * 2 - 1 + 1 / cells
            lbl = 1.0 if ((k % cells) + (k // cells)) % 2 == 0 else 0.0
            for _ in range(max(4, n // (cells * cells))):
                gx, gy = _gauss(rng)
                push(cx + gx * 0.16, cy + gy * 0.16, lbl)
    elif name == "sine":
        for i in range(n2):
            px = -1.0 + 2.0 * i / n2
            push(px, 0.55 * math.sin(px * 3.6) - 0.2, 0.0)
        for i in range(n2):
            px = -1.0 + 2.0 * i / n2
            push(px, 0.55 * math.sin(px * 3.6) + 0.2, 1.0)
    else:
        raise ValueError(f"dataset desconocido: {name}")
    return X, y


# --------------------------------------------------------------------------- #
# Render ASCII
# --------------------------------------------------------------------------- #

SHADES = " .:-=+*#%@"


def sparkline(values: Sequence[float], width: int = 46, height: int = 7) -> list[str]:
    """Minigráfico en caracteres para la curva de pérdida."""
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return [" (esperando datos…)"][:1] * height
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    grid = [[" "] * width for _ in range(height)]
    for x in range(width):
        i = round(x * (len(vals) - 1) / (width - 1))
        v = (vals[i] - lo) / (hi - lo)
        y = round((1.0 - v) * (height - 1))
        grid[max(0, min(height - 1, y))][x] = "▂"
    grid[height - 1] = ["─"] * width
    for x in range(width):
        i = round(x * (len(vals) - 1) / (width - 1))
        v = (vals[i] - lo) / (hi - lo)
        filled = round((1.0 - v) * (height - 1)) + 1
        for yy in range(filled, height):
            if grid[yy][x] == " ":
                grid[yy][x] = "·"
    return ["".join(row).rstrip() for row in grid]


def boundary_map(net: MLP, cols: int = 62, rows: int = 27,
                 X: Sequence[Sequence[float]] | None = None,
                 y: Sequence[float] | None = None) -> list[str]:
    """Mapa de decisión en caracteres: fondo = probabilidad, puntos = datos."""
    out: list[str] = []
    pts: dict[tuple[int, int], str] = {}
    if X:
        for i, (px, py) in enumerate(X):
            cx = int((px + 1.3) / 2.6 * (cols - 1))
            cy = int((1.3 - py) / 2.6 * (rows - 1))
            if 0 <= cx < cols and 0 <= cy < rows:
                pts[(cy, cx)] = "O" if (y and y[i] >= 0.5) else "x"
    for r in range(rows):
        wy = 1.3 - (r / (rows - 1)) * 2.6
        line: list[str] = []
        for c in range(cols):
            wx = -1.3 + (c / (cols - 1)) * 2.6
            p = net.forward((wx, wy))
            if (r, c) in pts:
                line.append(pts[(r, c)])
                continue
            k = int(round(p * (len(SHADES) - 1)))
            line.append(SHADES[max(0, min(len(SHADES) - 1, k))])
        out.append("".join(line))
    return out


def fmt(v: float, w: int = 6, d: int = 4) -> str:
    return f"{v:>{w}.{d}f}" if v == v else " " * w


# --------------------------------------------------------------------------- #
# Test de gradiente
# --------------------------------------------------------------------------- #

def gradient_check(sizes: Sequence[int] = (2, 5, 4, 1), seed: int = 3) -> bool:
    """Compara el gradiente analítico con el numérico (diferencias centrales)."""
    net = MLP(sizes, "tanh", seed=seed)
    xs, ys = make_dataset("xor", 16, 0.05, seed)
    gws, gbs, _, _ = net.gradient(xs[0], ys[0])
    eps = 1e-5
    worst = 0.0
    layer0 = net.layers[0]
    num = [0.0] * len(layer0.w)
    for i in range(len(layer0.w)):
        orig = layer0.w[i]
        layer0.w[i] = orig + eps
        lp = net.gradient(xs[0], ys[0])[2]
        layer0.w[i] = orig - eps
        lm = net.gradient(xs[0], ys[0])[2]
        layer0.w[i] = orig
        num[i] = (lp - lm) / (2 * eps)
    ana = gws[0]
    for i in range(len(num)):
        denom = max(1e-8, abs(num[i]) + abs(ana[i]))
        worst = max(worst, abs(num[i] - ana[i]) / denom)
    net.gradient(xs[0], ys[0])  # deja los buffers en un estado neutro
    print(f"  gradiente de la capa 0 (analítico vs numérico): error relativo máx = {worst:.2e}")
    return worst < 1e-3


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_hidden(text: str) -> list[int]:
    if not text:
        return [8, 6]
    vals = [int(t) for t in text.replace(" ", "").split(",") if t]
    return vals or [8]


def banner() -> str:
    return (
        "  ██╗  █████╗  ███████╗\n"
        "  ██║ ██╔══██║ ██╔═════╝   laboratorio de redes neuronales\n"
        "  ██║ ███████║ ███████╗    python puro · sin dependencias\n"
        " ░██║ ██╔══██║ ╚════██║\n"
        " █████║ ██  ██║ ███████║    ias_net.py\n"
        " ╚════╝ ╚═╝  ╚═╝ ╚══════╝\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entrena una red neuronal sin dependencias.")
    ap.add_argument("--data", choices=DATASETS, default="spiral", help="dataset sintético")
    ap.add_argument("-n", "--points", type=int, default=240, help="nº de puntos")
    ap.add_argument("-H", "--hidden", default="8,6", help="capas ocultas, ej. '12,8'")
    ap.add_argument("--act", choices=HIDDEN_ACTS, default="tanh", help="activación oculta")
    ap.add_argument("--epochs", type=int, default=600, help="épocas totales")
    ap.add_argument("--lr", type=float, default=0.03, help="tasa de aprendizaje")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay (L2)")
    ap.add_argument("--batch", type=int, default=0, help="tamaño de minibatch (0 = todos)")
    ap.add_argument("--noise", type=float, default=0.07, help="ruido gaussiano")
    ap.add_argument("--seed", type=int, default=7, help="semilla aleatoria")
    ap.add_argument("--ascii", dest="ascii_map", action="store_true", default=True,
                    help="imprime el mapa de decisión ASCII (viene activado)")
    ap.add_argument("--no-map", dest="ascii_map", action="store_false",
                    help="omite el mapa de decisión")
    ap.add_argument("--watch", action="store_true", help="animación en la terminal")
    ap.add_argument("--fps", type=float, default=22.0, help="fotogramas/s en --watch")
    ap.add_argument("--check", action="store_true", help="test de gradiente numérico y sale")
    ap.add_argument("--export", metavar="ARCHIVO", help="guarda los pesos en JSON")
    ap.add_argument("--quiet", action="store_true", help="solo la línea final")
    ap.add_argument("--plain", action="store_true", help="sin mapa de decisión (alias de --no-map)")
    args = ap.parse_args(argv)

    if args.check:
        print("  verificando backprop contra diferencias centrales…")
        ok = gradient_check()
        print("  ✔ backprop correcto" if ok else "  ✗ gradiente inconsistente")
        return 0 if ok else 1

    sizes = [2, *parse_hidden(args.hidden), 1]
    X, y = make_dataset(args.data, args.points, args.noise, args.seed)
    net = MLP(sizes, args.act, seed=args.seed, lr=args.lr, weight_decay=args.wd)

    if args.plain:
        args.ascii_map = False

    if not args.quiet:
        print(banner())
        print(f"  dataset={args.data}  puntos={len(X)}  red={sizes}  "
              f"params={net.n_params}  act={args.act}  lr={args.lr}")
        print("  " + "─" * 66)

    every = max(1, args.epochs // (40 if args.watch else 12))
    batch = args.batch or 0
    t0 = time.time()

    def render(epoch: int, loss: float, acc: float, clear: bool) -> None:
        if args.quiet:
            return
        block: list[str] = []
        if clear and not args.quiet:      # en modo animado hay que repintar todo
            block += banner().rstrip("\n").split("\n")
        p0 = acc * 100
        bar = "█" * int(p0 / 3.6)
        block.append(f"  época {epoch:>5}/{args.epochs}   pérdida {fmt(loss)}   "
                     f"exactitud {p0:5.1f}%")
        block.append(f"  [{bar:<28}▏]")
        if args.ascii_map:
            block.append("")
            block += ["    " + r for r in boundary_map(net)]
            block.append("")
            block += ["    " + r for r in sparkline([h[0] for h in net.history])]
            block.append(f"    pérdida (mín {min(h[0] for h in net.history):.4f}) · "
                         f"eje vertical = época →")
        txt = "\n".join(block)
        if clear:
            sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(txt + "\n")
        sys.stdout.flush()

    if args.watch:
        delay = 1.0 / max(1.0, args.fps)
        epoch = 0
        try:
            while epoch < args.epochs:
                loss, acc = net.fit(X, y, epochs=every, batch=batch or None)
                epoch += every
                render(min(epoch, args.epochs), loss, acc, clear=True)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\n  interrumpido por el usuario")
    else:
        for start in range(0, args.epochs, every):
            step = min(every, args.epochs - start)
            loss, acc = net.fit(X, y, epochs=step, batch=batch or None)
            render(start + step, loss, acc, clear=False)

    secs = time.time() - t0
    loss, acc = net.history[-1][0], net.history[-1][1]
    print("  " + "─" * 66)
    print(f"  ✔ listo en {secs:.1f}s · {net.t} actualizaciones · "
          f"pérdida final {loss:.4f} · exactitud {acc * 100:.1f}%")

    if args.export:
        payload = net.to_dict()
        payload["dataset"] = {"name": args.data, "seed": args.seed,
                              "noise": args.noise, "points": args.points,
                              "X": [[round(a, 5), round(b, 5)] for a, b in X],
                              "y": y}
        payload["epochs"] = args.epochs
        # predicciones de control: permiten comprobar la paridad con el motor JS
        probes = []
        for gy in range(5):
            for gx in range(7):
                # se redondea ANTES de predecir: así quien compare el JSON
                # evalúa exactamente el mismo punto que nosotros
                px = round(-1 + 2 * gx / 6, 5)
                py = round(-1 + 2 * gy / 4, 5)
                probes.append([px, py, round(net.forward((px, py)), 10)])
        payload["probes"] = probes
        with open(args.export, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"  ↪ pesos exportados a {args.export} — ábrelos en el navegador "
              f"(panel «Importar modelo»)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
