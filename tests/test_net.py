"""Tests del motor en Python puro (sin dependencias).

    python3 -m unittest discover -s tests -p "test_*.py" -v
"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "python"))

import ias_net as I  # noqa: E402


class TestGradientes(unittest.TestCase):
    """El gradiente analítico tiene que ser el numérico (diferencias centrales)."""

    @staticmethod
    def _loss(net, x, y):
        p = min(max(net.forward(x), 1e-7), 1.0 - 1e-7)
        return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

    def test_backprop_por_activacion(self):
        for act in ("tanh", "relu", "leaky", "elu"):
            with self.subTest(act=act):
                net = I.MLP([2, 5, 4, 1], act, seed=3)
                xs, ys = I.make_dataset("xor", 8, 0.0, seed=1)
                worst = 0.0
                for k in range(3):
                    x, y = xs[k], ys[k]
                    gws, _, _, _ = net.gradient(x, y)
                    for li, layer in enumerate(net.layers):
                        eps, step = 1e-6, max(1, len(layer.w) // 6)
                        for i in range(0, len(layer.w), step):
                            save = layer.w[i]
                            layer.w[i] = save + eps
                            up = self._loss(net, x, y)
                            layer.w[i] = save - eps
                            dn = self._loss(net, x, y)
                            layer.w[i] = save
                            num = (up - dn) / (2 * eps)
                            den = max(1e-8, abs(num) + abs(gws[li][i]))
                            worst = max(worst, abs(num - gws[li][i]) / den)
                self.assertLess(worst, 5e-3)

    def test_derivada_de_la_entrada(self):
        net = I.MLP([2, 4, 1], "tanh", seed=1)
        p, g = net.grad_input((0.3, -0.2))
        self.assertTrue(0.0 < p < 1.0)
        self.assertEqual(len(g), 2)
        self.assertTrue(all(math.isfinite(v) for v in g))
        # coherencia con la fórmula cerrada de una capa lineal-sigmoid
        lin = I.MLP([2, 1], "tanh", seed=4)
        p2, g2 = lin.grad_input((0.5, 0.5))
        w = lin.layers[0].w
        self.assertAlmostEqual(g2[0], p2 * (1 - p2) * w[0], places=6)


class TestAprendizaje(unittest.TestCase):
    def _train(self, sizes, name, epochs=200, lr=0.04, n=160, batch=32, seed=7):
        X, y = I.make_dataset(name, n, 0.06, seed)
        net = I.MLP(sizes, "tanh", seed=seed, lr=lr)
        net.fit(X, y, epochs=epochs, batch=batch)
        return net, y, X

    def test_xor_con_capa_oculta(self):
        net, y, X = self._train([2, 8, 1], "xor")
        good = sum(1 for x, t in zip(X, y) if (net.forward(x) >= 0.5) == (t >= 0.5))
        self.assertGreater(good / len(y), 0.9)

    def test_xor_lineal_debe_fracasar(self):
        net, y, X = self._train([2, 1], "xor", epochs=150)
        good = sum(1 for x, t in zip(X, y) if (net.forward(x) >= 0.5) == (t >= 0.5))
        self.assertLess(good / len(y), 0.8)

    def test_la_perdida_baja(self):
        net, _, _ = self._train([2, 10, 8, 1], "circle", epochs=120)
        first = min(h[0] for h in net.history[:5])
        last = min(h[0] for h in net.history[-5:])
        self.assertLess(last, first)

    def test_historial_registrado(self):
        net, _, _ = self._train([2, 6, 1], "blobs", epochs=25)
        self.assertEqual(len(net.history), 25)
        self.assertTrue(all(math.isfinite(l) and 0.0 <= a <= 1.0 for l, a in net.history))


class TestDatos(unittest.TestCase):
    def test_todos_los_datasets(self):
        for name in I.DATASETS:
            with self.subTest(data=name):
                X, y = I.make_dataset(name, 200, 0.07, 5)
                self.assertGreater(len(X), 20)
                self.assertTrue(all(abs(v) <= 1.37 for p in X for v in p))
                self.assertIn(0.0, y)
                self.assertIn(1.0, y)

    def test_determinismo(self):
        a = I.make_dataset("spiral", 120, 0.05, 11)
        b = I.make_dataset("spiral", 120, 0.05, 11)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0][0], I.make_dataset("spiral", 120, 0.05, 12)[0][0])


class TestSerializacion(unittest.TestCase):
    def test_round_trip(self):
        net = I.MLP([2, 6, 3, 1], "tanh", seed=8)
        d = net.to_dict()
        self.assertEqual(d["format"], "ias-mlp/1")
        back = I.MLP.from_dict(d)
        for x, y in ((0.2, 0.7), (-0.9, 0.4), (0.0, 0.0)):
            self.assertAlmostEqual(net.forward((x, y)), back.forward((x, y)), places=5)
        self.assertEqual(net.n_params, back.n_params)

    def test_forma_incorrecta(self):
        with self.assertRaises(Exception):
            I.MLP.from_dict({"sizes": [2, 4, 1], "layers": [{"nIn": 2, "nOut": 9, "w": [], "b": []}]})


class TestRender(unittest.TestCase):
    def test_mapa_de_decision(self):
        net = I.MLP([2, 5, 1], "tanh", seed=2)
        rows = I.boundary_map(net, cols=30, rows=12)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({len(r) for r in rows}), 1)

    def test_sparkline(self):
        rows = I.sparkline([0.7, 0.5, 0.3, 0.1], width=24, height=5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(0 < len(r) <= 24 for r in rows))  # se recortan los espacios finales
        self.assertTrue(any("▂" in r for r in rows))

    def test_sparkline_vacia(self):
        self.assertTrue(I.sparkline([]))


class TestCLI(unittest.TestCase):
    def test_main_exporta_y_no_revienta(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "m.json")
            code = I.main(["--data", "blobs", "-H", "4", "--epochs", "12",
                           "--quiet", "--no-map", "--export", out])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                d = json.load(fh)
            self.assertEqual(d["format"], "ias-mlp/1")
            self.assertEqual(len(d["probes"]), 35)
            self.assertTrue(all(0.0 <= p[2] <= 1.0 for p in d["probes"]))

    def test_check_pasa(self):
        self.assertTrue(I.gradient_check())


if __name__ == "__main__":
    unittest.main(verbosity=2)
