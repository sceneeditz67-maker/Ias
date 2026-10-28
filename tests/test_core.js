/* ==========================================================================
   Tests del motor (Node, sin dependencias):  node tests/test_core.js
   ========================================================================== */

"use strict";

const path = require("path");
const { MLP, makeDataset, isoSegments, bce, mulberry32 } = require(path.join(__dirname, "..", "assets", "core.js"));

let pass = 0;
let fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log(`  ✔ ${name}${extra ? "  " + extra : ""}`); }
  else { fail++; console.log(`  ✗ ${name}${extra ? "  " + extra : ""}`); }
};

const section = (t) => console.log(`\n\u001b[36m${t}\u001b[0m`);

/* -------------------------------------------------------------------------- */
section("gradientes: analítico vs numérico (diferencias centrales)");
/* -------------------------------------------------------------------------- */

for (const act of ["tanh", "relu", "leaky", "elu"]) {
  const net = new MLP([2, 5, 4, 1], { act, seed: 3 });
  const pts = makeDataset("xor", 8, 0, 1);
  let worst = 0;
  for (let k = 0; k < 4; k++) {
    const p = pts[k];
    for (const l of net.layers) l.resetGrads();
    net.accumulate(p.x, p.y, p.t);
    for (let li = 0; li < net.layers.length; li++) {
      const L = net.layers[li];
      const num = net.numericGradWeight(p.x, p.y, p.t, li);
      for (let i = 0; i < num.length; i++) {
        const den = Math.max(1e-8, Math.abs(num[i]) + Math.abs(L.gw[i]));
        worst = Math.max(worst, Math.abs(num[i] - L.gw[i]) / den);
      }
    }
  }
  ok(`act=${act}`, worst < 5e-3, `error relativo máx = ${worst.toExponential(2)}`);
}

/* -------------------------------------------------------------------------- */
section("aprendizaje: la red tiene que resolver lo que puede y fallar en lo que no");
/* -------------------------------------------------------------------------- */

function train(sizes, ds, opts = {}) {
  const net = new MLP(sizes, { act: "tanh", seed: 7, lr: opts.lr ?? 0.03, ...opts });
  const pts = makeDataset(ds, 240, 0.06, 7, 0);
  const order = pts.map((_, i) => i);
  const rnd = mulberry32(11);
  for (let e = 0; e < (opts.epochs ?? 500); e++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    net.trainEpoch(pts, order, opts.batch ?? 32, Infinity, () => 0);
  }
  const [acc, loss] = net.score(pts, false);
  return { net, acc, loss };
}

let r = train([2, 8, 1], "xor", { epochs: 700 });
ok("XOR con capa oculta [8] → >95 %", r.acc > 0.95, `acc ${(r.acc * 100).toFixed(1)} %, loss ${r.loss.toFixed(4)}`);

r = train([2, 1], "xor", { epochs: 400 }); // sin capa oculta: imposible
ok("XOR con red lineal [1] → ~50 % (debe fracasar)", r.acc < 0.75, `acc ${(r.acc * 100).toFixed(1)} %`);

r = train([2, 10, 8, 1], "circle", { epochs: 500 });
ok("dos anillos concéntricos con [10,8] → >97 %", r.acc > 0.97, `acc ${(r.acc * 100).toFixed(1)} %`);

r = train([2, 1], "blobs", { epochs: 150 });
ok("manchas lineales con red vacía → >95 %", r.acc > 0.95, `acc ${(r.acc * 100).toFixed(1)} %`);

r = train([2, 10, 8, 1], "spiral", { epochs: 900 });
ok("espiral con [10,8] aprende (>80 %)", r.acc > 0.8, `acc ${(r.acc * 100).toFixed(1)} %`);

{
  const net = new MLP([2, 8, 1], { seed: 5, lr: 0.2 });
  const pts = makeDataset("checker", 200, 0.05, 3, 0);
  const order = pts.map((_, i) => i);
  const loss0 = net.score(pts, false)[1];
  for (let e = 0; e < 60; e++) net.trainEpoch(pts, order, 32, Infinity, () => 0);
  const loss1 = net.score(pts, false)[1];
  ok("la pérdida baja de forma monótona global", loss1 < loss0, `${loss0.toFixed(3)} → ${loss1.toFixed(3)}`);
}

/* -------------------------------------------------------------------------- */
section("datasets y utilidades");
/* -------------------------------------------------------------------------- */

for (const ds of ["blobs", "xor", "circle", "moons", "spiral", "checker", "sine"]) {
  const a = makeDataset(ds, 200, 0.07, 42, 15);
  const b = makeDataset(ds, 200, 0.07, 42, 15);
  const same = a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y && p.t === b[i].t);
  const inBox = a.every((p) => Math.abs(p.x) <= 1.37 && Math.abs(p.y) <= 1.37);
  const both = a.some((p) => p.t === 0) && a.some((p) => p.t === 1);
  const nTest = a.filter((p) => p.test).length;
  ok(`${ds}: determinista, en caja, 2 clases, test ≈ 15 %`,
    same && inBox && both && Math.abs(nTest / a.length - 0.15) < 0.03,
    `${a.length} pts, ${nTest} test`);
}

{
  // rejilla con una frontera recta vertical en x=0 → marching squares debe hallarla
  const nx = 21, ny = 11;
  const grid = new Float32Array(nx * ny);
  for (let gy = 0; gy < ny; gy++)
    for (let gx = 0; gx < nx; gx++) grid[gy * nx + gx] = gx / (nx - 1);
  const segs = isoSegments(grid, nx, ny, 10, 10, 0.5);
  const xs = [];
  for (let i = 0; i < segs.length; i += 4) xs.push(segs[i], segs[i + 2]);
  // el nivel 0.5 cae en la columna central: gx = 10 → x = 10 * sx
  const mid = ((nx - 1) / 2) * 10;
  const vertical = xs.every((v) => Math.abs(v - mid) < 1e-6);
  ok("isoSegments: frontera vertical en el centro", segs.length > 0 && vertical,
    `${segs.length / 4} segmentos en x = ${Math.min(...xs)}`);

  const flat = isoSegments(new Float32Array(nx * ny).fill(0.2), nx, ny, 10, 10, 0.5);
  ok("isoSegments: campo sin cruce → 0 segmentos", flat.length === 0);
}

{
  const m = new MLP([2, 6, 3, 1], { seed: 8 });
  const d = m.toDict(6);
  const s2 = MLP.fromDict(d, 8);
  const x = 0.31, y = -0.77;
  const p1 = m.forward(x, y);
  const p2 = s2.forward(x, y);
  ok("ias-mlp/1: exportar → importar da las mismas predicciones", Math.abs(p1 - p2) < 5e-7,
    `p=${p1.toFixed(6)} vs ${p2.toFixed(6)} (Δ${Math.abs(p1 - p2).toExponential(1)})`);
  ok("ias-mlp/1: conteo de parámetros conservado", m.nParams === s2.nParams, `${m.nParams} params`);
  let threw = false;
  try { MLP.fromDict({ format: "otro/9", sizes: [2, 1] }); } catch { threw = true; }
  ok("ias-mlp/1: rechaza formatos desconocidos", threw);
}

{
  const net = new MLP([2, 4, 1], { seed: 1 });
  ok("bce: p=1,t=1 es ~0", bce(0.9999999, 1) < 1e-6);
  ok("forward dentro de (0,1)", [...Array(20)].every((_, i) => {
    const p = net.forward(Math.sin(i), Math.cos(i));
    return p > 0 && p < 1;
  }));
}

console.log(`\n${fail === 0 ? "\u001b[32m" : "\u001b[31m"}${pass} pasan, ${fail} fallan\u001b[0m\n`);
process.exit(fail === 0 ? 0 : 1);
