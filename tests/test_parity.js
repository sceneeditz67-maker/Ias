/* ==========================================================================
   test_parity.js — el motor de Python y el de JS tienen que dar lo mismo
   --------------------------------------------------------------------------
   Entrena en Python (`python3 python/ias_net.py --export …`), carga esos
   pesos en la red del navegador y comprueba que:
     · las predicciones coinciden en 35 puntos de control
     · el JSON viaja con los datos y la red importada los clasifica bien
     · un export→import propio del JS conserva el modelo bit a bit

   node tests/test_parity.js
   ========================================================================== */

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { MLP, makeDataset } = require(path.join(__dirname, "..", "assets", "core.js"));

const ROOT = path.join(__dirname, "..");
let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log(`  \u001b[32m✔\u001b[0m ${name}${extra ? "  " + extra : ""}`); }
  else { fail++; console.log(`  \u001b[31m✗\u001b[0m ${name}${extra ? "  " + extra : ""}`); }
};

const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "ias-parity-")), "modelo.json");

let python = null;
for (const exe of ["python3", "python"]) {
  try { execFileSync(exe, ["-c", "pass"], { stdio: "ignore" }); python = exe; break; } catch { /* sigue intentando */ }
}
if (!python) {
  console.log("\n\u001b[33mpython no está disponible: se omite el test de paridad\u001b[0m\n");
  process.exit(0);
}

console.log("\n\u001b[36mentrenando en Python y exportando pesos…\u001b[0m");
let exported = null;
try {
  exported = execFileSync(python, [
    path.join("python", "ias_net.py"),
    "--data", "xor", "-H", "6,4", "--epochs", "260", "--lr", "0.04",
    "--seed", "7", "--quiet", "--no-map", "--export", tmp,
  ], { cwd: ROOT, encoding: "utf8" });
} catch (e) {
  console.log(`  \u001b[31m✗\u001b[0m falló la exportación de Python: ${e.message}\n${e.stdout || ""}`);
  process.exit(1);
}
ok("python/ias_net.py exporta sin errores", fs.existsSync(tmp), exported.trim().split("\n").pop().trim());

const d = JSON.parse(fs.readFileSync(tmp, "utf8"));
ok("el JSON declara el formato ias-mlp/1", d.format === "ias-mlp/1", d.format);
ok("la arquitectura viaja intacta", d.sizes.join(",") === "2,6,4,1", d.sizes.join(" → "));

const net = MLP.fromDict(d, 1234);
ok("el navegador construye esa misma red", net.nParams === d.layers.reduce((s, l) => s + l.w.length + l.b.length, 0),
  `${net.nParams} parámetros`);

let worst = 0;
let n = 0;
for (const [x, y, pPy] of d.probes) {
  const pJs = net.forward(x, y);
  worst = Math.max(worst, Math.abs(pJs - pPy));
  n++;
}
ok(`mismas predicciones en los ${n} puntos de control`, worst < 2e-6,
  `|Δp| máx = ${worst.toExponential(2)} (los pesos se redondean a 8 decimales)`);

{
  const X = d.dataset.X, y = d.dataset.y;
  let good = 0;
  for (let i = 0; i < X.length; i++) if ((net.forward(X[i][0], X[i][1]) >= 0.5) === (y[i] >= 0.5)) good++;
  const acc = good / X.length;
  ok("la red importada clasifica los datos originales", acc > 0.9,
    `${(acc * 100).toFixed(1)} % sobre ${X.length} puntos (` + `${d.dataset.name})`);
}

/* ---------------- el camino inverso: JS exporta, Python importa ---------------- */

console.log("\n\u001b[36mJS → Python (mismo JSON, otra dirección)\u001b[0m");
const jsNet = new MLP([2, 7, 1], { act: "tanh", seed: 21 });
const pts = makeDataset("circle", 180, 0.05, 3, 0);
const order = pts.map((_, i) => i);
for (let e = 0; e < 300; e++) jsNet.trainEpoch(pts, order, 32, Infinity, () => 0);
const jsAcc = jsNet.score(pts, false)[0];
const outFile = path.join(path.dirname(tmp), "desde-js.json");
fs.writeFileSync(outFile, JSON.stringify(jsNet.toDict()));
ok("el navegador exporta un modelo entrenado", jsAcc > 0.95 && fs.existsSync(outFile),
  `acc ${(jsAcc * 100).toFixed(1)} % · ${jsNet.nParams} params`);

const py = `
import json, sys
sys.path.insert(0, ${JSON.stringify(path.join(ROOT, "python"))})
from ias_net import MLP
d = json.load(open(${JSON.stringify(outFile)}))
net = MLP.from_dict(d)
probes = d.get("probes") or []
print(json.dumps({"sizes": net.sizes, "n": net.n_params}))
`;
let pyOut = "";
try {
  pyOut = execFileSync(python, ["-c", py], { encoding: "utf8" }).trim();
} catch (e) {
  pyOut = "ERROR: " + (e.stderr || e.message);
}
ok("python/ias_net.py lee el JSON generado por el navegador", /"sizes": \[2, 7, 1\]/.test(pyOut),
  pyOut.replace(/\s+/g, " ").slice(0, 80));

fs.rmSync(path.dirname(tmp), { recursive: true, force: true });
console.log(`\n${fail === 0 ? "\u001b[32m" : "\u001b[31m"}${pass} pasan, ${fail} fallan\u001b[0m\n`);
process.exit(fail === 0 ? 0 : 1);
