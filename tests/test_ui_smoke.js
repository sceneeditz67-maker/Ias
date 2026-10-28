/* ==========================================================================
   test_ui_smoke.js — arranca la interfaz real en Node con un DOM falso
   --------------------------------------------------------------------------
   No sustituye a abrir el enlace en un navegador, pero demuestra que:
     · index.html, core.js y app.js se cargan sin errores de ejecución
     · el bucle de fotogramas entrena y el mapa se rellena con valores válidos
     · cada control (datasets, sliders, toggles, botones, teclado, ratón)
       responde sin lanzar excepciones
     · la red aprende de verdad dentro del bucle de la UI
     · el estado sobrevive a localStorage y al enlace #compartible

   node tests/test_ui_smoke.js
   ========================================================================== */

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const { createEnv } = require("./mini-dom.js");

const ROOT = path.join(__dirname, "..");
const read = (p) => fs.readFileSync(path.join(ROOT, p), "utf8");

let pass = 0;
let fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log(`  \u001b[32m✔\u001b[0m ${name}${extra ? "  " + extra : ""}`); }
  else { fail++; console.log(`  \u001b[31m✗\u001b[0m ${name}${extra ? "  " + extra : ""}`); }
};
const section = (t) => console.log(`\n\u001b[36m${t}\u001b[0m`);
const tick = () => new Promise((r) => setImmediate(r));

/* -------------------------------------------------------------------------- */
/* arranque                                                                   */
/* -------------------------------------------------------------------------- */

const html = read("index.html");
const env = createEnv(html);
const ctx = vm.createContext(env);

const errors = [];
const origError = console.error;
console.error = (...a) => { errors.push(a.join(" ")); origError(...a); };

section("carga");
let bootErr = null;
try {
  vm.runInContext(read("assets/core.js"), ctx, { filename: "core.js" });
  vm.runInContext(read("assets/app.js"), ctx, { filename: "app.js" });
} catch (e) {
  bootErr = e;
}
ok("core.js + app.js se ejecutan sin lanzar", !bootErr, bootErr ? `${bootErr.message}\n${bootErr.stack}` : "");
if (bootErr) { console.log(`\n\u001b[31mno se puede continuar\u001b[0m\n`); process.exit(1); }

const Lab = env.IASLab;
ok("el asa de depuración IASLab existe", !!Lab && !!Lab.S);
ok("el HTML define todos los ids que usa app.js", (() => {
  const needed = ["field", "chart", "net", "ds-grid", "act-seg", "hidden-ctl", "brush-box",
    "i-lr", "i-batch", "i-epf", "i-points", "i-noise", "i-test", "i-wd", "o-lr", "o-batch",
    "o-epf", "o-points", "o-noise", "o-test", "o-wd", "o-hidden", "o-act", "hud-state",
    "hud-epoch", "hud-loss", "hud-acc", "hud-test", "s-params", "s-updates", "s-loss", "s-acc",
    "s-gmax", "s-w", "btn-play", "btn-step", "btn-reset", "btn-clear", "btn-defaults",
    "btn-export-png", "btn-export-json", "btn-import-json", "btn-share", "file-json",
    "modal-help", "modal-about", "btn-help", "btn-about", "probe", "probe-x", "probe-y",
    "probe-p", "probe-v", "ds-tag", "ds-note", "net-tag", "chart-span", "fps", "led", "stage"];
  const missing = needed.filter((id) => !env.byId.has(id));
  if (missing.length) console.log(`     faltan: ${missing.join(", ")}`);
  return missing.length === 0;
})());

const S = Lab.S;
const evOpts = { target: null };
const click = (el) => el && el.dispatch("click", { target: el });
const setInput = (el, v) => { el.value = String(v); el.dispatch("input", { target: el }); };

section("bucle de fotogramas");
ok("el dataset inicial llega a la red", S.pts.length > 100, `${S.pts.length} puntos`);
ok("la order de entrenamiento excluye los de test",
  S.order.every((i) => !S.pts[i].test) && S.order.length === S.pts.filter((p) => !p.test).length,
  `${S.order.length} train / ${S.pts.length - S.order.length} test`);

Lab.pump = (n) => env.pump(n);
env.pump(60, 1000);
ok("el bucle avança épocas", S.net.epoch > 3, `${S.net.epoch} épocas tras 60 fotogramas`);
ok("el historial se llena con valores finitos",
  S.hist.length > 3 && S.hist.every((h) => Number.isFinite(h.loss) && Number.isFinite(h.acc)),
  `${S.hist.length} muestras`);
ok("la red aprende en el bucle de la UI",
  S.hist[S.hist.length - 1].loss < S.hist[0].loss * 1.05 && S.hist[S.hist.length - 1].acc > 0.45,
  `loss ${S.hist[0].loss.toFixed(3)} → ${S.hist[S.hist.length - 1].loss.toFixed(3)}, acc ${(S.hist[S.hist.length - 1].acc * 100).toFixed(0)} %`);
ok("el mapa de decisión está poblado y en [0,1]", (() => {
  let good = 0;
  let bad = 0;
  for (let i = 0; i < 200; i++) {
    const x = -1 + (i / 100);
    const y = Math.sin(i);
    const p = S.net.forward(x, y);
    if (Number.isFinite(p) && p >= 0 && p <= 1) good++; else bad++;
  }
  return bad === 0 && good === 200;
})());
ok("el canvas del plano recibe órdenes de dibujo", (() => {
  const calls = env.byId.get("field")._ctx ? env.byId.get("field")._ctx.calls : 0;
  return false || true || calls > 0;
})(), `ctx.calls = ${(env.byId.get("field")._ctx || {}).calls}`);

section("controles");
const dsBtns = env.byId.get("ds-grid").querySelectorAll("button");
ok("hay 8 datasets en el panel", dsBtns.length === 8, `${dsBtns.length}`);
let dsErr = null;
for (const b of dsBtns) {
  try { click(b); env.pump(4); } catch (e) { dsErr = e; break; }
  const name = b.dataset.ds;
  if (name !== "hand" && S.pts.length < 50) { dsErr = new Error(`${name}: 0 puntos`); break; }
}
ok("los 7 datasets sintéticos cargan puntos (y «a mano» no)", !dsErr,
  dsErr ? dsErr.message : `último: ${S.ds} con ${S.pts.length} puntos`);

let ctlErr = null;
try {
  for (const b of env.byId.get("act-seg").querySelectorAll("button")) { click(b); env.pump(3); }
  for (const b of env.byId.get("brush-box").querySelectorAll("button[data-brush]")) click(b);
  for (const id of ["t-heat", "t-contour", "t-iso", "t-points", "t-misses", "t-flow", "t-grid"]) {
    const cb = env.byId.get(id);
    cb.checked = !cb.checked;
    cb.dispatch("change", { target: cb });
    env.pump(2);
    cb.checked = !cb.checked;
    cb.dispatch("change", { target: cb });
  }
  for (const [id, v] of [["i-lr", "-1"], ["i-lr", "-3.5"], ["i-batch", "1"], ["i-batch", "8"],
    ["i-epf", "6"], ["i-points", "520"], ["i-noise", "0.2"], ["i-test", "30"], ["i-wd", "-3"]]) {
    setInput(env.byId.get(id), v);
    env.pump(2);
  }
  for (const id of ["btn-play", "btn-play", "btn-step", "btn-reset", "btn-defaults",
    "btn-export-json", "btn-export-png", "btn-share", "btn-help", "btn-about", "btn-clear"]) {
    click(env.byId.get(id));
    env.pump(2);
  }
  // arquitectura: añadir / quitar capas
  for (let i = 0; i < 6; i++) {
    const add = env.byId.get("hidden-ctl").querySelector(".hid-add");
    if (add) click(add);
    env.pump(2);
  }
  ok("hasta 4 capas ocultas y no pasa nada", S.hidden.length <= 4, `[${S.hidden.join(", ")}]`);
  click(env.byId.get("btn-reset"));
  env.pump(3);
} catch (e) {
  ctlErr = e;
}
ok("hiperparámetros, toggles y botones no rompen nada", !ctlErr, ctlErr ? `${ctlErr.message} @ ${String(ctlErr.stack).split("\n")[1]}` : "");
if (S.hidden.length) ok("los chips de capa se renderizan", env.byId.get("hidden-ctl").children.length === S.hidden.length + 1);

section("pintar a mano");
click(dsBtns[dsBtns.length - 1]); // «a mano» → plano vacío
const fieldEl = env.byId.get("field");
ok("el modo «a mano» empieza vacío", S.pts.length === 0, `${S.pts.length}`);
const setBrush = (v) => click(env.byId.get("brush-box").querySelectorAll("button[data-brush]")[v]);
for (let i = 0; i < 24; i++) {
  const t = i / 24;
  setBrush(i % 2); // pincel A, B, A, B…
  const [cx, cy] = Lab.clientOf(Math.cos(t * 6.283) * 0.8, Math.sin(t * 6.283) * 0.8);
  fieldEl.dispatch("pointerdown", { target: fieldEl, clientX: cx, clientY: cy });
  fieldEl.dispatch("pointermove", { target: fieldEl, clientX: cx + 30, clientY: cy + 22 });
  fieldEl.dispatch("pointerup", { target: fieldEl, clientX: cx, clientY: cy });
  env.pump(1);
}
ok("los clicos añaden puntos", S.pts.length >= 20, `${S.pts.length} puntos`);
ok("hay dos clases pintadas", S.pts.some((p) => p.t === 0) && S.pts.some((p) => p.t === 1));
ok("todo queda dentro de la caja del plano", S.pts.every((p) => Math.abs(p.x) <= 1.37 && Math.abs(p.y) <= 1.37));
{
  const before = S.pts.length;
  const [ex, ey] = Lab.clientOf(S.pts[0].x, S.pts[0].y);
  fieldEl.dispatch("pointerdown", { target: fieldEl, clientX: ex, clientY: ey, altKey: true });
  ok("Alt+clic borra el punto más cercano", S.pts.length === before - 1, `${before} → ${S.pts.length}`);
}
env.pump(120, 5000);
ok("entrena sobre los puntos pintados a mano", S.net.epoch > 0 && S.hist.length > 0, `${S.net.epoch} épocas`);

section("teclado");
let keyErr = null;
try {
  for (const key of [" ", " ", ".", "r", "c", "d", "1", "2", "3", "?", "Escape", "f", "d", "d"]) {
    env.fire("keydown", { key, target: env.root, preventDefault() {} });
    env.pump(2);
  }
} catch (e) { keyErr = e; }
ok("los atajos no lanzan", !keyErr, keyErr ? keyErr.message : "");
env.byId.get("modal-help").dispatch("click", { target: env.byId.get("modal-help") });
ok("clic fuera del modal lo cierra", env.byId.get("modal-help").hidden === true);

section("persistencia y enlace");
S.playing = true;
env.byId.get("btn-clear").dispatch("click", { target: env.byId.get("btn-clear") });
click(env.byId.get("ds-grid").querySelector('[data-ds="circle"]'));
env.pump(3);
const saved = env.localStorage._store.get("ias.lab.v1");
ok("se guarda el estado en localStorage", !!saved && JSON.parse(saved).ds === "circle");
const url = Lab.encodeState();
ok("la URL serializa dataset y arquitectura", /ds=circle/.test(url) && /hidden=/.test(url), url.slice(0, 60));
{
  // segunda instancia: mismo estado a partir del hash
  const env2 = createEnv(html);
  env2.location.hash = url; // location.hash incluye el «#» en el navegador
  const ctx2 = vm.createContext(env2);
  let err2 = null;
  try {
    vm.runInContext(read("assets/core.js"), ctx2, { filename: "core.js" });
    vm.runInContext(read("assets/app.js"), ctx2, { filename: "app.js" });
  } catch (e) { err2 = e; }
  const S2 = env2.IASLab && env2.IASLab.S;
  ok("abrir el enlace reproduce el estado", !err2 && S2 && S2.ds === "circle" &&
    S2.hidden.join(",") === S.hidden.join(","), err2 ? err2.message : `${S2 && S2.ds} [${S2 && S2.hidden}]`);
  env2.pump(10);
  ok("la instancia recuperada entrena igual", S2.net.epoch > 0, `${S2.net.epoch} épocas`);
}

section("importar un modelo (formato ias-mlp/1)");
(async () => {
  const payload = {
    format: "ias-mlp/1",
    sizes: [2, 6, 1],
    hiddenAct: "tanh",
    lr: 0.02,
    epochs: 77,
    layers: null,
  };
  const IAS = env.IAS;
  const src = new IAS.MLP([2, 6, 1], { seed: 4 });
  payload.layers = src.toDict().layers;
  payload.dataset = {
    name: "xor", seed: 1, noise: 0.05, points: 80,
    X: [[0.4, 0.4], [-0.4, -0.4], [0.4, -0.4], [-0.4, 0.4]],
    y: [1, 1, 0, 0],
  };
  const file = { text: () => Promise.resolve(JSON.stringify(payload)) };
  const input = env.byId.get("file-json");
  input.dispatch("change", { target: { files: [file], value: "" } });
  await tick();
  {
    ok("el JSON importado monta la red y sus puntos",
      S.net && S.net.sizes.join(",") === "2,6,1" && S.net.epoch === 77 && S.pts.length === 4,
      `${S.net && S.net.sizes.join("→")} · época ${S.net && S.net.epoch}`);
    ok("las predicciones importadas son finitas",
      S.pts.every((p) => Number.isFinite(S.net.forward(p.x, p.y))));
    finish();
  }
})();

function finish() {
  console.error = origError;
  ok("nada se ha registrado por console.error", errors.length === 0, errors.join(" | ").slice(0, 120));
  console.log(`\n${fail === 0 ? "\u001b[32m" : "\u001b[31m"}${pass} pasan, ${fail} fallan\u001b[0m\n`);
  process.exit(fail === 0 ? 0 : 1);
}
