/* ==========================================================================
   IAS · app.js — capa de interfaz (DOM)
   --------------------------------------------------------------------------
   Todo el álgebra vive en assets/core.js (globalThis.IAS). Aquí solo hay:
   bucle de fotogramas, render del mapa de decisión, diagrama de la red,
   curvas, controles, pintura a mano, persistencia e import/export.
   ========================================================================== */

"use strict";

const { MLP, makeDataset, clamp, isoSegments, buildLUT, LIMIT } = globalThis.IAS;

const $ = (id) => document.getElementById(id);
const nowMs = () => performance.now();
const lerp = (a, b, t) => a + (b - a) * t;

/* -------------------------------------------------------------------------- */
/* Estado                                                                     */
/* -------------------------------------------------------------------------- */

const DEFAULTS = {
  ds: "spiral", points: 240, noise: 0.07, seed: 7,
  hidden: [10, 8], act: "tanh", lr: 0.02, batch: 32, epf: 2, testPct: 15, wd: 0.0001,
};

const SHOW_DEFAULTS = {
  heat: true, contour: true, iso: false, points: true, misses: true, flow: true, grid: true,
};

const S = {
  ...DEFAULTS,
  hidden: [...DEFAULTS.hidden],
  show: { ...SHOW_DEFAULTS },
  pts: [],
  order: [],
  net: null,
  hist: [],
  playing: true,
  brush: 0,
  hover: null,
  frame: 0,
  fps: 0,
};

const DS_NOTES = {
  blobs: "Linealmente separable: una red sin capas ocultas ya la clava. Úsala de control.",
  xor: "El clásico imposible para una capa lineal. Pide a gritos una capa oculta.",
  circle: "Dos anillos concéntricos: la frontera es una circunferencia, no una recta.",
  moons: "Dos medias lunas entrelazadas. Con pocas neuronas se queda a medias.",
  spiral: "Dos brazos de espiral: el más difícil de los suaves. Paciencia y Adam.",
  checker: "Tablero 4×4: la frontera es un rompecabezas. Perfecto para ver sobrentrenamiento.",
  sine: "Dos bandas sinusoidales. Ideal para comparar tanh (suave) con relu (a trozos).",
  hand: "Pinta tú los datos: clic = pincel activo. Cambia con 1 / 2, borra con 3 o Alt+clic.",
};

/* -------------------------------------------------------------------------- */
/* Lienzo del plano                                                           */
/* -------------------------------------------------------------------------- */

const field = $("field");
const fctx = field.getContext("2d", { alpha: false });
const heatCv = document.createElement("canvas");
const hctx = heatCv.getContext("2d");
const LUT = buildLUT();

const RANGE = 1.4;   // semi-lado del cuadrado de mundo siempre visible
const GX = 104;      // resolución del mapa de probabilidad
let W = 0, H = 0, DPR = 1, SCALE = 1, OX = 0, OY = 0, GY = 64;
let grid = new Float32Array(GX * GY);
let heatImg = null;

const toWorld = (px, py) => [(px - OX) / SCALE, (OY - py) / SCALE];
const toScreen = (x, y) => [OX + x * SCALE, OY - y * SCALE];

function resize() {
  const rect = field.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = Math.max(300, Math.round(rect.width * DPR));
  H = Math.max(200, Math.round(rect.height * DPR));
  if (field.width !== W) field.width = W;
  if (field.height !== H) field.height = H;
  OX = W / 2;
  OY = H / 2;
  SCALE = Math.min(W, H) / (2 * RANGE);
  GY = Math.max(20, Math.round((GX * H) / W));
  grid = new Float32Array(GX * GY);
  heatCv.width = GX;
  heatCv.height = GY;
  heatImg = hctx.createImageData(GX, GY);
}

/** Rellena `grid` y la LUT de color. Se llama cada pocos fotogramas. */
function computeGrid() {
  const net = S.net;
  if (!net || !heatImg) return;
  const wx = OX / SCALE;
  const wy = OY / SCALE;
  const d = heatImg.data;
  for (let gy = 0; gy < GY; gy++) {
    const y = wy - (gy / (GY - 1)) * 2 * wy;
    for (let gx = 0; gx < GX; gx++) {
      const x = -wx + (gx / (GX - 1)) * 2 * wx;
      const p = net.forward(x, y);
      const idx = gy * GX + gx;
      grid[idx] = p;
      const c = (clamp(p, 0, 1) * 63) | 0;
      const o = idx * 4;
      d[o] = LUT[c * 3];
      d[o + 1] = LUT[c * 3 + 1];
      d[o + 2] = LUT[c * 3 + 2];
      d[o + 3] = 255;
    }
  }
  hctx.putImageData(heatImg, 0, 0);
}

const needsGrid = () => S.net && (S.show.heat || S.show.contour || S.show.iso);

function drawIsolines(level, style, width, glow) {
  const segs = isoSegments(grid, GX, GY, W / (GX - 1), H / (GY - 1), level);
  if (!segs.length) return;
  fctx.save();
  fctx.strokeStyle = style;
  fctx.lineWidth = width;
  fctx.lineCap = "round";
  if (glow) {
    fctx.shadowColor = style;
    fctx.shadowBlur = 9 * DPR;
  }
  fctx.beginPath();
  for (let i = 0; i < segs.length; i += 4) {
    fctx.moveTo(segs[i], segs[i + 1]);
    fctx.lineTo(segs[i + 2], segs[i + 3]);
  }
  fctx.stroke();
  fctx.restore();
}

function drawGrid() {
  fctx.save();
  fctx.lineWidth = 1;
  fctx.strokeStyle = "rgba(150,185,235,.075)";
  fctx.beginPath();
  for (let v = -1.25; v <= 1.251; v += 0.25) {
    const [gx] = toScreen(v, 0);
    const [, gy] = toScreen(0, v);
    if (gx >= 0 && gx <= W) { fctx.moveTo(gx, 0); fctx.lineTo(gx, H); }
    if (gy >= 0 && gy <= H) { fctx.moveTo(0, gy); fctx.lineTo(W, gy); }
  }
  fctx.stroke();
  fctx.strokeStyle = "rgba(150,185,235,.22)";
  fctx.lineWidth = 1.2;
  fctx.beginPath();
  fctx.moveTo(OX, 0); fctx.lineTo(OX, H);
  fctx.moveTo(0, OY); fctx.lineTo(W, OY);
  fctx.stroke();
  fctx.restore();
}

function drawPoints() {
  const net = S.net;
  const r = 2.9 * DPR;
  for (const p of S.pts) {
    const [px, py] = toScreen(p.x, p.y);
    if (px < -10 || py < -10 || px > W + 10 || py > H + 10) continue;
    const col = p.t >= 0.5 ? "#ff9052" : "#58a6ff";
    const pred = net ? net.forward(p.x, p.y) : 0.5;
    const wrong = net ? Math.abs(pred - p.t) > 0.5 : false;

    if (p.test) {
      fctx.strokeStyle = col;
      fctx.globalAlpha = 0.9;
      fctx.lineWidth = 1.3 * DPR;
      fctx.beginPath();
      fctx.arc(px, py, r * 1.75, 0, Math.PI * 2);
      fctx.stroke();
      fctx.globalAlpha = 1;
      continue;
    }
    if (S.show.misses && wrong) {
      fctx.strokeStyle = "rgba(255,45,85,.95)";
      fctx.lineWidth = 1.4 * DPR;
      fctx.beginPath();
      fctx.arc(px, py, r * 2.35, 0, Math.PI * 2);
      fctx.stroke();
    }
    fctx.fillStyle = col;
    fctx.beginPath();
    fctx.arc(px, py, r, 0, Math.PI * 2);
    fctx.fill();
  }
}

/** Resumen del estado actual de los datos: alimenta el pie del plano. */
function fieldSummary() {
  const net = S.net;
  if (!net || !S.pts.length) return null;
  let misses = 0, conf = 0, testMisses = 0, n = 0;
  for (const p of S.pts) {
    const pred = net.forward(p.x, p.y);
    const wrong = pred >= 0.5 !== (p.t >= 0.5);
    conf += 1 - Math.abs(pred - 0.5) * 2;
    n++;
    if (wrong) { if (p.test) testMisses++; else misses++; }
  }
  return { misses, testMisses, conf: conf / n, n };
}

function renderField() {
  fctx.fillStyle = "#060a12";
  fctx.fillRect(0, 0, W, H);
  if (S.show.heat && S.net) {
    fctx.imageSmoothingEnabled = true;
    fctx.globalAlpha = 0.94;
    fctx.drawImage(heatCv, 0, 0, W, H);
    fctx.globalAlpha = 1;
  }
  if (S.show.grid) drawGrid();
  if (S.net && needsGrid()) {
    if (S.show.iso) {
      drawIsolines(0.1, "rgba(88,166,255,.45)", 1 * DPR, false);
      drawIsolines(0.9, "rgba(255,144,82,.45)", 1 * DPR, false);
    }
    if (S.show.contour) drawIsolines(0.5, "rgba(255,255,255,.94)", 1.7 * DPR, true);
  }
  if (S.show.points) drawPoints();
  if (S.hover) {
    const [hx, hy] = toScreen(S.hover.x, S.hover.y);
    fctx.save();
    fctx.strokeStyle = "rgba(57,224,200,.4)";
    fctx.setLineDash([4 * DPR, 5 * DPR]);
    fctx.lineWidth = 1;
    fctx.beginPath();
    fctx.moveTo(hx, 0); fctx.lineTo(hx, H);
    fctx.moveTo(0, hy); fctx.lineTo(W, hy);
    fctx.stroke();
    fctx.restore();
    const col = S.brush === "erase" ? "#ff2d55" : S.brush ? "#ff9052" : "#58a6ff";
    fctx.strokeStyle = col;
    fctx.lineWidth = 1.6 * DPR;
    fctx.beginPath();
    fctx.arc(hx, hy, 5.5 * DPR, 0, Math.PI * 2);
    fctx.stroke();
  }
}

/* -------------------------------------------------------------------------- */
/* Diagrama de la red                                                         */
/* -------------------------------------------------------------------------- */

const netCv = $("net");
const nctx = netCv.getContext("2d");
let NW = 0, NH = 0;
let flowPhase = 0;

function drawNetDiagram() {
  const net = S.net;
  if (!net) return;
  const rect = netCv.getBoundingClientRect();
  NW = Math.max(200, Math.round(rect.width * DPR));
  NH = Math.round(128 * DPR);
  if (netCv.width !== NW) netCv.width = NW;
  if (netCv.height !== NH) netCv.height = NH;

  const L = net.layers;
  const sizes = [2, ...L.slice(0, -1).map((l) => l.nOut), 1];
  const cols = sizes.map((n, li) => {
    const t = li / Math.max(sizes.length - 1, 1);
    return { n, x: 24 * DPR + t * (NW - 48 * DPR), y: [], r: 0 };
  });
  const maxN = Math.max(...sizes);
  const gap = (NH - 30 * DPR) / maxN;
  const rad = clamp(gap * 0.3, 2.4 * DPR, 8.5 * DPR);
  for (const c of cols) {
    const total = (c.n - 1) * gap;
    for (let i = 0; i < c.n; i++) c.y.push((NH - 12 * DPR - total) / 2 + i * gap);
    c.r = rad;
  }

  nctx.clearRect(0, 0, NW, NH);
  nctx.fillStyle = "rgba(6,10,18,.5)";
  nctx.fillRect(0, 0, NW, NH);

  const acts = S.hover ? net.activations(S.hover.x, S.hover.y) : null;
  for (let li = 0; li < L.length; li++) {
    const layer = L[li];
    const mx = layer.maxWeight || 1;
    const from = cols[li];
    const to = cols[li + 1];
    const flow = S.show.flow;
    if (flow) {
      nctx.setLineDash([6 * DPR, 10 * DPR]);
      nctx.lineDashOffset = -flowPhase * DPR;
    }
    for (let o = 0; o < layer.nOut; o++) {
      for (let i = 0; i < layer.nIn; i++) {
        const wv = layer.w[o * layer.nIn + i];
        const rel = Math.abs(wv) / mx;
        if (rel < 0.07) continue;
        nctx.strokeStyle = wv >= 0
          ? `rgba(86,227,159,${(0.08 + rel * 0.6).toFixed(3)})`
          : `rgba(255,107,138,${(0.08 + rel * 0.6).toFixed(3)})`;
        nctx.lineWidth = (0.3 + rel * 2.1) * DPR;
        nctx.beginPath();
        nctx.moveTo(from.x + from.r, from.y[i]);
        nctx.lineTo(to.x - to.r, to.y[o]);
        nctx.stroke();
      }
    }
    nctx.setLineDash([]);
  }

  const labels = ["x, y", ...L.slice(0, -1).map(() => "oculta"), "p(B)"];
  cols.forEach((c, li) => {
    const vals = acts ? (li === 0 ? [S.hover.x, S.hover.y] : acts[li - 1]) : null;
    for (let i = 0; i < c.n; i++) {
      let v = 0.35;
      if (vals) {
        const raw = vals[i] ?? 0;
        v = li === 0 ? clamp(Math.abs(raw) / 1.4, 0.05, 1)
          : li === cols.length - 1 ? clamp(raw, 0, 1)
          : clamp(Math.abs(raw), 0, 1);
      }
      nctx.beginPath();
      nctx.arc(c.x, c.y[i], c.r, 0, Math.PI * 2);
      nctx.fillStyle = li === cols.length - 1
        ? `rgba(255,209,102,${(0.22 + v * 0.75).toFixed(3)})`
        : `rgba(57,224,200,${(0.1 + v * 0.8).toFixed(3)})`;
      nctx.shadowColor = li === cols.length - 1 ? "rgba(255,209,102,.85)" : "rgba(57,224,200,.8)";
      nctx.shadowBlur = (0.5 + v * 8) * DPR;
      nctx.fill();
      nctx.shadowBlur = 0;
      nctx.strokeStyle = "rgba(200,230,255,.28)";
      nctx.lineWidth = 1;
      nctx.stroke();
    }
    nctx.fillStyle = "rgba(160,185,215,.62)";
    nctx.font = `${9.5 * DPR}px ui-monospace, monospace`;
    nctx.textAlign = "center";
    nctx.fillText(`${labels[li]} · ${c.n}`, c.x, NH - 3 * DPR);
  });
}

/* -------------------------------------------------------------------------- */
/* Curvas                                                                     */
/* -------------------------------------------------------------------------- */

const chartCv = $("chart");
const cctx = chartCv.getContext("2d");

function drawChart() {
  const rect = chartCv.getBoundingClientRect();
  const w = Math.max(180, Math.round(rect.width * DPR));
  const h = Math.round(128 * DPR);
  if (chartCv.width !== w) chartCv.width = w;
  if (chartCv.height !== h) chartCv.height = h;
  cctx.clearRect(0, 0, w, h);
  cctx.fillStyle = "rgba(6,10,18,.5)";
  cctx.fillRect(0, 0, w, h);
  const pad = 9 * DPR;
  cctx.strokeStyle = "rgba(150,185,235,.1)";
  cctx.lineWidth = 1;
  cctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = pad + ((h - 2 * pad) * i) / 4;
    cctx.moveTo(pad, y);
    cctx.lineTo(w - pad, y);
  }
  cctx.stroke();

  const hist = S.hist.slice(-260);
  if (hist.length < 2) {
    cctx.fillStyle = "rgba(160,185,215,.5)";
    cctx.font = `${11 * DPR}px ui-monospace, monospace`;
    cctx.textAlign = "left";
    cctx.fillText("esperando épocas…", pad + 3, h / 2);
    return;
  }
  const maxLoss = Math.max(0.05, ...hist.map((d) => d.loss));
  const X = (i) => pad + ((w - 2 * pad) * i) / (hist.length - 1);
  const Yl = (v) => pad + (h - 2 * pad) * (1 - clamp(v / maxLoss, 0, 1));
  const Ya = (v) => pad + (h - 2 * pad) * (1 - clamp(v, 0, 1));

  const g = cctx.createLinearGradient(0, pad, 0, h - pad);
  g.addColorStop(0, "rgba(255,107,138,.32)");
  g.addColorStop(1, "rgba(255,107,138,0)");
  cctx.beginPath();
  cctx.moveTo(X(0), h - pad);
  hist.forEach((d, i) => cctx.lineTo(X(i), Yl(d.loss)));
  cctx.lineTo(X(hist.length - 1), h - pad);
  cctx.closePath();
  cctx.fillStyle = g;
  cctx.fill();

  const line = (get, color, width, dash) => {
    cctx.beginPath();
    let started = false;
    hist.forEach((d, i) => {
      const v = get(d);
      if (Number.isNaN(v)) { started = false; return; }
      if (!started) { cctx.moveTo(X(i), v); started = true; }
      else cctx.lineTo(X(i), v);
    });
    cctx.setLineDash(dash || []);
    cctx.strokeStyle = color;
    cctx.lineWidth = width * DPR;
    cctx.stroke();
    cctx.setLineDash([]);
  };
  line((d) => Ya(d.acc), "rgba(86,227,159,.95)", 1.5);
  line((d) => Ya(d.tAcc), "rgba(57,224,200,.8)", 1.3, [4 * DPR, 3 * DPR]);
  line((d) => Yl(d.loss), "rgba(255,107,138,.95)", 1.5);

  cctx.font = `${9.5 * DPR}px ui-monospace, monospace`;
  cctx.fillStyle = "rgba(160,185,215,.6)";
  cctx.textAlign = "left";
  cctx.fillText(`pérdida ${maxLoss.toFixed(2)}`, pad + 2, pad + 8 * DPR);
  $("chart-span").textContent = `${hist.length} épocas · escala ${maxLoss.toFixed(3)} → 0`;
}

/* -------------------------------------------------------------------------- */
/* Bucle principal                                                            */
/* -------------------------------------------------------------------------- */

let lastT = performance.now();
let fpsAcc = 0;
let fpsFrames = 0;

function trainFrame() {
  const net = S.net;
  if (!net || S.order.length < 2) return;
  const budget = 11;
  const t0 = performance.now();
  const fresh = [];
  for (let e = 0; e < S.epf; e++) {
    const remain = budget - (performance.now() - t0);
    const [loss, acc] = net.trainEpoch(S.pts, S.order, S.batch, Math.max(1.5, remain), nowMs);
    const rec = { loss, acc, tAcc: NaN };
    S.hist.push(rec);
    fresh.push(rec);
    if (performance.now() - t0 > budget) break;
  }
  if (fresh.length && S.testPct > 0) {
    const [tAcc] = net.score(S.pts, true);
    for (const rec of fresh) rec.tAcc = tAcc;
  }
  if (S.hist.length > 400) S.hist.splice(0, S.hist.length - 400);
}

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(now - lastT, 250);
  lastT = now;
  fpsAcc += dt;
  fpsFrames++;
  if (fpsAcc > 400) {
    S.fps = Math.round((fpsFrames * 1000) / fpsAcc);
    fpsAcc = 0;
    fpsFrames = 0;
    $("fps").textContent = `${S.fps} fps`;
  }
  if (S.playing) flowPhase = (flowPhase + dt * 0.02) % 120;

  if (S.playing) trainFrame();
  S.frame++;
  if (needsGrid() && (S.frame % 2 === 0 || S.hover)) computeGrid();
  renderField();
  drawNetDiagram();
  drawChart();
  updateReadouts();
  updateCaption();
}

function updateCaption() {
  const sum = S.show.points ? fieldSummary() : null;
  const sub = $("cap-sub");
  if (!sub) return;
  if (!sum) { sub.textContent = "la línea blanca es la frontera de decisión (p = 0.5)"; return; }
  const bits = [`${sum.misses} de ${sum.n} mal clasificados`];
  if (sum.testMisses) bits.push(`${sum.testMisses} en test`);
  bits.push(`confianza media ${(sum.conf * 100).toFixed(0)} %`);
  sub.textContent = bits.join(" · ");
  sub.style.color = sum.misses > sum.n * 0.12 ? "rgba(255,107,138,.85)" : "";
}

function updateReadouts() {
  const net = S.net;
  const last = S.hist[S.hist.length - 1];
  const pct = (v) => (Number.isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`);
  $("hud-epoch").textContent = net ? net.epoch : 0;
  $("hud-loss").textContent = last ? last.loss.toFixed(4) : "—";
  $("hud-acc").textContent = last ? pct(last.acc) : "—";
  $("hud-test").textContent = last ? pct(last.tAcc) : "—";
  $("s-params").textContent = net ? net.nParams : "—";
  $("s-updates").textContent = net ? net.t : 0;
  $("s-loss").textContent = last ? last.loss.toFixed(4) : "—";
  $("s-acc").textContent = last ? pct(last.acc) : "—";
  $("s-gmax").textContent = net ? net.maxGrad.toFixed(4) : "—";
  $("s-w").textContent = net
    ? (net.layers.reduce((s, l) => s + (l.meanWeight ?? 0), 0) / net.layers.length).toFixed(4)
    : "—";
}

/* -------------------------------------------------------------------------- */
/* Construcción                                                               */
/* -------------------------------------------------------------------------- */

function buildNet() {
  const sizes = [2, ...S.hidden, 1];
  S.net = new MLP(sizes, { act: S.act, seed: S.seed, lr: S.lr, wd: S.wd });
  S.hist = [];
  $("net-tag").textContent = sizes.join(" → ");
  if (needsGrid()) computeGrid();
}

function shuffleOrder() {
  const train = [];
  for (let i = 0; i < S.pts.length; i++) if (!S.pts[i].test) train.push(i);
  for (let i = train.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [train[i], train[j]] = [train[j], train[i]];
  }
  S.order = train;
}

function rebuildData(resetWeights = true) {
  S.pts = makeDataset(S.ds, S.points, S.noise, S.seed, S.testPct);
  shuffleOrder();
  if (resetWeights) buildNet();
  else if (needsGrid()) computeGrid();
  updateDsTag();
  saveState();
}

function updateDsTag() {
  const nTest = S.pts.filter((p) => p.test).length;
  $("ds-tag").textContent = S.ds === "hand" && !S.pts.length
    ? "0 puntos · pinta tú"
    : `${S.pts.length} puntos · ${nTest} de test`;
  $("ds-note").textContent = DS_NOTES[S.ds] || "";
}

/* -------------------------------------------------------------------------- */
/* Controles                                                                  */
/* -------------------------------------------------------------------------- */

function toast(msg) {
  let t = $("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.remove("show"), 2200);
}

function setPlayUI() {
  $("btn-play").textContent = S.playing ? "⏸ pausa" : "▶ entrenar";
  $("hud-state").textContent = S.playing ? "entrenando" : "en pausa";
  const chip = $("led").closest(".chip");
  if (chip) chip.dataset.off = S.playing ? "0" : "1";
}

function togglePlay() {
  S.playing = !S.playing;
  setPlayUI();
}

function setBrush(b) {
  S.brush = b;
  document.querySelectorAll(".brush").forEach((el) => {
    const v = el.dataset.brush === "erase" ? "erase" : Number(el.dataset.brush);
    el.classList.toggle("active", v === b);
  });
}

function setActive(container, btn) {
  container.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
}

function renderHiddenCtl() {
  const host = $("hidden-ctl");
  host.innerHTML = "";
  S.hidden.forEach((v, i) => {
    const chip = document.createElement("span");
    chip.className = "hid-layer";
    chip.innerHTML =
      `<span class="ly">L${i + 1}</span>` +
      `<button data-d="-1" title="menos neuronas">−</button><b>${v}</b>` +
      `<button data-d="1" title="más neuronas">+</button>` +
      `<button class="hid-del" title="quitar capa">✕</button>`;
    chip.querySelectorAll("button[data-d]").forEach((b) =>
      b.addEventListener("click", () => {
        S.hidden[i] = clamp(S.hidden[i] + Number(b.dataset.d), 1, 24);
        renderHiddenCtl();
        buildNet();
        saveState();
      })
    );
    chip.querySelector(".hid-del").addEventListener("click", () => {
      S.hidden.splice(i, 1);
      renderHiddenCtl();
      buildNet();
      saveState();
    });
    host.appendChild(chip);
  });
  const add = document.createElement("button");
  add.className = "hid-add";
  add.textContent = S.hidden.length ? "+ capa" : "sin capa oculta";
  add.title = S.hidden.length >= 4 ? "máximo 4 capas" : "añadir capa oculta";
  add.addEventListener("click", () => {
    if (S.hidden.length >= 4) {
      toast("4 capas ya es sufrir");
      return;
    }
    S.hidden.push(S.hidden.length ? Math.max(4, Math.round(S.hidden[S.hidden.length - 1] / 2)) : 8);
    renderHiddenCtl();
    buildNet();
    saveState();
  });
  host.appendChild(add);
  $("o-hidden").textContent = S.hidden.length ? S.hidden.join(", ") : "ninguna";
}

function bindRange(id, outId, fmt, apply) {
  const el = $(id);
  const out = $(outId);
  el.addEventListener("input", () => {
    const v = Number(el.value);
    out.textContent = fmt(v);
    apply(v);
  });
}

/* ---- pintar / borrar ---- */

let painting = false;
let lastPaint = null;

function pointerPos(e) {
  const r = field.getBoundingClientRect();
  return toWorld(((e.clientX - r.left) * DPR), ((e.clientY - r.top) * DPR));
}

function addPoint(x, y, t) {
  S.pts.push({ x: clamp(x, -LIMIT, LIMIT), y: clamp(y, -LIMIT, LIMIT), t, test: false });
  if (S.pts.length > 4000) S.pts.shift();
}

function eraseAt(x, y) {
  let best = -1;
  let bd = ((20 * DPR) / SCALE) ** 2;
  for (let i = 0; i < S.pts.length; i++) {
    const p = S.pts[i];
    const d = (p.x - x) ** 2 + (p.y - y) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0) S.pts.splice(best, 1);
  shuffleOrder();
}

function wireField() {
  field.addEventListener("pointerdown", (e) => {
    const [x, y] = pointerPos(e);
    if (e.button === 2 || e.altKey || S.brush === "erase") {
      eraseAt(x, y);
      updateDsTag();
      saveState();
      return;
    }
    field.setPointerCapture(e.pointerId);
    painting = true;
    lastPaint = [x, y];
    addPoint(x, y, S.brush);
    switchToHand();
    shuffleOrder();
    if (!S.playing && needsGrid()) computeGrid();
  });

  field.addEventListener("pointermove", (e) => {
    const [x, y] = pointerPos(e);
    const p = S.net ? S.net.forward(x, y) : 0.5;
    S.hover = { x, y, p };
    $("probe").hidden = false;
    $("probe-x").textContent = x.toFixed(2);
    $("probe-y").textContent = y.toFixed(2);
    $("probe-p").textContent = p.toFixed(3);
    const v = $("probe-v");
    v.textContent = p >= 0.5 ? "→ clase B" : "→ clase A";
    v.classList.toggle("b", p >= 0.5);
    if (!painting) return;
    if (lastPaint && Math.hypot(x - lastPaint[0], y - lastPaint[1]) * SCALE < 12 * DPR) return;
    lastPaint = [x, y];
    addPoint(x, y, S.brush);
    shuffleOrder();
    updateDsTag();
  });

  const stop = () => {
    if (!painting) return;
    painting = false;
    lastPaint = null;
    updateDsTag();
    saveState();
  };
  field.addEventListener("pointerup", stop);
  field.addEventListener("pointercancel", stop);
  field.addEventListener("pointerleave", () => {
    S.hover = null;
    $("probe").hidden = true;
  });
  field.addEventListener("contextmenu", (e) => e.preventDefault());
}

function switchToHand() {
  if (S.ds === "hand") return;
  S.ds = "hand";
  setActive($("ds-grid"), $('ds-grid').querySelector('[data-ds="hand"]'));
}

/* ---- importar / exportar ---- */

function exportJSON() {
  if (!S.net) return;
  const payload = {
    ...S.net.toDict(),
    dataset: {
      name: S.ds, seed: S.seed, noise: S.noise, points: S.points,
      X: S.pts.map((p) => [+p.x.toFixed(5), +p.y.toFixed(5)]),
      y: S.pts.map((p) => p.t),
    },
  };
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ias-modelo-${S.net.epoch}epocas.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  toast("pesos exportados · formato ias-mlp/1");
}

function importJSON(file) {
  file.text().then(
    (txt) => {
      let d;
      try { d = JSON.parse(txt); } catch { return toast("JSON inválido"); }
      try {
        S.net = MLP.fromDict(d, S.seed);
      } catch (err) {
        return toast("no es un modelo ias-mlp/1 válido: " + err.message);
      }
      S.hidden = d.sizes.slice(1, -1);
      S.act = d.hiddenAct || "tanh";
      const ds = d.dataset;
      if (ds && Array.isArray(ds.X) && ds.X.length) {
        const ys = ds.y || [];
        S.pts = ds.X.map(([x, y], i) => ({ x, y, t: ys[i] ?? 0, test: false }));
        const nTest = Math.round((S.pts.length * S.testPct) / 100);
        for (let i = 0; i < nTest; i++) {
          S.pts[Math.min(Math.floor((i * S.pts.length) / Math.max(nTest, 1)), S.pts.length - 1)].test = true;
        }
        S.ds = "hand";
        shuffleOrder();
      }
      S.hist = [];
      S.playing = false;
      setPlayUI();
      renderHiddenCtl();
      syncControls();
      if (needsGrid()) computeGrid();
      updateDsTag();
      toast(`modelo cargado · ${S.net.nParams} parámetros · ${S.net.epoch} épocas previas`);
    },
    () => toast("no se pudo leer el archivo")
  );
}

/* -------------------------------------------------------------------------- */
/* Persistencia y enlace compartible                                          */
/* -------------------------------------------------------------------------- */

const LS_KEY = "ias.lab.v1";

function pick() {
  return {
    ds: S.ds, points: S.points, noise: S.noise, seed: S.seed, hidden: S.hidden,
    act: S.act, lr: S.lr, batch: S.batch, epf: S.epf, testPct: S.testPct, wd: S.wd,
  };
}

function saveState() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      ...pick(),
      show: S.show,
      pts: S.pts.length && S.pts.length <= 3000
        ? S.pts.map((p) => [+p.x.toFixed(4), +p.y.toFixed(4), p.t, p.test ? 1 : 0])
        : null,
    }));
  } catch { /* sin localStorage (modo privado): seguimos sin él */ }
}

function encodeState() {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(pick())) {
    q.set(k, Array.isArray(v) ? v.join(".") : v);
  }
  return `#${q.toString()}`;
}

function decodeHash() {
  if (location.hash.length < 4) return false;
  const q = new URLSearchParams(location.hash.slice(1));
  const get = (k, conv) => (q.has(k) ? conv(q.get(k)) : undefined);
  const num = (v, d) => (v === undefined || Number.isNaN(Number(v)) ? d : Number(v));
  S.ds = get("ds", String) ?? S.ds;
  S.points = num(get("points", Number), S.points);
  S.noise = num(get("noise", Number), S.noise);
  S.seed = num(get("seed", Number), S.seed);
  S.act = get("act", String) ?? S.act;
  S.lr = num(get("lr", Number), S.lr);
  S.batch = num(get("batch", Number), S.batch);
  S.epf = num(get("epf", Number), S.epf);
  S.testPct = num(get("testPct", Number), S.testPct);
  S.wd = num(get("wd", Number), S.wd);
  const hidden = get("hidden", String);
  if (hidden) {
    const arr = hidden.split(/[.,]/).map(Number).filter((v) => v > 0).slice(0, 4);
    S.hidden = arr.map((v) => clamp(Math.round(v), 1, 24));
  }
  if (!["blobs", "xor", "circle", "moons", "spiral", "checker", "sine", "hand"].includes(S.ds)) S.ds = "spiral";
  if (!["tanh", "relu", "leaky", "elu"].includes(S.act)) S.act = "tanh";
  return true;
}

function loadStored() {
  let raw = null;
  try { raw = localStorage.getItem(LS_KEY); } catch { return false; }
  if (!raw) return false;
  try {
    const d = JSON.parse(raw);
    const { pts, show, ...rest } = d;
    Object.assign(S, rest);
    if (d.show) Object.assign(S.show, d.show);
    if (!Array.isArray(S.hidden)) S.hidden = [...DEFAULTS.hidden];
    if (Array.isArray(pts) && pts.length) {
      S.pts = pts.map(([x, y, t, test]) => ({ x, y, t, test: !!test }));
      shuffleOrder();
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/* -------------------------------------------------------------------------- */
/* Sincronizar controles ← estado                                             */
/* -------------------------------------------------------------------------- */

const TOGGLE_IDS = { "t-heat": "heat", "t-contour": "contour", "t-iso": "iso", "t-points": "points", "t-misses": "misses", "t-flow": "flow", "t-grid": "grid" };

function syncControls() {
  $("i-lr").value = String(Math.log10(S.lr).toFixed(3));
  $("o-lr").textContent = S.lr.toFixed(4);
  $("i-batch").value = String(Math.round(Math.log2(S.batch)) + 1);
  $("o-batch").textContent = S.batch;
  $("i-epf").value = String(S.epf);
  $("o-epf").textContent = S.epf;
  $("i-points").value = String(S.points);
  $("o-points").textContent = S.points;
  $("i-noise").value = String(S.noise);
  $("o-noise").textContent = Number(S.noise).toFixed(2);
  $("i-test").value = String(S.testPct);
  $("o-test").textContent = `${S.testPct}%`;
  $("i-wd").value = String(Math.log10(S.wd || 1e-6).toFixed(2));
  $("o-wd").textContent = Number(S.wd).toFixed(5);
  $("act-seg").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.act === S.act));
  $("o-act").textContent = S.act;
  const dsBtn = $("ds-grid").querySelector(`[data-ds="${S.ds}"]`);
  if (dsBtn) setActive($("ds-grid"), dsBtn);
  for (const [id, key] of Object.entries(TOGGLE_IDS)) $(id).checked = !!S.show[key];
  renderHiddenCtl();
}

/* -------------------------------------------------------------------------- */
/* Interactividad                                                             */
/* -------------------------------------------------------------------------- */

function wireUI() {
  $("ds-grid").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-ds]");
    if (!b) return;
    setActive($("ds-grid"), b);
    S.ds = b.dataset.ds;
    S.playing = true;
    setPlayUI();
    if (S.ds === "hand") {
      S.pts = [];
      S.order = [];
      S.hist = [];
      buildNet();
      updateDsTag();
      saveState();
    } else {
      rebuildData(true);
    }
  });

  bindRange("i-lr", "o-lr", (v) => Math.pow(10, v).toFixed(4), (v) => {
    S.lr = Math.pow(10, v);
    if (S.net) S.net.lr = S.lr;
    saveState();
  });
  bindRange("i-batch", "o-batch", (v) => String(2 ** (v - 1)), (v) => {
    S.batch = 2 ** (v - 1);
    saveState();
  });
  bindRange("i-epf", "o-epf", (v) => String(v), (v) => { S.epf = v; saveState(); });
  bindRange("i-points", "o-points", (v) => String(v), (v) => {
    S.points = v;
    if (S.ds !== "hand") rebuildData(false);
  });
  bindRange("i-noise", "o-noise", (v) => v.toFixed(2), (v) => {
    S.noise = v;
    if (S.ds !== "hand") rebuildData(false);
  });
  bindRange("i-test", "o-test", (v) => `${v}%`, (v) => {
    S.testPct = v;
    if (S.ds !== "hand") rebuildData(false);
    else {
      S.pts.forEach((p) => (p.test = false));
      shuffleOrder();
    }
  });
  bindRange("i-wd", "o-wd", (v) => Math.pow(10, v).toFixed(5), (v) => {
    S.wd = Math.pow(10, v);
    if (S.net) S.net.wd = S.wd;
    saveState();
  });

  $("act-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b) return;
    setActive($("act-seg"), b);
    S.act = b.dataset.act;
    $("o-act").textContent = S.act;
    buildNet();
    saveState();
  });

  $("btn-defaults").addEventListener("click", () => {
    Object.assign(S, DEFAULTS, { hidden: [...DEFAULTS.hidden] });
    S.show = { ...SHOW_DEFAULTS };
    syncControls();
    rebuildData(true);
    toast("hiperparámetros restaurados");
  });

  $("btn-play").addEventListener("click", togglePlay);
  $("btn-step").addEventListener("click", () => {
    if (!S.net || S.order.length < 2) return toast("pinta al menos dos puntos o carga un dataset");
    const [loss, acc] = S.net.trainEpoch(S.pts, S.order, S.batch, Infinity, () => 0);
    const tAcc = S.testPct > 0 ? S.net.score(S.pts, true)[0] : NaN;
    S.hist.push({ loss, acc, tAcc });
    if (needsGrid()) computeGrid();
  });
  $("btn-reset").addEventListener("click", () => {
    S.seed = (S.seed * 7 + 13) % 100003;
    buildNet();
    toast("pesos reiniciados · otra inicialización");
    saveState();
  });
  $("btn-clear").addEventListener("click", () => {
    S.pts = [];
    S.order = [];
    S.hist = [];
    if (S.net) S.net.epoch = 0;
    switchToHand();
    if (needsGrid()) computeGrid();
    updateDsTag();
    saveState();
    toast("plano vacío — pinta tus propios puntos");
  });

  for (const [id, key] of Object.entries(TOGGLE_IDS)) {
    const cb = $(id);
    cb.checked = !!S.show[key];
    cb.addEventListener("change", () => {
      S.show[key] = cb.checked;
      if (needsGrid()) computeGrid();
      saveState();
    });
  }

  $("brush-box").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-brush]");
    if (!b) return;
    setBrush(b.dataset.brush === "erase" ? "erase" : Number(b.dataset.brush));
  });

  $("btn-export-json").addEventListener("click", exportJSON);
  $("btn-import-json").addEventListener("click", () => $("file-json").click());
  $("file-json").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) importJSON(f);
    e.target.value = "";
  });
  $("btn-export-png").addEventListener("click", () => {
    field.toBlob((blob) => {
      if (!blob) return toast("no se pudo generar el PNG");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `ias-${S.ds}-${S.net ? S.net.epoch : 0}-epocas.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    });
  });
  $("btn-share").addEventListener("click", async () => {
    const url = encodeState();
    history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(location.href);
      toast("enlace copiado · otro navegador reproduce este estado");
    } catch {
      toast("la barra de direcciones ya tiene tu estado");
    }
  });

  [["btn-help", "modal-help"], ["btn-about", "modal-about"]].forEach(([btn, modal]) =>
    $(btn).addEventListener("click", () => { $(modal).hidden = false; })
  );
  document.querySelectorAll(".modal").forEach((m) =>
    m.addEventListener("click", (e) => {
      if (e.target === m || e.target.closest("[data-close]")) m.hidden = true;
    })
  );

  addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === " ") { e.preventDefault(); togglePlay(); }
    else if (k === ".") { e.preventDefault(); $("btn-step").click(); }
    else if (k === "r") $("btn-reset").click();
    else if (k === "c") $("btn-clear").click();
    else if (k === "d") {
      const btns = [...$("ds-grid").querySelectorAll("button")];
      const i = btns.findIndex((b) => b.dataset.ds === S.ds);
      btns[(i + 1 + btns.length) % btns.length].click();
    } else if (k === "1") setBrush(0);
    else if (k === "2") setBrush(1);
    else if (k === "3") setBrush("erase");
    else if (k === "f") {
      e.preventDefault();
      if (document.fullscreenElement) document.exitFullscreen();
      else if ($("stage").requestFullscreen) $("stage").requestFullscreen().catch(() => toast("pantalla completa no disponible"));
    } else if (k === "?" || k === "/") {
      e.preventDefault();
      $("modal-help").hidden = !$("modal-help").hidden;
    } else if (k === "escape") document.querySelectorAll(".modal").forEach((m) => { m.hidden = true; });
  });

  let rTimer = 0;
  addEventListener("resize", () => {
    clearTimeout(rTimer);
    rTimer = setTimeout(() => {
      resize();
      if (needsGrid()) computeGrid();
    }, 90);
  });
  if (window.ResizeObserver) new ResizeObserver(() => resize()).observe(field);
}

/* -------------------------------------------------------------------------- */
/* Arranque                                                                   */
/* -------------------------------------------------------------------------- */

function boot() {
  const fromHash = decodeHash();
  if (!fromHash) loadStored();
  resize();
  wireUI();
  wireField();
  setBrush(S.brush);
  syncControls();

  if (!S.pts.length) S.pts = makeDataset(S.ds, S.points, S.noise, S.seed, S.testPct);
  shuffleOrder();
  buildNet();
  S.net.epoch = 0;
  setPlayUI();
  updateDsTag();
  if (needsGrid()) computeGrid();
  requestAnimationFrame(loop);
  setInterval(() => { if (S.pts.length) saveState(); }, 7000);
  addEventListener("beforeunload", saveState);
}

/* Asa de depuración: desde la consola del navegador (y desde los tests de
   Node con un DOM falso) se puede inspeccionar y pilotar el laboratorio. */
globalThis.IASLab = {
  S, get net() { return S.net; }, buildNet, computeGrid, resize, renderField,
  drawChart, drawNetDiagram, trainFrame, shuffleOrder, encodeState, decodeHash,
  /** mundo → coordenadas del puntero (lo usan los tests y la consola) */
  clientOf: (x, y) => {
    const [sx, sy] = toScreen(x, y);
    const r = field.getBoundingClientRect();
    return [sx / DPR + r.left, sy / DPR + r.top];
  },
};

if (typeof document !== "undefined" && document.getElementById("field")) boot();
