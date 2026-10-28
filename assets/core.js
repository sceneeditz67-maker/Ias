/* ==========================================================================
   IAS · core.js — el motor de la red, sin una sola línea de DOM
   --------------------------------------------------------------------------
   Multilayer perceptron para clasificación binaria 2D:
     forward · backprop (regla de la cadena) · Adam · BCE · datasets · marching squares

   Se usa de dos formas:
     · navegador  →  <script src="assets/core.js">  expone `globalThis.IAS`
     · Node/tests →  `require("./assets/core.js")`  (ver tests/test_core.js)

   Es el mismo algoritmo que `python/ias_net.py`, capa por capa.
   Licencia MIT.
   ========================================================================== */

"use strict";

/* IIFE: así `core.js` no contamina el ámbito léxico global del navegador. */
const IAS = (() => {

const CL = 30; // corte de exponente: evita overflow en Math.exp
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const TAU = Math.PI * 2;

/* -------------------------------------------------------------------------- */
/* PRNG determinista + gaussianas                                             */
/* -------------------------------------------------------------------------- */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller: dos gaussianas estándar por llamada (como en el módulo Python). */
function gaussPair(rnd) {
  const u = Math.max(rnd(), 1e-12);
  const r = Math.sqrt(-2 * Math.log(u));
  const t = TAU * rnd();
  return [r * Math.cos(t), r * Math.sin(t)];
}

/* -------------------------------------------------------------------------- */
/* Activaciones                                                               */
/* -------------------------------------------------------------------------- */

const sig = (z) => 1 / (1 + Math.exp(-clamp(z, -CL, CL)));

const ACTS = {
  tanh: { f: Math.tanh, d: (z) => 1 - Math.tanh(clamp(z, -CL, CL)) ** 2 },
  relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
  leaky: { f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01) },
  elu: { f: (z) => (z > 0 ? z : Math.exp(clamp(z, -CL, CL)) - 1),
         d: (z) => (z > 0 ? 1 : Math.exp(clamp(z, -CL, CL))) },
};
const RELUISH = new Set(["relu", "leaky", "elu"]);

/** Pérdida de entropía cruzada binaria, con log recortado. */
function bce(p, t) {
  const pc = clamp(p, 1e-7, 1 - 1e-7);
  return -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
}

/* -------------------------------------------------------------------------- */
/* Capa densa                                                                 */
/* -------------------------------------------------------------------------- */

class Layer {
  constructor(nIn, nOut, act, isOut, rnd) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.act = act;
    this.isOut = isOut;
    const n = nOut * nIn;
    this.w = new Float64Array(n);
    this.b = new Float64Array(nOut);
    this.gw = new Float64Array(n);
    this.gb = new Float64Array(nOut);
    this.mw = new Float64Array(n);
    this.vw = new Float64Array(n);
    this.mb = new Float64Array(nOut);
    this.vb = new Float64Array(nOut);
    this.z = new Float64Array(nOut); // pre-activación
    this.a = new Float64Array(nOut); // activación
    this.maxWeight = 1;
    this.meanWeight = 0;

    // He (relu-ish) / Xavier (tanh)
    const scale = RELUISH.has(act) ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
    for (let i = 0; i < n; i++) {
      const [g] = gaussPair(rnd);
      this.w[i] = g * scale;
    }
  }

  /** Lee `aIn` y escribe this.z / this.a. */
  forward(aIn) {
    const { nIn, nOut, w, b, z, a, isOut } = this;
    for (let o = 0; o < nOut; o++) {
      let s = b[o];
      const base = o * nIn;
      for (let i = 0; i < nIn; i++) s += w[base + i] * aIn[i];
      z[o] = s;
    }
    if (isOut) {
      for (let o = 0; o < nOut; o++) a[o] = sig(z[o]);
    } else {
      const f = ACTS[this.act].f;
      for (let o = 0; o < nOut; o++) a[o] = f(z[o]);
    }
  }

  /** Adam con sesgo corregido + weight decay desacoplado (L2 sobre pesos). */
  step(lr, t, b1, b2, wd) {
    const bc1 = 1 - b1 ** t;
    const bc2 = 1 - b2 ** t;
    const { w, gw, mw, vw, b, gb, mb, vb } = this;
    for (let i = 0; i < w.length; i++) {
      const g = gw[i] + (wd ? wd * w[i] : 0);
      mw[i] = b1 * mw[i] + (1 - b1) * g;
      vw[i] = b2 * vw[i] + (1 - b2) * g * g;
      w[i] -= (lr * (mw[i] / bc1)) / (Math.sqrt(vw[i] / bc2) + 1e-8);
    }
    for (let i = 0; i < b.length; i++) {
      mb[i] = b1 * mb[i] + (1 - b1) * gb[i];
      vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i];
      b[i] -= (lr * (mb[i] / bc1)) / (Math.sqrt(vb[i] / bc2) + 1e-8);
    }
    let mx = 0;
    let sum = 0;
    for (let i = 0; i < w.length; i++) {
      const v = Math.abs(w[i]);
      if (v > mx) mx = v;
      sum += v;
    }
    this.maxWeight = mx || 1;
    this.meanWeight = w.length ? sum / w.length : 0;
  }

  resetGrads() {
    this.gw.fill(0);
    this.gb.fill(0);
  }

  toDict() {
    return { nIn: this.nIn, nOut: this.nOut, w: Array.from(this.w), b: Array.from(this.b) };
  }
}

/* -------------------------------------------------------------------------- */
/* Red                                                                        */
/* -------------------------------------------------------------------------- */

class MLP {
  /** sizes = [2, h1, h2, …, 1] */
  constructor(sizes, { act = "tanh", seed = 7, lr = 0.02, wd = 0 } = {}) {
    if (sizes.length < 2) throw new Error("hace falta [n_in, n_out]");
    this.sizes = sizes.slice();
    this.act = act;
    this.lr = lr;
    this.wd = wd;
    this.rnd = mulberry32(seed);
    const last = sizes.length - 2;
    this.layers = [];
    for (let i = 0; i <= last; i++) {
      this.layers.push(new Layer(sizes[i], sizes[i + 1], act, i === last, this.rnd));
    }
    this.xbuf = new Float64Array(sizes[0]);
    this.deltas = null; // se asigna al primer backward
    this.t = 0;
    this.epoch = 0;
    this.maxGrad = 0;
    this.history = [];
  }

  get nParams() {
    let s = 0;
    for (const l of this.layers) s += l.w.length + l.b.length;
    return s;
  }

  /** p(clase B | x, y). Deja las activaciones en layer.a (para el diagrama). */
  forward(x, y) {
    const buf = this.xbuf;
    buf[0] = x;
    buf[1] = y;
    let a = buf;
    const L = this.layers;
    for (let i = 0; i < L.length; i++) {
      L[i].forward(a);
      a = L[i].a;
    }
    return a[0];
  }

  /** Activación de cada capa para la entrada dada (diagrama de la red). */
  activations(x, y) {
    this.forward(x, y);
    return this.layers.map((l) => l.a);
  }

  /** Backprop de un ejemplo: acumula en gw/gb. Devuelve [loss, ¿acierta?]. */
  accumulate(x, y, t) {
    const p = this.forward(x, y);
    const loss = bce(p, t);
    const L = this.layers;
    if (!this.deltas || this.deltas.length !== L.length) {
      this.deltas = L.map((l) => new Float64Array(l.nOut));
    }
    this.deltas[L.length - 1][0] = p - t; // sigmoid + BCE ⇒ la derivada es (p − t)
    for (let li = L.length - 1; li >= 0; li--) {
      const layer = L[li];
      const { nIn, nOut, w, gw, gb } = layer;
      const cur = this.deltas[li];
      const aIn = li === 0 ? this.xbuf : L[li - 1].a;
      for (let o = 0; o < nOut; o++) {
        const d = cur[o];
        gb[o] += d;
        const base = o * nIn;
        for (let i = 0; i < nIn; i++) gw[base + i] += d * aIn[i];
      }
      if (li > 0) {
        // dL/dz_{li-1} = (W_liᵀ · delta) ⊙ f'(z_{li-1})
        const prev = L[li - 1];
        const dact = ACTS[prev.act].d;
        const dPrev = this.deltas[li - 1];
        const zPrev = prev.z;
        dPrev.fill(0, 0, nIn);
        for (let o = 0; o < nOut; o++) {
          const d = cur[o];
          if (d === 0) continue;
          const base = o * nIn;
          for (let i = 0; i < nIn; i++) dPrev[i] += d * w[base + i];
        }
        for (let i = 0; i < nIn; i++) dPrev[i] *= dact(zPrev[i]);
      }
    }
    return [loss, p >= 0.5 === (t >= 0.5)];
  }

  /** Media el gradiente del batch, recorta por norma ∞ y aplica Adam. */
  applyGrads(n, clip = 5) {
    const k = 1 / Math.max(n, 1);
    let gmax = 0;
    for (const layer of this.layers) {
      const { gw, gb } = layer;
      for (let i = 0; i < gw.length; i++) {
        gw[i] *= k;
        const v = gw[i] < 0 ? -gw[i] : gw[i];
        if (v > gmax) gmax = v;
      }
      for (let i = 0; i < gb.length; i++) {
        gb[i] *= k;
        const v = gb[i] < 0 ? -gb[i] : gb[i];
        if (v > gmax) gmax = v;
      }
    }
    if (clip > 0 && gmax > clip) {
      const s = clip / gmax;
      for (const layer of this.layers) {
        for (let i = 0; i < layer.gw.length; i++) layer.gw[i] *= s;
        for (let i = 0; i < layer.gb.length; i++) layer.gb[i] *= s;
      }
    }
    this.maxGrad = gmax;
    this.t += 1;
    for (const layer of this.layers) layer.step(this.lr, this.t, 0.9, 0.999, this.wd);
    for (const layer of this.layers) layer.resetGrads();
  }

  /**
   * Una época sobre `order` (índices de entrenamiento barajados).
   * `budgetMs` corta si el fotograma amenaza con pasarse de 16 ms.
   */
  trainEpoch(pts, order, batch = 0, budgetMs = Infinity, now = () => 0) {
    const n = order.length;
    if (!n) return [0, 1, 0];
    const size = batch > 0 ? Math.min(batch, n) : n;
    let loss = 0;
    let good = 0;
    let used = 0;
    const t0 = now();
    for (let start = 0; start < n; start += size) {
      const end = Math.min(start + size, n);
      for (let j = start; j < end; j++) {
        const p = pts[order[j]];
        const r = this.accumulate(p.x, p.y, p.t);
        loss += r[0];
        good += r[1] ? 1 : 0;
      }
      this.applyGrads(end - start);
      used += end - start;
      if (budgetMs < Infinity && now() - t0 > budgetMs) break;
    }
    this.epoch += 1;
    return [loss / used, good / used, used];
  }

  /** Exactitud/pérdida sobre train (wantTest=false) o test (true). */
  score(pts, wantTest) {
    let good = 0, loss = 0, m = 0;
    for (const p of pts) {
      if (!!p.test !== !!wantTest) continue;
      const pr = this.forward(p.x, p.y);
      loss += bce(pr, p.t);
      good += pr >= 0.5 === (p.t >= 0.5) ? 1 : 0;
      m++;
    }
    return m ? [good / m, loss / m, m] : [NaN, NaN, 0];
  }

  /* -------------------------- ias-mlp/1 (JSON) --------------------------- */

  toDict(precision = 8) {
    return {
      format: "ias-mlp/1",
      sizes: this.sizes,
      hiddenAct: this.act,
      lr: this.lr,
      epochs: this.epoch,
      layers: this.layers.map((l) => ({
        nIn: l.nIn,
        nOut: l.nOut,
        w: Array.from(l.w, (v) => +v.toFixed(precision)),
        b: Array.from(l.b, (v) => +v.toFixed(precision)),
      })),
    };
  }

  static fromDict(d, seed = 99) {
    if (d.format && d.format !== "ias-mlp/1") throw new Error("formato desconocido: " + d.format);
    const sizes = d.sizes;
    if (!Array.isArray(sizes) || sizes.length < 2) throw new Error("falta `sizes`");
    const net = new MLP(sizes, { act: d.hiddenAct || "tanh", seed, lr: d.lr ?? 0.02 });
    (d.layers || []).forEach((src, i) => {
      const L = net.layers[i];
      if (!L) throw new Error("capa de más en el JSON");
      if (L.nIn !== src.nIn || L.nOut !== src.nOut) throw new Error(`forma de capa ${i} incompatible`);
      if (src.w.length !== L.w.length || src.b.length !== L.b.length) throw new Error(`tamaño de vector en capa ${i}`);
      L.w.set(src.w);
      L.b.set(src.b);
      L.maxWeight = Math.max(1, ...Array.from(L.w, Math.abs));
    });
    net.epoch = d.epochs || 0;
    return net;
  }

  /* ------------------------ gradiente numérico (tests) ------------------- */

  /** ∂BCE/∂w de la capa `li`, por diferencias centrales. */
  numericGradWeight(x, y, t, li, eps = 1e-6) {
    const L = this.layers[li];
    const out = new Float64Array(L.w.length);
    for (let i = 0; i < L.w.length; i++) {
      const save = L.w[i];
      L.w[i] = save + eps;
      const up = bce(this.forward(x, y), t);
      L.w[i] = save - eps;
      const dn = bce(this.forward(x, y), t);
      L.w[i] = save;
      out[i] = (up - dn) / (2 * eps);
    }
    return out;
  }
}

/* -------------------------------------------------------------------------- */
/* Datasets sintéticos (espejo de make_dataset en ias_net.py)                 */
/* -------------------------------------------------------------------------- */

const LIMIT = 1.36;
const DATASETS = ["blobs", "xor", "circle", "moons", "spiral", "checker", "sine", "hand"];

/** Devuelve [{x, y, t, test}] en coordenadas de mundo [-LIMIT, LIMIT]. */
function makeDataset(name, n = 240, noise = 0.07, seed = 7, testPct = 0) {
  const rnd = mulberry32(seed);
  const pts = [];
  const push = (px, py, t) => {
    const [gx, gy] = gaussPair(rnd);
    pts.push({
      x: clamp(px + gx * noise, -LIMIT, LIMIT),
      y: clamp(py + gy * noise, -LIMIT, LIMIT),
      t,
      test: false,
    });
  };
  const half = Math.floor(n / 2);

  switch (name) {
    case "blobs":
      for (let k = 0; k < 2; k++) {
        const cx = k ? 0.45 : -0.45;
        for (let i = 0; i < half; i++) {
          const [gx, gy] = gaussPair(rnd);
          push(cx + gx * 0.25, gy * 0.25, k);
        }
      }
      break;
    case "xor":
      for (const sx of [-1, 1])
        for (const sy of [-1, 1]) {
          const t = sx * sy > 0 ? 1 : 0;
          for (let i = 0; i < n / 4; i++) {
            const [gx, gy] = gaussPair(rnd);
            push(sx * 0.45 + gx * 0.2, sy * 0.45 + gy * 0.2, t);
          }
        }
      break;
    case "circle":
    case "rings3": {
      const radii = name === "circle" ? [0.28, 0.72] : [0.2, 0.52, 0.85];
      const per = Math.floor(n / radii.length);
      radii.forEach((r, k) => {
        for (let i = 0; i < per; i++) {
          const th = (TAU * i) / per + rnd() * 0.35;
          push(r * Math.cos(th), r * Math.sin(th), k % 2);
        }
      });
      break;
    }
    case "moons":
      for (let i = 0; i < half; i++) {
        const th = (Math.PI * i) / half;
        push(Math.cos(th) * 0.7, Math.sin(th) * 0.55 - 0.22, 0);
      }
      for (let i = 0; i < half; i++) {
        const th = (Math.PI * i) / half;
        push(0.7 - Math.cos(th) * 0.7, 0.22 - Math.sin(th) * 0.55, 1);
      }
      break;
    case "spiral": {
      for (let k = 0; k < 2; k++)
        for (let i = 0; i < half; i++) {
          const u = i / half;
          const ang = 3.4 * Math.PI * u + (k * Math.PI) / 2;
          const r = 0.14 + 0.78 * u;
          push(r * Math.cos(ang), r * Math.sin(ang), k);
        }
      break;
    }
    case "checker": {
      const cells = 4;
      const per = Math.max(4, Math.floor(n / (cells * cells)));
      for (let k = 0; k < cells * cells; k++) {
        const cx = ((k % cells) / cells) * 2 - 1 + 1 / cells;
        const cy = (Math.floor(k / cells) / cells) * 2 - 1 + 1 / cells;
        const t = ((k % cells) + Math.floor(k / cells)) % 2 === 0 ? 1 : 0;
        for (let i = 0; i < per; i++) {
          const [gx, gy] = gaussPair(rnd);
          push(cx + gx * 0.16, cy + gy * 0.16, t);
        }
      }
      break;
    }
    case "sine":
      for (let s = 0; s < 2; s++)
        for (let i = 0; i < half; i++) {
          const px = -1 + (2 * i) / half;
          push(px, 0.55 * Math.sin(px * 3.6) + (s ? 0.2 : -0.2), s);
        }
      break;
    case "hand":
      return [];
    default:
      throw new Error("dataset desconocido: " + name);
  }

  if (testPct > 0) {
    const nTest = Math.round((pts.length * testPct) / 100);
    for (let i = 0; i < nTest; i++) {
      const idx = Math.floor((i * pts.length) / Math.max(nTest, 1));
      pts[Math.min(idx, pts.length - 1)].test = true;
    }
  }
  return pts;
}

/**
 * Marching squares sobre `grid` (Float32Array gy*nx+gx) buscando `level`.
 * Devuelve un array plano [x0,y0,x1,y1,…] en píxeles del lienzo.
 */
function isoSegments(grid, nx, ny, sx, sy, level) {
  const segs = [];
  const at = (gx, gy) => grid[gy * nx + gx] - level;
  for (let gy = 0; gy < ny - 1; gy++) {
    for (let gx = 0; gx < nx - 1; gx++) {
      const a = at(gx, gy);       // arriba-izq
      const b = at(gx + 1, gy);   // arriba-der
      const c = at(gx + 1, gy + 1); // abajo-der
      const e = at(gx, gy + 1);   // abajo-izq
      let mask = 0;
      if (a > 0) mask |= 8;
      if (b > 0) mask |= 4;
      if (c > 0) mask |= 2;
      if (e > 0) mask |= 1;
      if (mask === 0 || mask === 15) continue;
      const t = (v0, v1) => {
        const d = v0 - v1;
        return d === 0 ? 0.5 : clamp(v0 / d, 0, 1);
      };
      const pT = [(gx + t(a, b)) * sx, gy * sy];
      const pR = [(gx + 1) * sx, (gy + t(b, c)) * sy];
      const pB = [(gx + t(e, c)) * sx, (gy + 1) * sy];
      const pL = [gx * sx, (gy + t(a, e)) * sy];
      const seg = (p0, p1) => segs.push(p0[0], p0[1], p1[0], p1[1]);
      switch (mask) {
        case 1: case 14: seg(pL, pB); break;
        case 2: case 13: seg(pB, pR); break;
        case 3: case 12: seg(pL, pR); break;
        case 4: case 11: seg(pT, pR); break;
        case 6: case 9: seg(pT, pB); break;
        case 7: case 8: seg(pT, pL); break;
        case 5: seg(pT, pL); seg(pB, pR); break;
        case 10: seg(pT, pR); seg(pL, pB); break;
      }
    }
  }
  return segs;
}

/** LUT divergente A → neutro → B, 64 entradas RGB. */
function buildLUT() {
  const n = 64;
  const c0 = [24, 74, 158];
  const cm = [10, 15, 27];
  const c1 = [196, 92, 34];
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const tt = i / (n - 1);
    const [A, B, k] = tt < 0.5 ? [c0, cm, tt * 2] : [cm, c1, (tt - 0.5) * 2];
    for (let j = 0; j < 3; j++) out[i * 3 + j] = A[j] + (B[j] - A[j]) * k;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Exportación                                                                */
/* -------------------------------------------------------------------------- */

  return {
    mulberry32, gaussPair, clamp, sig, bce, ACTS, Layer, MLP,
    makeDataset, DATASETS, isoSegments, buildLUT, LIMIT,
  };
})();

if (typeof globalThis !== "undefined") globalThis.IAS = IAS;
if (typeof module !== "undefined" && module.exports) module.exports = IAS;
