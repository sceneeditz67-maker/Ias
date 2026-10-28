/* ==========================================================================
   mini-dom.js — un DOM suficiente para ejecutar la UI en Node
   --------------------------------------------------------------------------
   Parsea index.html de verdad (etiquetas, ids, clases, data-*, atributos de
   <input>) y añade lo que usa assets/app.js: listeners, eventos, canvas 2D,
   localStorage, rAF manual. No es un navegador: es una red de seguridad para
   detectar errores de ejecución y ids mal escritos antes de abrir el enlace.

   Lo usan tests/test_ui_smoke.js. Sin dependencias.
   ========================================================================== */

"use strict";

/* -------------------------------------------------------------------------- */
/* Elemento                                                                   */
/* -------------------------------------------------------------------------- */

class El {
  constructor(tag, attrs = {}, parent = null) {
    this.tagName = tag.toUpperCase();
    this.attrs = attrs;
    this.parent = parent;
    this.children = [];
    this._text = "";
    this.listeners = new Map();
    this.dataset = {};
    this.style = {};
    const classes = (attrs.class || "").split(/\s+/).filter(Boolean);
    const self = this;
    this.classList = {
      add: (c) => { if (!classes.includes(c)) classes.push(c); },
      remove: (c) => { const i = classes.indexOf(c); if (i >= 0) classes.splice(i, 1); },
      contains: (c) => classes.includes(c),
      toggle: (c, force) => {
        const on = force === undefined ? !classes.includes(c) : !!force;
        if (on && !classes.includes(c)) classes.push(c);
        if (!on) { const i = classes.indexOf(c); if (i >= 0) classes.splice(i, 1); }
        return on;
      },
      get value() { return classes.join(" "); },
    };
    Object.defineProperty(this, "className", {
      get: () => classes.join(" "),
      set: (v) => { classes.length = 0; classes.push(...String(v).split(/\s+/).filter(Boolean)); },
    });
    for (const [k, v] of Object.entries(attrs)) {
      if (k.startsWith("data-")) {
        const key = k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        this.dataset[key] = v === "" ? "true" : v;
      }
    }
    if (attrs.id) Object.defineProperty(this, "id", { get: () => attrs.id, set: (v) => (attrs.id = v), enumerable: true });
    if ("value" in attrs) this.value = attrs.value;
    if ("checked" in attrs) this.checked = attrs.checked !== "false";
    void self;
  }

  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); }
  set innerHTML(v) {
    this._text = String(v);
    this.children = [];
    const frag = parseHTML(String(v)); // mismo parser que el del documento
    const adopt = (node) => {
      node.parent = this;
      this.children.push(node);
      node.children.forEach(adopt);
    };
    frag.children.forEach(adopt);
  }
  get innerHTML() { return ""; }

  appendChild(child) { child.parent = this; this.children.push(child); return child; }

  /* ------------------------------- selección ---------------------------- */

  matches(sel) {
    return sel.split(",").map((s) => s.trim()).some((part) => matchOne(this, part));
  }

  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }

  querySelectorAll(sel) {
    const out = [];
    const parts = sel.split(",").map((s) => s.trim());
    const walk = (node) => {
      for (const c of node.children) {
        if (parts.some((p) => matchOne(c, p.split(/\s+/).pop()))) out.push(c);
        walk(c);
      }
    };
    walk(this);
    return out;
  }

  closest(sel) {
    const first = sel.split(",").map((s) => s.trim())[0].split(/\s+/).pop();
    let n = this;
    while (n) {
      if (matchOne(n, first)) return n;
      n = n.parent;
    }
    return null;
  }

  /* -------------------------------- eventos ----------------------------- */

  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  removeEventListener(type, fn) {
    const a = this.listeners.get(type) || [];
    const i = a.indexOf(fn);
    if (i >= 0) a.splice(i, 1);
  }
  /** Despacha el evento y lo deja burbujear por los padres, como un DOM real. */
  dispatch(type, ev = {}) {
    let node = this;
    while (node) {
      const handlers = node.listeners && node.listeners.get(type);
      if (handlers && handlers.length) {
        for (const fn of [...handlers]) {
          fn(makeEvent(type, ev, this, node));
          if (ev._stopped) return;
        }
      }
      node = node.parent;
    }
  }
  click() { this.dispatch("click"); }

  /* --------------------------------- varios ----------------------------- */

  getBoundingClientRect() {
    return { left: 0, top: 0, width: this._w || 900, height: this._h || 460, right: this._w || 900, bottom: this._h || 460 };
  }
  setPointerCapture() {}
  releasePointerCapture() {}
  focus() {}
  toBlob(cb) { cb(new BlobStub()); }
  getContext() {
    if (!this._ctx) this._ctx = makeCtx(this);
    return this._ctx;
  }
  scrollIntoView() {}
}

/** Blob mínimo: solo se usa para URL.createObjectURL. */
class BlobStub {
  constructor(parts = [], opts = {}) { this.parts = parts; this.type = opts.type || ""; }
  text() { return Promise.resolve(this.parts.join("")); }
}

function matchOne(el, sel) {
  // soporta tag, .clase, #id, [attr], [attr="v"], tag[attr="v"] y combinaciones
  let rest = sel;
  const tag = /^[a-zA-Z][\w-]*/.exec(rest);
  if (tag) {
    if (el.tagName !== tag[0].toUpperCase()) return false;
    rest = rest.slice(tag[0].length);
  }
  const re = /([.#[])([^\]=]+)(?:=["']?([^\]"']*)["']?)?\]/g;
  let m;
  while ((m = re.exec(rest)) || (m = identMatch(rest, m))) {
    if (m[1] === ".") {
      if (!el.classList.contains(m[2])) return false;
    } else if (m[1] === "#") {
      if (el.attrs.id !== m[2]) return false;
    } else if (!(m[2] in el.attrs) && !(m[2] in el.dataset)) {
      return false;
    } else if (m[3] !== undefined && m[3] !== "" ) {
      const v = m[2] in el.attrs ? el.attrs[m[2]] : el.dataset[m[2]];
      if (String(v) !== m[3]) return false;
    }
    break; // solo el primer predicador: basta para nuestros selectores
  }
  return true;
}
function identMatch() { return null; }

/* -------------------------------------------------------------------------- */
/* Canvas 2D                                                                  */
/* -------------------------------------------------------------------------- */

function makeCtx(canvas) {
  const state = { calls: 0, fillStyle: "#000", strokeStyle: "#000", lineWidth: 1, font: "", textAlign: "left",
    globalAlpha: 1, shadowBlur: 0, shadowColor: "", lineDashOffset: 0, imageSmoothingEnabled: true };
  const base = {
    canvas,
    ...state,
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {},
    createLinearGradient: () => ({ addColorStop: () => {} }),
    measureText: (t) => ({ width: String(t).length * 6 }),
  };
  return new Proxy(base, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === "symbol") return undefined;
      if (prop in state) return state[prop];
      // cualquier otro método (fillRect, arc, moveTo…) es un no-op contabilizado
      return (...args) => { target.calls++; void args; };
    },
    set(target, prop, value) {
      state[prop] = value;
      target[prop] = value;
      return true;
    },
  });
}

/* -------------------------------------------------------------------------- */
/* Parseo de HTML → árbol                                                     */
/* -------------------------------------------------------------------------- */

function parseHTML(html) {
  const root = new El("body");
  const stack = [root];
  const VOID = new Set(["INPUT", "IMG", "BR", "HR", "META", "LINK", "SOURCE", "AREA", "COL", "EMBED", "TRACK", "WB" + "EM"]);
  const src = html.replace(/<!--[\s\S]*?-->/g, "").replace(/<script[\s\S]*?<\/script>/gi, "").replace(/<style[\s\S]*?<\/style>/gi, "");
  const tagRe = /<(\/?)([a-zA-Z][\w-]*)((?:[^>"']|"[^"]*"|'[^']*')*)(\/?)>/g;
  const attrRe = /([:a-zA-Z_][\w.:-]*)((?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?)/g;
  let m;
  while ((m = tagRe.exec(src))) {
    const [, closing, name, attrTxt, selfClose] = m;
    const tag = name.toLowerCase();
    if (closing === "/") {
      for (let i = stack.length - 1; i > 0; i--) {
        if (stack[i].tagName === tag.toUpperCase()) { stack.length = i; break; }
      }
      continue;
    }
    const attrs = {};
    let a;
    attrRe.lastIndex = 0;
    while ((a = attrRe.exec(attrTxt))) {
      attrs[a[1]] = a[3] ?? a[4] ?? a[5] ?? "";
    }
    const el = new El(tag, attrs, stack[stack.length - 1]);
    stack[stack.length - 1].children.push(el);
    if (!VOID.has(el.tagName) && !selfClose) stack.push(el);
  }
  return root;
}

/* -------------------------------------------------------------------------- */
/* Entorno completo                                                           */
/* -------------------------------------------------------------------------- */

function createEnv(html) {
  const root = parseHTML(html);
  const byId = new Map();
  (function index(el) {
    if (el.attrs.id) byId.set(el.attrs.id, el);
    el.children.forEach(index);
  })(root);

  // tamaños plausibles para los tres lienzos
  for (const [id, w, h] of [["field", 900, 460], ["chart", 520, 128], ["net", 380, 128]]) {
    const el = byId.get(id);
    if (el) { el._w = w; el._h = h; }
  }

  const raf = [];
  const store = new Map();
  const listeners = new Map();
  const log = [];

  const document = {
    body: root,
    documentElement: root,
    fullscreenElement: null,
    getElementById: (id) => byId.get(id) || null,
    querySelector: (s) => root.querySelector(s),
    querySelectorAll: (s) => root.querySelectorAll(s),
    createElement: (tag) => new El(tag, {}),
    addEventListener: (t, f) => {
      if (!listeners.has(t)) listeners.set(t, []);
      listeners.get(t).push(f);
    },
    removeEventListener: () => {},
  };

  const window = {
    devicePixelRatio: 2,
    innerWidth: 1280,
    innerHeight: 800,
    addEventListener: (t, f) => {
      if (!listeners.has(t)) listeners.set(t, []);
      listeners.get(t).push(f);
    },
    removeEventListener: () => {},
    ResizeObserver: undefined,
    matchMedia: () => ({ matches: false, addEventListener: () => {} }),
  };

  const env = {
    root, byId, document, window, log,
    El,
    listeners,
    fire(type, ev = {}) {
      for (const fn of listeners.get(type) || []) fn(makeEvent(type, ev, root));
    },
    pump(frames = 1, t0 = 1000) {
      for (let i = 0; i < frames; i++) {
        const cbs = raf.splice(0, raf.length);
        for (const cb of cbs) cb(t0 + i * 16.7);
      }
    },
    requestAnimationFrame: (cb) => { raf.push(cb); return raf.length; },
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => { store.set(k, String(v)); log.push(`localStorage:${k}`); },
      removeItem: (k) => store.delete(k),
      _store: store,
    },
    Blob: BlobStub,
    URL: { createObjectURL: () => "blob:fake", revokeObjectURL: () => {} },
    navigator: { clipboard: { writeText: () => Promise.resolve() }, userAgent: "node-mini-dom" },
    location: { href: "http://localhost:8000/", hash: "", search: "" },
    history: { replaceState(s, t, url) { this.last = url; env.location.hash = String(url).replace(/^[^#]*/, ""); } },
    setTimeout: () => 0,
    clearTimeout: () => {},
    setInterval: () => 0,
    clearInterval: () => {},
    performance: { now: () => Date.now() },
    console,
    Math, URLSearchParams, TextEncoder, Intl, JSON, Date, Float64Array, Float32Array, Uint8ClampedArray, Set, Map, Promise, Number, String, Array, Object, Error, isNaN,
  };
  env.globalThis = env;
  env.self = env;
  env.addEventListener = window.addEventListener;
  env.removeEventListener = () => {};
  env.requestIdleCallback = undefined;
  return env;
}

function makeEvent(type, ev, target, current) {
  const e = {
    type,
    target: ev.target || target,
    currentTarget: current || target,
    preventDefault: () => {},
    stopPropagation: () => { ev._stopped = true; },
    shiftKey: false, ctrlKey: false, altKey: false, metaKey: false,
    button: 0, buttons: 1, pointerId: 1, isPrimary: true,
    clientX: 450, clientY: 230,
    ...ev,
  };
  return e;
}

module.exports = { createEnv, parseHTML, El, makeEvent };
