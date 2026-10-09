/* ClariDi whole-sample viewer: client-side stitching of 256-px tiles onto canvases with shared pan/zoom.
   World coordinates = native frame pixels relative to the frame origin (shared by both grids). */
(function () {
  "use strict";
  const D = window.CLARIDI;
  const $ = s => document.querySelector(s);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const MB = Object.fromEntries(D.models.map(m => [m.slug, m]));
  const FOLD = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7"];
  const PANELS = [{ id: "ua", name: "UA", sub: "input" }, { id: "csf", name: "C&SF", sub: "reference" }]
    .concat(D.models.map(m => ({ id: m.slug, name: m.name, sub: "VS", vs: true })));
  const PREF = "claridi_wsi_prefs_v2";

  const S = { frame: D.frames[0].id, scale: "10x10", draw: 0, ngen: 5, panels: PANELS.map(p => p.id), grid: false, fold: false, mask: true };
  try { const p = JSON.parse(localStorage.getItem(PREF) || "{}"); Object.assign(S, p); } catch (e) {}
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("frame") && D.frames.some(f => f.id === h.get("frame"))) S.frame = h.get("frame");
  if (h.get("scale") === "5x5" || h.get("scale") === "10x10") S.scale = h.get("scale");
  let focusTile = h.get("tile") || null;
  const savePrefs = () => { try { localStorage.setItem(PREF, JSON.stringify({ scale: S.scale, draw: S.draw, ngen: S.ngen, panels: S.panels, grid: S.grid, fold: S.fold, mask: S.mask })); } catch (e) {} };

  // view transform (screen px of a panel viewport = world * k + t)
  const V = { k: 1, tx: 0, ty: 0, vw: 300, vh: 300 };
  let F = null, T = [], hover = null, pinned = null;

  // ---------------------------------------------------------------- image cache
  const cache = new Map();
  let pending = 0;
  function img(url) {
    let im = cache.get(url);
    if (!im) {
      im = new Image(); im.decoding = "async"; pending++;
      im.onload = im.onerror = () => { pending--; im._ok = im.naturalWidth > 0; dirty(); };
      im.src = url; cache.set(url, im);
    }
    return im;
  }
  const drawOf = (slug, k) => MB[slug].draws.includes(k) ? k : MB[slug].draws[0];
  function url(o, t) {
    const pid = o.p.id;
    if (pid === "ua") return `img/ua/${t.id}.jpg`;
    if (pid === "csf") return `img/csf/${t.id}.jpg`;
    return `img/vs/${pid}/${t.id}_${drawOf(pid, o.draw)}.jpg`;
  }

  // ---------------------------------------------------------------- controls
  const fsel = $("#frame");
  D.frames.forEach(f => { const o = document.createElement("option"); o.value = f.id; o.textContent = `${f.label} · ${f.tis}`; fsel.appendChild(o); });
  fsel.onchange = () => setFrame(fsel.value);
  const th = $("#thumbs");
  D.frames.forEach(f => {
    const b = document.createElement("button"); b.dataset.f = f.id; b.title = `${f.label} (${f.tis}), frame ${f.id}`;
    b.innerHTML = `<img src="img/thumb/${f.id}_10x10.jpg" alt=""><span>${esc(f.label)} · ${esc(f.tis)}</span>`;
    b.onclick = () => setFrame(f.id); th.appendChild(b);
  });
  $("#scale").onclick = e => { const b = e.target.closest("button"); if (b) { S.scale = b.dataset.v; changed(); } };
  const dseg = $("#draw");
  [0, 1, 2, 3, 4].forEach(k => { const b = document.createElement("button"); b.dataset.v = k; b.textContent = k; b.title = `draw ${k} (seed ${D.seeds[k]})`; dseg.appendChild(b); });
  dseg.onclick = e => { const b = e.target.closest("button"); if (b && S.ngen === 1) { S.draw = +b.dataset.v; buildPanels(); changed(); } };
  const gseg = $("#ngen");
  [1, 2, 3, 4, 5].forEach(k => { const b = document.createElement("button"); b.dataset.v = k; b.textContent = k; b.title = k === 1 ? "one draw per method (choose it with 'draw')" : `draws 0-${k - 1} of every method, one row per method`; gseg.appendChild(b); });
  gseg.onclick = e => { const b = e.target.closest("button"); if (b) { S.ngen = +b.dataset.v; buildPanels(); changed(); } };
  const pc = $("#pchips");
  PANELS.forEach(p => {
    const l = document.createElement("label"); l.className = "chip"; l.dataset.p = p.id;
    l.innerHTML = `<input type="checkbox">${esc(p.vs ? MB[p.id].short : p.name)}`;
    l.querySelector("input").onchange = e => {
      S.panels = PANELS.map(x => x.id).filter(id => id === p.id ? e.target.checked : S.panels.includes(id));
      buildPanels(); changed();
    };
    pc.appendChild(l);
  });
  [["oGrid", "grid"], ["oFold", "fold"], ["oMask", "mask"]].forEach(([id, key]) => $("#" + id).onchange = e => { S[key] = e.target.checked; changed(); });
  $("#zin").onclick = () => zoomAt(V.vw / 2, V.vh / 2, 1.5);
  $("#zout").onclick = () => zoomAt(V.vw / 2, V.vh / 2, 1 / 1.5);
  $("#fit").onclick = () => { fit(); dirty(); };

  function syncControls() {
    fsel.value = S.frame;
    th.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.f === S.frame));
    $("#scale").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === S.scale));
    dseg.querySelectorAll("button").forEach(b => { b.classList.toggle("on", S.ngen === 1 && +b.dataset.v === S.draw); b.disabled = S.ngen > 1; });
    gseg.querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.v === S.ngen));
    pc.querySelectorAll("label").forEach(l => { const on = S.panels.includes(l.dataset.p); l.classList.toggle("on", on); l.querySelector("input").checked = on; });
    [["oGrid", "cGrid", "grid"], ["oFold", "cFold", "fold"], ["oMask", "cMask", "mask"]].forEach(([i, c, k]) => { $("#" + i).checked = S[k]; $("#" + c).classList.toggle("on", S[k]); });
    const sel = th.querySelector("button.on"); if (sel && sel.scrollIntoView) sel.scrollIntoView({ block: "nearest", inline: "nearest" });
    legend();
  }
  function legend() {
    const nT = T.length, nM = T.filter(t => t.m).length;
    $("#legend").innerHTML =
      `<span><b>${esc(F.label)}</b> (${esc(F.tis)}), ${S.scale} grid: ${nT} tiles${nM ? `, ${nM} masked` : ""}</span>` +
      `<span>held out in fold: ${FOLD.map((c, i) => `<span class="fold-sw" style="background:${c}"></span> ${i}`).join(" &nbsp;")}</span>` +
      `<span><span class="hatch"></span> no tile</span><span><span class="msk"></span> masked crop</span>`;
  }

  // ---------------------------------------------------------------- panels + layout
  const box = $("#panels");
  let P = [];
  function mkPanel(parent, p, draw, title, cls) {
    const el = document.createElement("div"); el.className = "panel " + cls;
    el.innerHTML = `<div class="ph"><span class="t"></span><span class="s"></span></div><div class="vp"><canvas></canvas><span class="load"></span></div>`;
    parent.appendChild(el);
    const o = { p, draw, el, vp: el.querySelector(".vp"), cv: el.querySelector("canvas"), sub: el.querySelector(".s"), t: el.querySelector(".t"), load: el.querySelector(".load") };
    o.t.textContent = title;
    attach(o); P.push(o);
  }
  function buildPanels() {
    box.innerHTML = ""; P = [];
    const sel = PANELS.filter(p => S.panels.includes(p.id));
    box.classList.toggle("rows", S.ngen > 1);
    if (S.ngen === 1) {
      sel.forEach(p => mkPanel(box, p, S.draw, p.vs ? `VS · ${MB[p.id].name}` : p.name, "frame" + (p.vs ? " vsp" : "")));
    } else {
      // one row per method: [UA | C&SF] then that method's draws, so every generation sits next to its input and reference
      const refs = sel.filter(p => !p.vs), vs = sel.filter(p => p.vs);
      const addRefs = cells => { if (!refs.length) return; const r = document.createElement("div"); r.className = "rrefs"; cells.appendChild(r); refs.forEach(p => mkPanel(r, p, 0, p.name, "bare")); };
      if (!vs.length && refs.length) {
        const g = document.createElement("div"); g.className = "wrow frame";
        g.innerHTML = `<div class="rh">References</div><div class="cells"></div>`; box.appendChild(g); addRefs(g.querySelector(".cells"));
      }
      vs.forEach(p => {
        const g = document.createElement("div"); g.className = "wrow vsg";
        g.innerHTML = `<div class="rh">VS · ${esc(MB[p.id].name)}</div><div class="cells"></div>`; box.appendChild(g);
        const cells = g.querySelector(".cells"); addRefs(cells);
        MB[p.id].draws.slice(0, S.ngen).forEach(k => mkPanel(cells, p, k, MB[p.id].draws.length > 1 ? `draw ${k}` : "single output", "bare"));
      });
    }
    if (!P.length) box.innerHTML = `<p class="mute" style="padding:30px">Select at least one panel above.</p>`;
    layout(true);
  }
  function layout(refit) {
    const n = P.length; if (!n || !F) return;
    const top = box.getBoundingClientRect().top + window.scrollY;
    const availW = document.documentElement.clientWidth - 32, availH = Math.max(260, window.innerHeight - (top - window.scrollY) - 14);
    const gap = 10, head = 30, pad = 16, ar = F.W / F.H;
    let best = null;
    if (S.ngen > 1) {
      // one row per method: ngen panels side by side; cap the height so about two rows fit on screen
      const nref = S.panels.filter(id => !MB[id]).length, nvs = S.panels.filter(id => MB[id]).length;
      const c = window.innerWidth < 640 ? 2 : (nvs ? S.ngen : 0) + nref, gpad = 24 + (nref && nvs ? 14 : 0), cgap = 8;
      let w = (availW - gpad - (c - 1) * cgap) / c - 4;
      const hMax = Math.max(160, (window.innerHeight - 40) * 0.40);
      if (w / ar > hMax) w = hMax * ar;
      best = { c, w: Math.floor(w), h: Math.floor(w / ar) };
      box.style.gridTemplateColumns = "";
    } else {
      for (let c = 1; c <= n; c++) {
        const r = Math.ceil(n / c);
        const cw = (availW - (c - 1) * gap) / c - pad, ch = (availH - (r - 1) * gap) / r - pad - head;
        let w = Math.min(cw, ch * ar), hh = w / ar;
        if (w < 120) { w = Math.min(cw, 120); hh = w / ar; }
        const score = w * hh * n - (c * r - n) * 1e-3;
        if (!best || score > best.score) best = { c, w: Math.floor(w), h: Math.floor(hh), score };
      }
      if (window.innerWidth < 640) { best.c = 1; best.w = Math.floor(availW - pad); best.h = Math.floor(Math.min(best.w / ar, window.innerHeight * 0.6)); }
      box.style.gridTemplateColumns = `repeat(${best.c}, ${best.w + pad}px)`;
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const oldW = V.vw, oldH = V.vh;
    P.forEach(o => {
      o.vp.style.width = best.w + "px"; o.vp.style.height = best.h + "px";
      o.cv.width = Math.round(best.w * dpr); o.cv.height = Math.round(best.h * dpr); o.dpr = dpr;
    });
    V.vw = best.w; V.vh = best.h;
    if (refit) fit();
    else { // keep the same world point at the centre
      const cx = (oldW / 2 - V.tx) / V.k, cy = (oldH / 2 - V.ty) / V.k, s = Math.min(V.vw / oldW, V.vh / oldH);
      V.k *= s; V.tx = V.vw / 2 - cx * V.k; V.ty = V.vh / 2 - cy * V.k;
    }
    dirty();
  }
  function fit() {
    V.k = Math.min(V.vw / F.W, V.vh / F.H) * 0.98;
    V.tx = (V.vw - F.W * V.k) / 2; V.ty = (V.vh - F.H * V.k) / 2;
  }
  function zoomAt(mx, my, f) {
    const k0 = Math.min(V.vw / F.W, V.vh / F.H) * 0.98;
    const nk = Math.max(k0 * 0.5, Math.min(V.k * f, 4 * 256 / tileW()));
    V.tx = mx - (mx - V.tx) * nk / V.k; V.ty = my - (my - V.ty) * nk / V.k; V.k = nk;
    dirty();
  }
  const tileW = () => T.length ? T[0].w : 1000;
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => layout(false), 120); });

  // ---------------------------------------------------------------- interaction
  function worldAt(o, e) {
    const r = o.cv.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    return { sx, sy, x: (sx - V.tx) / V.k, y: (sy - V.ty) / V.k };
  }
  const tileAt = (x, y) => T.find(t => x >= t.lx && x < t.lx + t.w && y >= t.ly && y < t.ly + t.h) || null;
  function attach(o) {
    let drag = null;
    const ptrs = new Map();
    o.vp.addEventListener("wheel", e => {
      e.preventDefault();
      const w = worldAt(o, e), dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      zoomAt(w.sx, w.sy, Math.exp(-dy * 0.0015));
    }, { passive: false });
    o.vp.addEventListener("pointerdown", e => {
      o.vp.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      drag = { x: e.clientX, y: e.clientY, tx: V.tx, ty: V.ty, moved: false };
      if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; drag.pinch = Math.hypot(a.x - b.x, a.y - b.y); }
      o.vp.classList.add("drag");
    });
    o.vp.addEventListener("pointermove", e => {
      if (ptrs.has(e.pointerId)) ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (drag && ptrs.size === 2 && drag.pinch) {
        const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y), r = o.cv.getBoundingClientRect();
        zoomAt((a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top, d / drag.pinch); drag.pinch = d; drag.moved = true; return;
      }
      if (drag && ptrs.size === 1) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        if (drag.moved) { V.tx = drag.tx + dx; V.ty = drag.ty + dy; dirty(); }
      }
      const w = worldAt(o, e), t = tileAt(w.x, w.y);
      if (t !== hover) { hover = t; dirty(); }
      tip(e, o, t);
    });
    const end = e => {
      ptrs.delete(e.pointerId);
      if (drag && !drag.moved && e.type === "pointerup") { const w = worldAt(o, e); pin(tileAt(w.x, w.y)); }
      if (!ptrs.size) { drag = null; o.vp.classList.remove("drag"); }
    };
    o.vp.addEventListener("pointerup", end); o.vp.addEventListener("pointercancel", end);
    o.vp.addEventListener("pointerleave", () => { if (!drag) { hover = null; $("#tip").style.display = "none"; dirty(); } });
    o.vp.addEventListener("dblclick", e => { const w = worldAt(o, e); zoomAt(w.sx, w.sy, 2); });
  }
  function lpOf(pid, t, k) { const a = t.lp && t.lp[pid]; return a ? a[drawOf(pid, k)] : null; }
  function lpMean(pid, t) { const a = t.lp && t.lp[pid]; if (!a) return null; const b = a.slice(0, S.ngen > 1 ? S.ngen : 5); return S.ngen > 1 ? b.reduce((x, y) => x + y, 0) / b.length : a[drawOf(pid, S.draw)]; }
  function tip(e, o, t) {
    const el = $("#tip");
    if (!t) { el.style.display = "none"; return; }
    const lp = o.p.vs ? lpOf(o.p.id, t, o.draw) : null;
    el.innerHTML = `<b>${esc(t.id)}</b><br><span class="fold-sw" style="background:${FOLD[t.fold]}"></span> held out in fold ${t.fold} · row ${t.row}, col ${t.col}` +
      `${t.m ? ' · <span class="badge masked">masked</span>' : ""}${lp != null ? `<br><span class="mute">LPIPS (context) ${lp.toFixed(3)}</span>` : ""}<br><span class="mute">click to pin</span>`;
    el.style.display = "block";
    const x = Math.min(e.clientX + 14, window.innerWidth - el.offsetWidth - 8), y = Math.min(e.clientY + 14, window.innerHeight - el.offsetHeight - 8);
    el.style.left = x + "px"; el.style.top = y + "px";
  }
  function pin(t) {
    pinned = t; dirty();
    const el = $("#pin");
    if (!t) { el.style.display = "none"; return; }
    const vs = S.panels.filter(id => MB[id]).map(id => { const l = lpMean(id, t); return l != null ? `${esc(MB[id].short)} ${l.toFixed(3)}` : ""; }).filter(Boolean).join(" · ");
    el.innerHTML = `Pinned <b style="font-family:var(--mono)">${esc(t.id)}</b> · ${esc(t.sc)} · <span class="fold-sw" style="background:${FOLD[t.fold]}"></span> fold ${t.fold}` +
      `${t.m ? ' · <span class="badge masked">masked</span>' : ""} · <a href="tiles.html#tile=${encodeURIComponent(t.id)}">open in tile viewer &rarr;</a>` +
      `${vs ? `<br><span class="mute small">LPIPS (context only, ${S.ngen > 1 ? `mean of draws 0-${S.ngen - 1}` : `draw ${S.draw}`}): ${vs}</span>` : ""} <button id="unpin" style="margin-left:8px">unpin</button>`;
    el.style.display = "block";
    $("#unpin").onclick = () => pin(null);
  }

  document.addEventListener("keydown", e => {
    if ((e.target.matches && e.target.matches("input[type=text], select, textarea")) || e.ctrlKey || e.metaKey || e.altKey) return;
    const i = D.frames.findIndex(f => f.id === S.frame);
    if (e.key === "n" && i < D.frames.length - 1) setFrame(D.frames[i + 1].id);
    else if (e.key === "N" && i > 0) setFrame(D.frames[i - 1].id);
    else if (e.key === "s") { S.scale = S.scale === "10x10" ? "5x5" : "10x10"; changed(); }
    else if (e.key === "d" && S.ngen === 1) { S.draw = (S.draw + 1) % 5; buildPanels(); changed(); }
    else if (e.key === "g") { S.ngen = S.ngen === 5 ? 1 : 5; buildPanels(); changed(); }
    else if (e.key === "0") { fit(); dirty(); }
    else if (e.key === "+" || e.key === "=") zoomAt(V.vw / 2, V.vh / 2, 1.5);
    else if (e.key === "-") zoomAt(V.vw / 2, V.vh / 2, 1 / 1.5);
    else if (e.key === "Escape") pin(null);
  });

  // ---------------------------------------------------------------- state changes
  function setFrame(id) { S.frame = id; focusTile = null; pin(null); changed(true); }
  function changed(refit) {
    F = D.frames.find(f => f.id === S.frame);
    T = D.tiles.filter(t => t.fr === S.frame && t.sc === S.scale).map(t => Object.assign(t, { lx: t.x - F.x0, ly: t.y - F.y0 }));
    savePrefs(); syncControls();
    try { history.replaceState(null, "", `#frame=${S.frame}&scale=${S.scale}`); } catch (e) {}
    if (refit) layout(true); else dirty();
    if (pinned && pinned.sc !== S.scale) pin(null); else if (pinned) pin(pinned);
  }

  // ---------------------------------------------------------------- drawing
  let raf = 0;
  function dirty() { if (!raf) raf = requestAnimationFrame(render); }
  let hatch = null;
  function hatchPattern(ctx) {
    if (hatch) return hatch;
    const c = document.createElement("canvas"); c.width = c.height = 8;
    const g = c.getContext("2d"); g.fillStyle = "#1c1c1f"; g.fillRect(0, 0, 8, 8);
    g.strokeStyle = "#55555b"; g.lineWidth = 1.6; g.beginPath(); g.moveTo(-2, 10); g.lineTo(10, -2); g.moveTo(6, 10); g.lineTo(10, 6); g.moveTo(-2, 2); g.lineTo(2, -2); g.stroke();
    hatch = ctx.createPattern(c, "repeat"); return hatch;
  }
  function render() {
    raf = 0;
    if (!F) return;
    const sx = x => x * V.k + V.tx, sy = y => y * V.k + V.ty;
    P.forEach(o => {
      const ctx = o.cv.getContext("2d"), d = o.dpr;
      ctx.setTransform(d, 0, 0, d, 0, 0);
      ctx.fillStyle = "#1c1c1f"; ctx.fillRect(0, 0, V.vw, V.vh);
      ctx.fillStyle = hatchPattern(ctx);
      ctx.fillRect(sx(0), sy(0), F.W * V.k, F.H * V.k);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      let missing = 0;
      for (const t of T) {
        const x0 = Math.floor(sx(t.lx) * d) / d, y0 = Math.floor(sy(t.ly) * d) / d;
        const x1 = Math.ceil(sx(t.lx + t.w) * d) / d, y1 = Math.ceil(sy(t.ly + t.h) * d) / d;
        if (x1 < 0 || y1 < 0 || x0 > V.vw || y0 > V.vh) continue;
        const im = img(url(o, t));
        if (im._ok) ctx.drawImage(im, x0, y0, x1 - x0, y1 - y0);
        else { missing++; ctx.fillStyle = "#2a2a2e"; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); }
      }
      o.load.textContent = missing ? `loading ${missing}…` : "";
      // overlays
      ctx.lineWidth = 1;
      if (S.grid) {
        ctx.strokeStyle = "rgba(255,255,255,.45)";
        for (const t of T) ctx.strokeRect(Math.round(sx(t.lx)) + .5, Math.round(sy(t.ly)) + .5, Math.round(t.w * V.k) - 1, Math.round(t.h * V.k) - 1);
      }
      if (S.fold) {
        const lw = Math.max(2, Math.min(4, t0w() * V.k / 40));
        ctx.lineWidth = lw;
        for (const t of T) { ctx.strokeStyle = FOLD[t.fold]; ctx.strokeRect(sx(t.lx) + lw / 2 + .5, sy(t.ly) + lw / 2 + .5, t.w * V.k - lw - 1, t.h * V.k - lw - 1); }
      }
      if (S.mask) {
        ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.strokeStyle = "#ffb000";
        for (const t of T) if (t.m) ctx.strokeRect(sx(t.lx) + 2, sy(t.ly) + 2, t.w * V.k - 4, t.h * V.k - 4);
        ctx.setLineDash([]);
        const fs = Math.max(9, Math.min(13, t0w() * V.k / 6));
        if (t0w() * V.k > 28) {
          ctx.font = `600 ${fs}px sans-serif`; ctx.textBaseline = "top";
          for (const t of T) if (t.m) { const x = sx(t.lx) + 5, y = sy(t.ly) + 5; ctx.fillStyle = "#ffb000"; ctx.fillRect(x, y, fs * 1.15, fs * 1.25); ctx.fillStyle = "#000"; ctx.fillText("M", x + fs * .1, y + fs * .15); }
        }
      }
      const hl = (t, col, w) => { if (!t || t.sc !== S.scale || t.fr !== S.frame) return; ctx.lineWidth = w; ctx.strokeStyle = col; ctx.strokeRect(sx(t.lx), sy(t.ly), t.w * V.k, t.h * V.k); };
      const ft = focusTile && T.find(t => t.id === focusTile);
      if (ft) hl(ft, "#ff3b30", 3);
      hl(pinned, "#ffffff", 3); hl(pinned, "#2B3A8F", 1.2);
      if (hover && hover !== pinned) hl(hover, "rgba(255,255,255,.9)", 1.5);
      o.sub.textContent = o.p.vs ? (S.ngen > 1 ? `seed ${D.seeds[o.draw]}` : `draw ${o.draw}`) : o.p.sub;
    });
  }
  const t0w = () => T.length ? T[0].w : 1000;

  // ---------------------------------------------------------------- start
  F = D.frames.find(f => f.id === S.frame);
  buildPanels();
  changed(true);
  if (focusTile) {
    const t = T.find(x => x.id === focusTile);
    if (t) { pin(t); V.k = Math.min(V.vw / F.W, V.vh / F.H) * 0.98 * 2.2; V.tx = V.vw / 2 - (t.lx + t.w / 2) * V.k; V.ty = V.vh / 2 - (t.ly + t.h / 2) * V.k; dirty(); }
  }
})();
