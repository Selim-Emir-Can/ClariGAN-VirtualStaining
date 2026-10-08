"""Build slides.html (static deck, no build step at view time) + slides/img/*.jpg.

  conda activate chatgarment
  python /local/emir/ClariDi/website_ClariDi/build_slides.py

The method figure is an SVG redraw of manuscript Fig. 7 (ClariDi_BME_Frontiers_v10_TRACKED.docx), filled with
real images of one held-out tile. Each variant slide reuses the same figure and highlights only what changes.
Reconstructions in panel A were made on CPU with tools/vqgan_recon.py (fine-tuned and stock checkpoints).
"""
import json, os
from PIL import Image

R = "/local/emir/ClariDi"
OUT = os.path.dirname(os.path.abspath(__file__))
IMG = f"{OUT}/slides/img"
TILE = "A_row7_col1_10x10"                        # specimen A, brain, held out in fold 0
REFS = ["A_row1_col7_10x10", "A_row1_col8_10x10", "A_row4_col7_10x10"]   # train bands for fold 0 (2, 3, 3)
NAVY, HL = "#2B3A8F", "#D55E00"
SERIF = "'Times New Roman', Times, 'Liberation Serif', serif"


# ------------------------------------------------------------------ images
def save(im, name, size=256, q=88):
    im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    im.save(f"{IMG}/{name}.jpg", "JPEG", quality=q, optimize=True)


def chan(im, c):
    """Single channel c of an RGB image, shown in its own colour (F1 red, F2 green) or grey (AF)."""
    bands = im.convert("RGB").split()
    z = Image.new("L", im.size, 0)
    if c == "grey":
        raise ValueError
    return Image.merge("RGB", [bands[0], z, z]) if c == "r" else Image.merge("RGB", [z, bands[1], z])


def build_images():
    os.makedirs(IMG, exist_ok=True)
    ua = Image.open(f"{OUT}/img/ua/{TILE}.jpg")
    csf = Image.open(f"{OUT}/img/csf/{TILE}.jpg")
    rec = {"ft": Image.open(f"{OUT}/tools/recon/recon_ft.png"), "stock": Image.open(f"{OUT}/tools/recon/recon_stock.png")}
    save(ua, "ua"); save(csf, "csf")
    for i, b in enumerate(ua.convert("RGB").split()):
        save(Image.merge("RGB", [b, b, b]), f"ua_AF{i+1}", 160)
    for nm, im in [("csf", csf), ("rec_ft", rec["ft"]), ("rec_stock", rec["stock"])] + \
                  [(f"vs_{s}", Image.open(f"{OUT}/img/vs/{s}/{TILE}_0.jpg")) for s in ("stock", "ours", "spec", "refA", "refB")]:
        save(im, nm)
        save(chan(im, "r"), f"{nm}_F1", 160); save(chan(im, "g"), f"{nm}_F2", 160)
    for i, t in enumerate(REFS):
        save(Image.open(f"{OUT}/img/csf/{t}.jpg"), f"refA_{i}", 128)
        save(Image.open(f"{OUT}/img/ua/{t}.jpg"), f"refB_{i}", 128)


# ------------------------------------------------------------------ svg helpers
def T(x, y, s, size=24, anchor="middle", weight="normal", fill="#111", style=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}" '
            f'font-family="{SERIF}" {style}>{s}</text>')


def I(x, y, w, h, name):
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="slides/img/{name}.jpg" preserveAspectRatio="none"/>'


def A(x1, y1, x2, y2, col="#111", w=2.2):
    m = "ah" if col == "#111" else "ahH"
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}" marker-end="url(#{m})"/>'


def box(x, y, w, h, fill, stroke="#444", rx=3, sw=1.4, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def bars(x0, cy, heights, w, gap, fill):
    out, x = [], x0
    for h in heights:
        out.append(box(x, cy - h / 2, w, h, fill, "#1a1a1a", 0, 1.2))
        x += w + gap
    return "".join(out)


def lock(cx, cy, s=1.0):
    return (f'<g transform="translate({cx},{cy}) scale({s})"><path d="M-9,-4 v-8 a9,9 0 0 1 18,0 v8" fill="none" stroke="#111" stroke-width="4"/>'
            f'<rect x="-14" y="-5" width="28" height="22" rx="3" fill="#111"/><circle cx="0" cy="5" r="3" fill="#ddd"/></g>')


def inset(x, y, names, labels, size, pad=8, gap=8, lsize=17):
    w = pad * 2 + size * len(names) + gap * (len(names) - 1)
    h = pad + size + 24
    out = [box(x, y, w, h, "none", NAVY, 0, 2.6, 'stroke-dasharray="7 5"')]
    for i, (n, l) in enumerate(zip(names, labels)):
        ix = x + pad + i * (size + gap)
        out.append(I(ix, y + pad, size, size, n))
        out.append(T(ix + size / 2, y + pad + size + 19, l, lsize))
    return "".join(out)


def hlbox(x, y, w, h):
    return box(x, y, w, h, "none", HL, 10, 2.6, 'stroke-dasharray="8 5"')


BLUE, BLUEBAR, GREEN, GREENBAR, GREY = "#CDD5F0", "#9FB0E2", "#D8EBD1", "#8DC47C", "#8E8E8E"


def figure(v):
    """v: ours | stock | spec | refA | refB"""
    stock = v == "stock"
    rec = "rec_stock" if stock else "rec_ft"
    vs = f"vs_{v}"
    o = ['<svg class="fig" viewBox="0 0 1000 985" xmlns="http://www.w3.org/2000/svg" role="img">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#111"/></marker>'
         f'<marker id="ahH" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{HL}"/></marker></defs>',
         '<rect x="0" y="0" width="1000" height="985" fill="#fff"/>']
    # ---------------- panel A
    o.append(box(70, 12, 915, 412, "none", "#111", 28, 2.6))
    o.append(T(24, 46, "A", 34, weight="bold"))
    o.append(T(0, 0, "Stock VQ-GAN (no fine-tuning)" if stock else "VAE Fine-tuning Strategy", 25,
               fill=HL if stock else "#111", style='transform="translate(52,218) rotate(-90)"'))
    o.append(T(175, 48, "Cleared &amp;", 24)); o.append(T(175, 76, "Stained Tissue", 24))
    o.append(I(100, 88, 150, 150, "csf"))
    o.append(inset(95, 282, ["csf_F1", "csf_F2"], ["F1", "F2"], 108))
    o.append(A(175, 282, 175, 244))
    o.append(T(548, 50, "ImageNet VQ-GAN, weights kept as is" if stock else "Stain Representation Network", 25,
               fill=HL if stock else "#111"))
    o.append(box(290, 70, 120, 180, BLUE)); o.append(bars(300, 160, [150, 112, 80, 50], 17, 10, BLUEBAR))
    o.append(A(252, 163, 287, 163))
    o.append(box(440, 138, 150, 46, "#fff", "#222", 14, 1.6)); o.append(T(515, 169, "Latent Space", 22))
    o.append(A(412, 161, 437, 161)); o.append(A(592, 161, 617, 161))
    o.append(box(620, 70, 120, 180, BLUE)); o.append(bars(632, 160, [50, 80, 112, 150], 17, 10, BLUEBAR))
    o.append(A(742, 163, 767, 163))
    o.append(T(845, 76, "Reconstruction", 24)); o.append(I(770, 88, 150, 150, rec))
    o.append(inset(735, 282, [f"{rec}_F1", f"{rec}_F2"], ["F1", "F2"], 108))
    o.append(A(845, 240, 845, 280))
    if stock:
        o.append(hlbox(282, 62, 466, 196))
        o.append(T(515, 278, "no fine-tuning on C&amp;SF", 19, fill=HL, style='font-style="italic"'))
    # ---------------- panel B
    o.append(box(70, 440, 915, 500, "none", "#111", 28, 2.6))
    o.append(T(24, 474, "B", 34, weight="bold"))
    o.append(T(0, 0, "Virtual Clearing &amp; Staining", 25, style='transform="translate(52,690) rotate(-90)"'))
    o.append(T(175, 478, "Uncleared", 24)); o.append(T(175, 506, "Tissue", 24))
    o.append(I(100, 518, 150, 150, "ua"))
    o.append(inset(95, 712, ["ua_AF1", "ua_AF2", "ua_AF3"], ["AF1", "AF2", "AF3"], 86))
    o.append(A(175, 712, 175, 672))
    o.append(T(535, 478, "Virtual Clearing &amp; Staining Network", 25))
    o.append(box(285, 500, 100, 180, BLUE)); o.append(bars(293, 590, [150, 112, 80, 50], 15, 8, GREY)); o.append(lock(337, 590, .9))
    o.append(A(252, 590, 282, 590))
    o.append(box(402, 500, 266, 180, GREEN))
    o.append(bars(421, 590, [150, 115, 82, 55, 55, 82, 115, 150], 18, 12, GREENBAR))
    o.append(A(387, 590, 399, 590)); o.append(A(670, 590, 682, 590))
    o.append(box(685, 500, 100, 180, BLUE)); o.append(bars(693, 590, [50, 80, 112, 150], 15, 8, GREY)); o.append(lock(733, 590, .9))
    o.append(A(787, 590, 802, 590))
    o.append(T(880, 478, "Virtually", 24)); o.append(T(880, 506, "Stained Image", 24))
    o.append(I(805, 518, 150, 150, vs))
    o.append(inset(735, 712, [f"{vs}_F1", f"{vs}_F2"], ["F1", "F2"], 96))
    o.append(A(880, 670, 880, 710))
    if stock:
        o.append(hlbox(279, 494, 112, 192)); o.append(hlbox(679, 494, 112, 192))
        o.append(T(335, 704, "ImageNet", 17, fill=HL, style='font-style="italic"'))
        o.append(T(735, 704, "ImageNet", 17, fill=HL, style='font-style="italic"'))
    # timestep + conditioning lane (x 395..730, y 690..935)
    o.append(f'<circle cx="452" cy="712" r="15" fill="#fff" stroke="#111" stroke-width="1.8"/>')
    o.append(T(452, 719, "t", 21, style='font-style="italic"'))
    if v in ("ours", "stock"):
        o.append(f'<path d="M467,712 H535 V684" fill="none" stroke="#111" stroke-width="2.2" marker-end="url(#ah)"/>')
        o.append(T(470, 748, "timestep", 17, anchor="start", fill="#444"))
    else:
        o.append(A(467, 712, 520, 712))
        o.append(f'<circle cx="535" cy="712" r="13" fill="#fff" stroke="{HL}" stroke-width="2.2"/>'
                 f'<path d="M528,712 h14 M535,705 v14" stroke="{HL}" stroke-width="2.2"/>')
        o.append(A(535, 699, 535, 684, HL))
        o.append(T(535, 757, "c", 22, fill=HL, style='font-style="italic"'))
        o.append(A(593, 712, 550, 712, HL))
        if v == "spec":
            o.append(box(595, 694, 122, 36, "#fff", HL, 6, 1.8)); o.append(T(656, 718, "embed W[s]", 18))
            o.append(box(595, 756, 122, 36, "#fff", "#444", 6, 1.4)); o.append(T(656, 780, "specimen s", 18))
            o.append(A(656, 756, 656, 732, HL))
            o.append(T(656, 815, "one of A &#8230; K", 15, fill="#444"))
            o.append(hlbox(489, 682, 244, 145))
            o.append(T(619, 852, "new: c = W[s]", 19, fill=HL, weight="bold"))
        else:
            ref, what = ("refA", "C&amp;SF tiles") if v == "refA" else ("refB", "UA tiles")
            o.append(box(595, 694, 122, 36, "#fff", HL, 6, 1.8)); o.append(T(656, 718, "MLP", 18))
            o.append(box(595, 750, 122, 44, "#fff", HL, 6, 1.8))
            o.append(T(656, 769, "[&#956;, &#963;] per ch.", 16)); o.append(T(656, 787, "mean over k", 14, fill="#444"))
            o.append(A(656, 750, 656, 732, HL))
            o.append(box(595, 812, 122, 34, BLUE, "#444", 3, 1.4)); o.append(T(643, 834, "VQ-GAN enc.", 15)); o.append(lock(703, 829, .5))
            o.append(A(656, 812, 656, 796, HL))
            for i in range(3):
                o.append(I(612 + 14 * i, 866 + 6 * i, 46, 46, f"{ref}_{i}"))
                o.append(box(612 + 14 * i, 866 + 6 * i, 46, 46, "none", "#fff", 0, 1))
            o.append(A(650, 866, 650, 848, HL))
            o.append(T(600, 882, what, 15, anchor="end")); o.append(T(600, 900, "same specimen,", 15, anchor="end"))
            o.append(T(600, 918, "train split", 15, anchor="end"))
            o.append(hlbox(489, 682, 244, 250))
    # ---------------- legend
    o.append(box(118, 950, 42, 26, BLUE, "#444", 0)); o.append(T(168, 971, "VQ-GAN (pretrained)", 22, anchor="start"))
    o.append(box(390, 950, 42, 26, GREEN, "#444", 0)); o.append(T(440, 971, "Latent Brownian Bridge Diffusion Model", 22, anchor="start"))
    if v != "ours":
        o.append(box(822, 951, 38, 24, "none", HL, 5, 2.4, 'stroke-dasharray="6 4"')); o.append(T(868, 971, "changed", 22, anchor="start", fill=HL))
    o.append("</svg>")
    svg = "".join(o)   # marker ids must be unique per figure (hidden slides cannot host shared defs)
    for a in ("ahH", "ah"):
        svg = svg.replace(f'url(#{a})', f'url(#{a}_{v})').replace(f'id="{a}"', f'id="{a}_{v}"')
    return svg


# ------------------------------------------------------------------ slides
SLIDES = [
    dict(id="ours", kicker="Method", title="Ours: stain-aware VQ-GAN + latent Brownian bridge",
         pts=["<b>(A) Stage 1.</b> The pretrained VQ-GAN is fine-tuned on C&amp;SF images, so its latent space captures stain-specific structure.",
              "<b>(B) Stage 2.</b> The VQ-GAN is frozen and a Latent Brownian Bridge Diffusion Model is trained from scratch to map UA latents to C&amp;SF latents.",
              "<b>Inference.</b> Encode UA, run 200 reverse bridge steps starting from <i>y</i> = E(UA), decode with the frozen decoder.",
              "<b>Objective</b> (same for every variant): <i>x<sub>t</sub></i> = (1&minus;<i>m<sub>t</sub></i>)<i>x</i><sub>0</sub> + <i>m<sub>t</sub> y</i> + &radic;<i>&delta;<sub>t</sub></i> &epsilon;; "
              "the UNet &epsilon;<sub>&theta;</sub>(<i>x<sub>t</sub></i>, <i>t</i>, <i>c</i>) regresses <i>m<sub>t</sub></i>(<i>y</i> &minus; <i>x</i><sub>0</sub>) + &radic;<i>&delta;<sub>t</sub></i> &epsilon;. Here <i>c</i> = 0."],
         need="the UA tile only.",
         foot="Shared by all five runs: 50 epochs &middot; batch 8 &times; 4 accumulation &middot; Adam, lr 10<sup>&minus;4</sup> &middot; T = 1000 &middot; EMA 0.995 &middot; same spatial folds."),
    dict(id="stock", kicker="Baseline", title="Vanilla L-BBDM: stock ImageNet VQ-GAN",
         pts=["<b>Stage 1 is skipped.</b> Encoder and decoder keep the ImageNet weights; nothing is tuned on tissue.",
              "<b>Stage 2 is unchanged.</b> Same L-BBDM, objective, schedule, sampler and folds; <i>c</i> = 0.",
              "Isolates what stain-aware latent tuning contributes.",
              "Panel A shows the stock VQ-GAN's own reconstruction of the C&amp;SF tile."],
         need="the UA tile only.", foot=""),
    dict(id="spec", kicker="Variant 1", title="+ specimen label (oracle): one learned vector per specimen",
         pts=["<i>c</i> = W[<i>s</i>]: a learned 512-d embedding per specimen (11 specimens + a null label), added to the UNet's timestep embedding.",
              "Learns a per-specimen bias, such as colour balance and stain intensity, from that specimen's training tiles.",
              "10% null-label dropout in training, as in classifier-free guidance training; sampled without guidance.",
              "Standard class conditioning (Dhariwal &amp; Nichol 2021). 6,144 extra parameters."],
         need="the specimen's identity, and that specimen must have been seen in training: an oracle-identity ablation that does not transfer to a new specimen.",
         foot=""),
    dict(id="refA", kicker="Variant 2 &middot; A", title="+ A (extra input): statistics of stained reference tiles",
         pts=["Each reference tile is encoded by the frozen VQ-GAN and summarised by its per-channel latent mean and std (512-d); these are averaged over <i>k</i> references and mapped by an MLP to <i>c</i>.",
              "References: C&amp;SF tiles of the same specimen from that fold's training split, never from the tile's own crop unit. <i>k</i> = 8 random in training, all of them at test.",
              "Last MLP layer zero-initialised (training starts as ours); learned null, 10% dropout; about 0.53M extra parameters.",
              "Pooled mean and std carry colour and intensity more than texture."],
         need="stained (C&amp;SF) tissue of the same specimen, so this is a few-shot, extra-input setting.", foot=""),
    dict(id="refB", kicker="Variant 3 &middot; B", title="+ B: the same, from unstained context tiles",
         pts=["Identical to A, except the statistics are computed on the <b>UA</b> inputs of the specimen's training tiles.",
              "Tells the model what this specimen looks like before staining, not what its stain should be.",
              "Same MLP, <i>k</i>, dropout and null; about 0.53M extra parameters.",
              "In these runs the bank holds the specimen's training tiles, so the comparison with A is like for like."],
         need="UA tiles of the same specimen, which always exist, so B is the form of specimen context that could be deployed on a new specimen.",
         foot=""),
]

TABLE = [
    ("Vanilla L-BBDM", "ImageNet (stock)", "0", "0", "UA tile"),
    ("Ours", "fine-tuned on C&amp;SF", "0", "0", "UA tile"),
    ("+ specimen label (oracle)", "fine-tuned on C&amp;SF", "W[<i>s</i>]", "6.1k", "+ specimen seen in training"),
    ("+ A stained refs (extra input)", "fine-tuned on C&amp;SF", "MLP of C&amp;SF tile statistics", "0.53M", "+ C&amp;SF tiles of the specimen"),
    ("+ B unstained ctx", "fine-tuned on C&amp;SF", "MLP of UA tile statistics", "0.53M", "+ UA tiles of the specimen"),
]


def method_slide(s, n, total):
    pts = "".join(f"<li>{p}</li>" for p in s["pts"])
    foot = f'<p class="foot">{s["foot"]}</p>' if s["foot"] else ""
    return f'''<section class="slide" id="s{n}" data-title="{s['id']}">
  <div class="figwrap">{figure(s["id"])}</div>
  <div class="txt">
    <div class="kicker">{s["kicker"]}</div>
    <h2>{s["title"]}</h2>
    <ul>{pts}</ul>
    <p class="need"><b>Needs at test time:</b> {s["need"]}</p>
    {foot}
  </div>
  <div class="pg">{n} / {total}</div>
</section>'''


def page():
    total = len(SLIDES) + 2
    cover = f'''<section class="slide cover" id="s1">
  <div class="cv">
    <div class="kicker">ClariDi &middot; model-design experiments</div>
    <h1>Four ways to condition the same latent bridge model</h1>
    <p class="sub">Ours, a stock-VQ-GAN baseline, and three conditioning variants (specimen label, A: stained references, B: unstained context), trained and tested on the same spatially blocked 5-fold split.</p>
    <div class="trio">
      <figure><img src="slides/img/ua.jpg" alt=""><figcaption>UA</figcaption></figure>
      <figure><img src="slides/img/csf.jpg" alt=""><figcaption>C&amp;SF</figcaption></figure>
      <figure><img src="slides/img/vs_ours.jpg" alt=""><figcaption>VS (ours)</figcaption></figure>
    </div>
    <p class="small">Images throughout: tile {TILE} (specimen A, brain), held out in fold 0; VS = draw 0. Reconstructions in panel A are real outputs of the fine-tuned and the stock VQ-GAN.</p>
  </div>
  <div class="pg">1 / {total}</div>
</section>'''
    body = [cover] + [method_slide(s, i + 2, total) for i, s in enumerate(SLIDES)]
    rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in TABLE)
    body.append(f'''<section class="slide" id="s{total}">
  <div class="cmp">
    <div class="kicker">Summary</div>
    <h2>What each variant adds, and what it needs at test time</h2>
    <table><tr><th>Model</th><th>VQ-GAN</th><th>conditioning <i>c</i></th><th>extra params</th><th>needs at test time</th></tr>{rows}</table>
    <p>Same split, schedule and sampler for all five; only the encoder (vanilla) or <i>c</i> (the three variants) changes.
      Ours and vanilla are plain image-to-image translation. The label is an oracle that does not exist for a new specimen.
      A needs stained tissue from the same specimen, so it must be presented as a few-shot or extra-input setting.
      B needs only uncleared tissue, which is always available.</p>
    <p class="links">Look at the outputs: <a href="tiles.html">tile viewer</a> &middot; <a href="wholesample.html">whole-sample viewer</a></p>
  </div>
  <div class="pg">{total} / {total}</div>
</section>''')
    return TEMPLATE.replace("__SLIDES__", "\n".join(body))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>ClariDi Model Variants</title>
<!-- generated by build_slides.py; edit that script, not this file -->
<style>
  :root { --navy: #2B3A8F; --hl: #D55E00; --ink: #111; --serif: "Times New Roman", Times, "Liberation Serif", serif; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #e7e7ea; }
  body { font-family: var(--serif); color: var(--ink); overflow: hidden; }
  #deck { position: absolute; left: 50%; top: 50%; width: 1280px; height: 720px; transform-origin: 0 0; }
  .slide { position: absolute; inset: 0; width: 1280px; height: 720px; background: #fff; display: none; box-shadow: 0 8px 40px rgba(0,0,0,.18); overflow: hidden; }
  .slide.on { display: block; }
  .figwrap { position: absolute; left: 14px; top: 10px; width: 710px; height: 700px; }
  .fig { width: 100%; height: 100%; display: block; }
  .txt { position: absolute; left: 746px; top: 34px; width: 506px; bottom: 40px; display: flex; flex-direction: column; }
  .kicker { font-family: var(--serif); font-variant: small-caps; letter-spacing: .06em; color: var(--navy); font-size: 21px; }
  h1 { font-size: 50px; line-height: 1.08; margin: 8px 0 14px; }
  h2 { font-size: 31px; line-height: 1.15; margin: 4px 0 14px; }
  ul { margin: 0; padding-left: 20px; font-size: 18.5px; line-height: 1.38; }
  li { margin: 0 0 9px; }
  .need { margin: 10px 0 0; font-size: 18.5px; line-height: 1.35; border: 2px dashed var(--navy); border-radius: 12px; padding: 9px 13px; }
  .foot { margin-top: auto; font-size: 15px; color: #444; }
  .pg { position: absolute; right: 18px; bottom: 10px; font-size: 14px; color: #777; }
  .cover .cv { position: absolute; left: 80px; top: 70px; right: 80px; }
  .cover .sub { font-size: 23px; line-height: 1.35; max-width: 1000px; color: #222; margin: 0 0 22px; }
  .trio { display: flex; gap: 14px; border: 2.5px solid #111; border-radius: 16px; padding: 12px 14px 6px; width: max-content; }
  .trio figure { margin: 0; text-align: center; }
  .trio img { width: 220px; height: 220px; display: block; }
  .trio figcaption { font-size: 24px; padding-top: 4px; }
  .small { font-size: 15px; color: #555; margin-top: 16px; }
  .cmp { position: absolute; left: 70px; top: 50px; right: 70px; }
  table { border-collapse: collapse; width: 100%; font-size: 20px; margin: 6px 0 22px; border: 2.5px solid #111; border-radius: 12px; }
  th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #ccc; }
  th { color: var(--navy); font-weight: 700; border-bottom: 2px solid #111; }
  .cmp p { font-size: 19.5px; line-height: 1.4; max-width: 1100px; }
  .cmp .links { font-size: 20px; }
  a { color: var(--navy); }
  #nav { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); display: flex; gap: 8px; align-items: center; font: 13px -apple-system, "Segoe UI", sans-serif; color: #444; background: rgba(255,255,255,.9); border: 1px solid #ccc; border-radius: 999px; padding: 3px 6px; z-index: 5; }
  #nav button, #nav a { font: inherit; border: 0; background: none; cursor: pointer; padding: 3px 9px; border-radius: 999px; color: #222; text-decoration: none; }
  #nav button:hover, #nav a:hover { background: #eee; }
  @media print {
    @page { size: 1280px 720px; margin: 0; }
    html, body { background: #fff; overflow: visible; height: auto; }
    #deck { position: static; transform: none !important; width: auto; height: auto; }
    .slide { display: block !important; position: relative; box-shadow: none; page-break-after: always; break-after: page; }
    #nav { display: none; }
  }
</style>
</head>
<body>
<div id="deck">
__SLIDES__
</div>
<div id="nav"><a href="index.html" title="back to the results page">&#8962;</a><button id="pv" title="previous (&larr;)">&larr;</button><span id="pn"></span><button id="nx" title="next (&rarr;, space)">&rarr;</button></div>
<script>
(function () {
  var S = [].slice.call(document.querySelectorAll(".slide")), deck = document.getElementById("deck"), i = 0;
  function fit() {
    var k = Math.min(window.innerWidth / 1280, (window.innerHeight - 44) / 720);
    deck.style.transform = "translate(" + (-640 * k) + "px," + (-360 * k - 16) + "px) scale(" + k + ")";
  }
  function go(n) {
    i = Math.max(0, Math.min(S.length - 1, n));
    S.forEach(function (s, j) { s.classList.toggle("on", j === i); });
    document.getElementById("pn").textContent = (i + 1) + " / " + S.length;
    try { history.replaceState(null, "", "#" + (i + 1)); } catch (e) {}
  }
  document.getElementById("pv").onclick = function () { go(i - 1); };
  document.getElementById("nx").onclick = function () { go(i + 1); };
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowRight" || e.key === "PageDown" || e.key === " ") { e.preventDefault(); go(i + 1); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); go(i - 1); }
    else if (e.key === "Home") go(0); else if (e.key === "End") go(S.length - 1);
  });
  window.addEventListener("resize", fit);
  fit(); go((parseInt(location.hash.slice(1), 10) || 1) - 1);
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build_images()
    open(f"{OUT}/slides.html", "w").write(page())
    print("wrote slides.html and", len(os.listdir(IMG)), "images in slides/img")
