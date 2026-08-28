from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from clipart_icons import make_icon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"
PAPER_FONT = "Times New Roman"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [PAPER_FONT, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "axes.unicode_minus": False,
    }
)


def rounded_box(ax, xy, width, height, edge, face, lw=2.2, radius=0.012, zorder=2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def label_box(ax, x, y, w, h, text, edge, face, fontsize=8.0, weight="bold", align="center"):
    rounded_box(ax, (x, y), w, h, edge=edge, face=face, lw=2.0, radius=0.006, zorder=5)
    ax.text(
        x + (w / 2 if align == "center" else 0.012),
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
        zorder=6,
    )


def _dist_pts(ax, a, b):
    """Distance between two data-space points, measured in typographic points."""
    (x0, y0), (x1, y1) = ax.transData.transform([a, b])
    return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) * 72.0 / ax.figure.dpi


def _trim_toward(ax, p_from, p_to, back_pts):
    """Move p_to back toward p_from by back_pts typographic points."""
    if back_pts <= 0:
        return p_to
    d = _dist_pts(ax, p_from, p_to)
    if d <= back_pts:
        return p_to
    t = back_pts / d
    return (p_to[0] + (p_from[0] - p_to[0]) * t, p_to[1] + (p_from[1] - p_to[1]) * t)


def _halo(lw):
    # A single stroked outline keeps dash phase and arrowhead geometry identical
    # to the colored pass, so dashed arrows get a clean thin white rim instead of
    # a mismatched solid white underlay blob.
    return [patheffects.withStroke(linewidth=lw + 3.0, foreground="white")]


def _fit_short_gap(ax, start, end, shrinkA, shrinkB, ms):
    """On short hops, cap shrink and head size so a visible shaft always remains."""
    gap = _dist_pts(ax, start, end)
    if shrinkA + shrinkB + 0.4 * ms > 0.62 * gap:
        shrinkA = min(shrinkA, 5.0)
        shrinkB = min(shrinkB, 5.0)
        inner = max(gap - shrinkA - shrinkB, 4.0)
        ms = min(ms, inner * 1.5)  # head length (0.4*ms) at most ~60% of remaining span
    return shrinkA, shrinkB, ms


def strong_arrow(ax, start, end, color="#111827", rad=0.0, zorder=9, scale=1.0, shrink=9):
    lw = 3.4 * scale
    shrinkA, shrinkB, ms = _fit_short_gap(ax, start, end, shrink, shrink, 20 * scale)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder + 1,
            joinstyle="round",
            capstyle="butt",
            path_effects=_halo(lw),
        )
    )


# Emoji clipart icons (Segoe UI Emoji, full color).  Signature: fn(ax, x, y, size, color).
draw_doc_icon = make_icon("\U0001F4DD")             # memo: text / field notes
draw_image_icon = make_icon("\U0001F5BC️")     # framed picture: generic image input
draw_fused_icon = make_icon("\U0001F9E9")           # puzzle piece: VLM fusion / fused representation
draw_gate_icon = make_icon("\U0001F39A️")      # level slider: learned gate
draw_transformer_icon = make_icon("⚙️")     # gear: shared transformer encoder


def input_box(ax, x, y, w, h, text, kind, color):
    icon_size = min(w * 0.82, h * 0.72)
    ix = x + (w - icon_size) / 2
    iy = y + h * 0.02
    if kind == "doc":
        draw_doc_icon(ax, ix, iy, icon_size, color)
    elif kind == "image":
        draw_image_icon(ax, ix, iy, icon_size, color)
    elif kind == "gate":
        draw_gate_icon(ax, ix, iy, icon_size, color)
    else:
        draw_transformer_icon(ax, ix, iy, icon_size, color)
    ax.text(x + w * 0.50, y + h * 0.86, text, ha="center", va="center", fontsize=7.4, fontweight="bold", color="#111827", zorder=8)


def fused_box(ax, x, y, w, h):
    rounded_box(ax, (x, y), w, h, edge="#ef4444", face="#fff7ed", lw=2.0, radius=0.006, zorder=5)
    draw_fused_icon(ax, x + w * 0.07, y + h * 0.09, h * 0.82)
    ax.text(x + w * 0.64, y + h / 2, "Fused hf", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#111827", zorder=8)


def draw_card(ax, x, y, w, h, idx, title, dim, operator, note, theme, extras=None):
    edge, face = theme
    rounded_box(ax, (x, y), w, h, edge=edge, face=face, lw=2.4, radius=0.012, zorder=1)
    badge_w = w * 0.075
    rounded_box(ax, (x + w * 0.025, y + h * 0.875), badge_w, h * 0.075, edge=edge, face=edge, lw=1.8, radius=0.006, zorder=5)
    ax.text(x + w * 0.025 + badge_w / 2, y + h * 0.913, str(idx), ha="center", va="center", fontsize=8.8, fontweight="bold", color="white", zorder=6)
    ax.text(x + w * 0.125, y + h * 0.905, title, ha="left", va="center", fontsize=10.0, fontweight="bold", color="#111827", zorder=6)

    tx = x + w * 0.045
    ix = x + w * 0.675
    top_y = y + h * 0.690
    in_w = w * 0.300
    in_h = h * 0.180
    op_w = w * 0.310
    op_h = h * 0.120
    op_x = x + w * 0.345
    op_y = y + h * 0.445
    fused_w = w * 0.330
    fused_h = h * 0.105
    fused_x = x + w * 0.335
    fused_y = y + h * 0.060

    input_box(ax, tx, top_y, in_w, in_h, "Text", "doc", "#2563eb")
    input_box(ax, ix, top_y, in_w, in_h, "Image", "image", "#16a34a")

    if extras == "flamingo":
        op_x = x + w * 0.255
        op_w = w * 0.490
        op_h = h * 0.135
        operator = "Gated XAttn\n+ Perceiver"
    elif extras == "blip":
        op_x = x + w * 0.255
        op_w = w * 0.490
        op_h = h * 0.135
        operator = "[ht ; q]\nQ-Former"
    elif extras == "coca":
        op_x = x + w * 0.255
        op_w = w * 0.490
        op_h = h * 0.135
        operator = "Contrast + Caption\nDual [...]"
    label_box(ax, op_x, op_y, op_w, op_h, operator, edge=edge, face="#ffffff", fontsize=7.1)

    strong_arrow(ax, (tx + in_w * 0.50, top_y), (op_x + op_w * 0.20, op_y + op_h), color="#111827", shrink=8)
    strong_arrow(ax, (ix + in_w * 0.50, top_y), (op_x + op_w * 0.80, op_y + op_h), color="#111827", shrink=8)

    strong_arrow(ax, (op_x + op_w / 2, op_y), (fused_x + fused_w / 2, fused_y + fused_h), color=edge, shrink=10)
    fused_box(ax, fused_x, fused_y, fused_w, fused_h)


def make_vlm_fusion_clipart():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 9.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    rounded_box(ax, (0.026, 0.045), 0.948, 0.910, edge="#2563eb", face="#f8fbff", lw=2.8, radius=0.016, zorder=0)
    ax.text(0.5, 0.915, "VLM Fusion Strategy Details", ha="center", va="center", fontsize=17.5, fontweight="bold", color="#2563eb")

    themes = [
        ("#2563eb", "#eaf4ff"),
        ("#7c3aed", "#f3efff"),
        ("#16a34a", "#ecfdf0"),
        ("#0891b2", "#e8fbff"),
        ("#f97316", "#fff2dc"),
        ("#d99a00", "#fff8d8"),
        ("#ef4444", "#fff0ed"),
        ("#64748b", "#f2f5f9"),
    ]
    cards = [
        ("Concatenation", "768-d output", "[ht ; hv]", "Direct feature stack: simplest baseline", None),
        ("Cross-Attention", "256-d output", "MHA(Q,K,V)", "Text queries visual keys and values", None),
        ("Gated Fusion", "256-d output", "g = sigmoid(W[.])", "Learned gate balances text/image evidence", None),
        ("CLIP-Style", "512-d output", "Norm + Project", "Normalized projections before fusion", None),
        ("Flamingo", "256-d output", "Gated XAttn", "Compressed visual tokens guide text", "flamingo"),
        ("BLIP-2", "256-d output", "[ht ; q]", "Visual query vector joins with text", "blip"),
        ("CoCa", "768-d output", "Dual [...]", "Contrastive and captioning cues combine", "coca"),
        ("Unified-IO", "256-d output", "Transformer", "Modality embeddings feed shared encoder", "unified"),
    ]

    left = 0.050
    top = 0.505
    card_w = 0.212
    card_h = 0.330
    gap_x = 0.018
    gap_y = 0.047
    for i, (title, dim, op, note, extra) in enumerate(cards):
        row = i // 4
        col = i % 4
        x = left + col * (card_w + gap_x)
        y = top - row * (card_h + gap_y)
        draw_card(ax, x, y, card_w, card_h, i + 1, title, dim, op, note, themes[i], extra)

    fig.savefig(OUT / "farmfederate_vlm_fusion_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arch_arrow(ax, start, end, color="#1f2937", scale=1.0, dashed=False, rad=0.0, zorder=6, shrink=8):
    lw = 3.8 * scale
    shrinkA, shrinkB, ms = _fit_short_gap(ax, start, end, shrink, shrink, 23 * scale)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            linestyle=(0, (5, 5)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder + 1,
            joinstyle="round",
            capstyle="butt",
            path_effects=_halo(lw),
        )
    )


def routed_arch_arrow(ax, points, color="#1f2937", scale=1.0, dashed=True, zorder=6, shrink=0, shrink_end=None):
    """Draw a multi-segment routed arrow as one continuous path with a clear head."""
    if len(points) < 2:
        return
    pts = list(points)
    shrink_end = shrink if shrink_end is None else shrink_end
    pts[0] = _trim_toward(ax, pts[1], pts[0], shrink)
    pts[-1] = _trim_toward(ax, pts[-2], pts[-1], shrink_end)
    lw = 3.8 * scale
    # Keep the head short enough to fit inside the final segment.
    last = _dist_pts(ax, pts[-2], pts[-1])
    ms = min(23 * scale, max(2.0 * last, 7.0))
    path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=(0, (5, 5)) if dashed else "solid",
            zorder=zorder + 1,
            joinstyle="round",
            capstyle="butt",
            path_effects=_halo(lw),
        )
    )


# Emoji clipart icons for the architecture / edge-cloud / RAG diagrams.
draw_network_icon = make_icon("\U0001F310")          # globe with meridians: FedAvg network
draw_bar_icon = make_icon("\U0001F4CA")              # bar chart: classifier / softmax scores
draw_check_icon = make_icon("✅")                # check mark button: verified output
draw_search_icon = make_icon("\U0001F50D")           # magnifying glass: Sentence-BERT retrieval
draw_db_icon = make_icon("\U0001F5C4️")         # file cabinet: FAISS knowledge base
draw_sensor_icon = make_icon("\U0001F321️")     # thermometer: IoT sensor telemetry
draw_token_icon = make_icon("\U0001F524")            # input latin letters: tokenizer / lexical
draw_chip_icon = make_icon("\U0001F9E0")             # brain: LLM encoder / model
draw_augment_icon = make_icon("\U0001F500")          # shuffle arrows: augmentation / normalize
draw_vit_icon = make_icon("\U0001F441️")        # eye: ViT vision encoder
draw_client_icon = make_icon("\U0001F4F1")           # mobile phone: edge client device
draw_pipeline_icon = make_icon("\U0001F517")         # link: chained pipeline
draw_leaf_icon = make_icon("\U0001F343")             # leaf fluttering: tea leaf image
draw_passages_icon = make_icon("\U0001F4D1")         # bookmark tabs: top-k ranked passages
draw_weather_icon = make_icon("\U0001F326️")    # sun behind rain cloud: live garden telemetry
draw_priority_icon = make_icon("\U0001F6A6")         # traffic light: High/Medium/Low priority
draw_treatment_icon = make_icon("\U0001F48A")        # pill: treatment / advisory action
draw_lock_icon = make_icon("\U0001F512")             # locked padlock: privacy boundary
draw_global_icon = make_icon("\U0001F4E4")           # outbox tray: global model broadcast


def arch_node(ax, x, y, w, h, label, sub, edge, face, icon_fn, icon_color=None, fontsize=10.5):
    rounded_box(ax, (x, y), w, h, edge=edge, face=face, lw=2.2, radius=0.010, zorder=3)
    if icon_fn:
        small_node = w <= 0.150
        icon_size = h * (0.50 if small_node else 0.56)
        icon_fn(ax, x + w * 0.060, y + (h - icon_size) / 2, icon_size, icon_color or edge)
    text_x = x + (w * (0.78 if icon_fn and w <= 0.150 else 0.72) if icon_fn else w * 0.50)
    ax.text(text_x, y + h * 0.58, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#1f2937", zorder=9)
    if sub:
        sub_size = 5.6 if len(sub) > 24 else 6.2
        ax.text(text_x, y + h * 0.34, sub, ha="center", va="center", fontsize=sub_size, color="#374151", zorder=9)


def make_architecture_clipart():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 8.3), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    blue = "#2563eb"
    green = "#16a34a"
    orange = "#d97706"
    red = "#dc2626"
    gold = "#d99a00"
    dark = "#1f2937"
    gray = "#6b7280"

    rounded_box(ax, (0.035, 0.665), 0.392, 0.225, edge=blue, face="#edf6ff", lw=2.5, radius=0.016, zorder=1)
    ax.text(0.222, 0.852, "Text Branch", ha="center", va="center", fontsize=13.5, fontweight="bold", color=blue, zorder=9)
    rounded_box(ax, (0.035, 0.430), 0.392, 0.190, edge=green, face="#effaf0", lw=2.5, radius=0.016, zorder=1)
    ax.text(0.222, 0.585, "Image Branch", ha="center", va="center", fontsize=13.5, fontweight="bold", color=green, zorder=9)

    arch_node(ax, 0.058, 0.700, 0.105, 0.092, "Text", "field notes", blue, "#ffffff", draw_doc_icon, blue, fontsize=9.2)
    arch_node(ax, 0.185, 0.700, 0.112, 0.092, "Tokenizer", "tokens", blue, "#ffffff", draw_token_icon, blue, fontsize=8.4)
    arch_node(ax, 0.322, 0.700, 0.095, 0.092, "LLM", "features", blue, "#ffffff", draw_chip_icon, blue, fontsize=8.8)
    for sx, ex in ((0.163, 0.185), (0.297, 0.322)):
        arch_arrow(ax, (sx, 0.746), (ex, 0.746), color=dark, scale=0.72, shrink=4)

    arch_node(ax, 0.058, 0.460, 0.105, 0.092, "Image", "tea leaf", green, "#ffffff", draw_image_icon, green, fontsize=9.2)
    arch_node(ax, 0.185, 0.460, 0.112, 0.092, "Augment", "normalize", green, "#ffffff", draw_augment_icon, green, fontsize=8.4)
    arch_node(ax, 0.322, 0.460, 0.095, 0.092, "ViT", "features", green, "#ffffff", draw_vit_icon, green, fontsize=8.8)
    for sx, ex in ((0.163, 0.185), (0.297, 0.322)):
        arch_arrow(ax, (sx, 0.506), (ex, 0.506), color=dark, scale=0.72, shrink=4)

    arch_node(ax, 0.465, 0.455, 0.135, 0.105, "VLM Fusion", "combine", red, "#fff1f0", draw_fused_icon, red, fontsize=10.8)
    arch_arrow(ax, (0.417, 0.746), (0.465, 0.510), color=dark, scale=0.78, shrink=7)
    arch_arrow(ax, (0.417, 0.506), (0.465, 0.510), color=dark, scale=0.78, shrink=7)

    arch_node(ax, 0.640, 0.455, 0.145, 0.105, "Classifier", "softmax", red, "#fff1f0", draw_bar_icon, red, fontsize=10.2)
    arch_arrow(ax, (0.600, 0.508), (0.640, 0.508), color=dark, scale=0.82, shrink=5)

    arch_node(ax, 0.825, 0.455, 0.155, 0.105, "5-Class Output", "disease label", gold, "#fff7d6", draw_check_icon, gold, fontsize=10.0)
    arch_arrow(ax, (0.785, 0.508), (0.825, 0.508), color=dark, scale=0.82, shrink=5)

    # FedAvg area: large clip-art server plus long visible client arrows.
    rounded_box(ax, (0.615, 0.745), 0.300, 0.135, edge=orange, face="#fff1dc", lw=2.5, radius=0.014, zorder=3)
    draw_network_icon(ax, 0.635, 0.770, 0.075, orange)
    ax.text(0.790, 0.818, "FedAvg Aggregator", ha="center", va="center", fontsize=14.0, fontweight="bold", color="#1f2937", zorder=9)
    ax.text(0.790, 0.785, "weight averaging only", ha="center", va="center", fontsize=8.0, color="#374151", zorder=9)
    client_x = [0.640, 0.765, 0.890]
    for i, cx in enumerate(client_x, start=1):
        rounded_box(ax, (cx - 0.038, 0.657), 0.076, 0.044, edge=orange, face="#fff7ed", lw=1.9, radius=0.008, zorder=4)
        draw_client_icon(ax, cx - 0.032, 0.664, 0.028, orange)
        ax.text(cx + 0.014, 0.679, f"C{i}", ha="center", va="center", fontsize=8.2, fontweight="bold", color=orange, zorder=9)
        arch_arrow(ax, (cx, 0.705), (cx, 0.745), color=orange, scale=0.70, shrink=2)
    routed_arch_arrow(
        ax,
        [(0.715, 0.745), (0.715, 0.615), (0.705, 0.615), (0.705, 0.560)],
        color=orange,
        scale=0.72,
        dashed=True,
        zorder=9,
        shrink=6,
    )

    # RAG pipeline with compact clip-art nodes.
    rounded_box(ax, (0.405, 0.295), 0.210, 0.050, edge="#cbd5e1", face="#f8fafc", lw=1.8, radius=0.012, zorder=3)
    draw_pipeline_icon(ax, 0.420, 0.306, 0.030, gray)
    ax.text(0.530, 0.320, "RAG Diagnosis Pipeline", ha="center", va="center", fontsize=10.8, fontweight="bold", color="#1f2937", zorder=9)
    arch_arrow(ax, (0.878, 0.455), (0.610, 0.320), color=gray, scale=0.68, dashed=True, rad=-0.18, zorder=9, shrink=6)

    arch_node(ax, 0.055, 0.090, 0.130, 0.095, "Sentence-BERT", "query", blue, "#edf6ff", draw_search_icon, blue, fontsize=8.8)
    arch_node(ax, 0.245, 0.090, 0.130, 0.095, "FAISS KB", "top-k", blue, "#edf6ff", draw_db_icon, blue, fontsize=9.2)
    arch_node(ax, 0.445, 0.090, 0.145, 0.095, "VLM Re-ranker", "image-guided", red, "#fff1f0", draw_fused_icon, red, fontsize=9.2)
    arch_node(ax, 0.650, 0.090, 0.130, 0.095, "IoT Fusion", "weather + soil", orange, "#fff1dc", draw_sensor_icon, orange, fontsize=9.2)
    arch_node(ax, 0.835, 0.090, 0.130, 0.095, "Treatment", "priority", gold, "#fff7d6", draw_treatment_icon, gold, fontsize=9.2)
    for start, end in (((0.185, 0.138), (0.245, 0.138)), ((0.375, 0.138), (0.445, 0.138)), ((0.590, 0.138), (0.650, 0.138)), ((0.780, 0.138), (0.835, 0.138))):
        arch_arrow(ax, start, end, color=dark, scale=0.72, shrink=4)
    arch_arrow(ax, (0.510, 0.295), (0.510, 0.185), color=gray, scale=0.66, dashed=True, zorder=9, shrink=4)

    fig.savefig(OUT / "farmfederate_architecture_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_edgecloud_clipart():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 7.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    green = "#16a34a"
    blue = "#2563eb"
    orange = "#d97706"
    red = "#dc2626"
    gold = "#ca8a04"
    dark = "#1f2937"
    gray = "#6b7280"

    rounded_box(ax, (0.030, 0.165), 0.635, 0.720, edge=green, face="#f0fdf4", lw=2.6, radius=0.014, zorder=0)
    rounded_box(ax, (0.705, 0.165), 0.265, 0.720, edge=blue, face="#eff6ff", lw=2.6, radius=0.014, zorder=0)
    ax.plot([0.685, 0.685], [0.175, 0.865], color=gray, linewidth=2.0, linestyle=(0, (5, 5)), zorder=1)

    ax.text(0.345, 0.820, "Tea Garden Edge Device", ha="center", va="center", fontsize=18.0, fontweight="bold", color=green)
    ax.text(0.835, 0.820, "Global Server", ha="center", va="center", fontsize=18.0, fontweight="bold", color=blue)
    ax.text(0.685, 0.902, "edge / cloud", ha="center", va="center", fontsize=8.0, color=gray)

    arch_node(ax, 0.050, 0.565, 0.135, 0.120, "Local Data", "image + text + IoT", green, "#ffffff", draw_image_icon, green, fontsize=9.2)
    arch_node(ax, 0.205, 0.565, 0.135, 0.120, "Preprocess", "224x224 + tokens", green, "#ffffff", draw_doc_icon, green, fontsize=9.2)
    arch_node(ax, 0.360, 0.565, 0.135, 0.120, "Encoders", "LLM + ViT", green, "#ffffff", draw_chip_icon, green, fontsize=9.2)
    arch_node(ax, 0.515, 0.565, 0.145, 0.120, "VLM + Classifier", "5-class output", red, "#fff1f0", draw_fused_icon, red, fontsize=8.8)
    for start, end in (((0.185, 0.625), (0.205, 0.625)), ((0.340, 0.625), (0.360, 0.625)), ((0.495, 0.625), (0.515, 0.625))):
        arch_arrow(ax, start, end, color=dark, scale=0.68, shrink=3)

    arch_node(ax, 0.280, 0.310, 0.160, 0.115, "Local RAG", "per-estate FAISS + KB", blue, "#eff6ff", draw_db_icon, blue, fontsize=10.0)
    arch_node(ax, 0.485, 0.310, 0.145, 0.115, "Advisory", "priority actions", gold, "#fff7d6", draw_treatment_icon, gold, fontsize=10.0)
    arch_arrow(ax, (0.565, 0.565), (0.415, 0.425), color=dark, scale=0.66, rad=-0.10, shrink=5)
    arch_arrow(ax, (0.440, 0.368), (0.485, 0.368), color=dark, scale=0.60, shrink=4)

    rounded_box(ax, (0.105, 0.195), 0.205, 0.100, edge=gold, face="#fef9c3", lw=2.0, radius=0.010, zorder=3)
    draw_lock_icon(ax, 0.122, 0.215, 0.055, gold)
    ax.text(0.220, 0.252, "Privacy Boundary", ha="center", va="center", fontsize=10.6, fontweight="bold", color=dark, zorder=9)
    ax.text(0.220, 0.225, "raw images, text, and KB stay local", ha="center", va="center", fontsize=6.8, color="#374151", zorder=9)
    routed_arch_arrow(
        ax,
        [(0.113, 0.565), (0.113, 0.385), (0.205, 0.385), (0.205, 0.300)],
        color=gray,
        scale=0.46,
        dashed=True,
        zorder=5,
        shrink=2,
    )

    arch_node(ax, 0.755, 0.560, 0.195, 0.125, "FedAvg Aggregator", "client weight averaging", blue, "#dbeafe", draw_network_icon, blue, fontsize=10.0)
    arch_node(ax, 0.755, 0.285, 0.195, 0.125, "Global Model", "broadcast weights only", orange, "#fff7ed", draw_global_icon, orange, fontsize=10.0)
    arch_arrow(ax, (0.668, 0.625), (0.755, 0.625), color=orange, scale=0.80, dashed=True, zorder=9, shrink=3)
    arch_arrow(ax, (0.852, 0.560), (0.852, 0.410), color=orange, scale=0.82, dashed=True, zorder=9, shrink=4)
    routed_arch_arrow(
        ax,
        [(0.755, 0.348), (0.690, 0.348), (0.690, 0.735), (0.575, 0.735), (0.575, 0.685)],
        color=orange,
        scale=0.68,
        dashed=True,
        zorder=9,
        shrink=4,
    )

    rounded_box(ax, (0.724, 0.185), 0.232, 0.045, edge="#cbd5e1", face="#ffffff", lw=1.4, radius=0.006, zorder=3)
    ax.text(0.840, 0.207, "Only model weights cross the boundary", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#4b5563")

    rounded_box(ax, (0.045, 0.052), 0.405, 0.070, edge="#cbd5e1", face="#ffffff", lw=1.4, radius=0.006, zorder=3)
    ax.plot([0.070, 0.118], [0.087, 0.087], color=dark, linewidth=3.6, zorder=4)
    ax.text(0.133, 0.087, "local processing flow", ha="left", va="center", fontsize=9.4, color="#4b5563", zorder=4)
    ax.plot([0.280, 0.328], [0.087, 0.087], color=orange, linewidth=3.6, linestyle=(0, (5, 5)), zorder=4)
    ax.text(0.343, 0.087, "federated weight update", ha="left", va="center", fontsize=9.4, color="#4b5563", zorder=4)

    fig.savefig(OUT / "farmfederate_edgecloud_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_ragpipeline_clipart():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 8.7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    green = "#16a34a"
    blue = "#2563eb"
    orange = "#d97706"
    red = "#dc2626"
    purple = "#7c3aed"
    gold = "#ca8a04"
    dark = "#1f2937"
    gray = "#6b7280"

    rounded_box(ax, (0.025, 0.065), 0.950, 0.870, edge=blue, face="#f8fafc", lw=2.6, radius=0.014, zorder=0)
    ax.text(0.500, 0.880, "RAG Diagnosis Module", ha="center", va="center", fontsize=19.0, fontweight="bold", color=blue)
    ax.text(
        0.500,
        0.845,
        "Layered retrieval and decision flow: query evidence, local knowledge retrieval, visual re-ranking, sensor adjustment, and treatment actions.",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#4b5563",
    )

    rounded_box(ax, (0.060, 0.170), 0.215, 0.615, edge=green, face="#f0fdf4", lw=2.4, radius=0.014, zorder=1)
    rounded_box(ax, (0.305, 0.170), 0.280, 0.615, edge=blue, face="#eff6ff", lw=2.4, radius=0.014, zorder=1)
    rounded_box(ax, (0.615, 0.170), 0.340, 0.615, edge=orange, face="#fff7ed", lw=2.4, radius=0.014, zorder=1)
    ax.text(0.168, 0.745, "1. Query Evidence", ha="center", va="center", fontsize=12.0, fontweight="bold", color=green)
    ax.text(0.445, 0.745, "2. Local Retrieval", ha="center", va="center", fontsize=12.0, fontweight="bold", color=blue)
    ax.text(0.780, 0.745, "3. Decision Context", ha="center", va="center", fontsize=12.0, fontweight="bold", color=orange)

    arch_node(ax, 0.085, 0.620, 0.165, 0.075, "Agronomist Query", "field notes, symptoms", green, "#ffffff", draw_doc_icon, green, fontsize=8.5)
    arch_node(ax, 0.085, 0.475, 0.165, 0.075, "Optional Leaf Image", "visual evidence", green, "#ffffff", draw_leaf_icon, green, fontsize=8.5)
    arch_node(ax, 0.085, 0.255, 0.165, 0.075, "TF-IDF Fallback", "lexical support", gray, "#f8fafc", draw_token_icon, gray, fontsize=8.5)

    arch_node(ax, 0.335, 0.620, 0.230, 0.075, "Sentence-BERT Encoder", "dense query vector", blue, "#dbeafe", draw_search_icon, blue, fontsize=8.5)
    arch_node(ax, 0.335, 0.475, 0.230, 0.075, "FAISS IndexFlatIP", "top-k cosine search over local KB", blue, "#dbeafe", draw_db_icon, blue, fontsize=8.5)
    arch_node(ax, 0.335, 0.255, 0.105, 0.075, "5-Class KB", "treatment docs", gold, "#fef9c3", draw_db_icon, gold, fontsize=8.0)
    arch_node(ax, 0.455, 0.255, 0.110, 0.075, "Top-k Passages", "ranked candidates", blue, "#dbeafe", draw_passages_icon, blue, fontsize=8.0)

    decision_x = 0.650
    decision_w = 0.140
    side_x = 0.845
    side_w = 0.105
    decision_r = decision_x + decision_w
    side_l = side_x
    decision_center_x = decision_x + decision_w / 2
    side_center_x = side_x + side_w / 2

    arch_node(ax, decision_x, 0.620, decision_w, 0.075, "VLM Re-ranker", "image evidence", red, "#fee2e2", draw_fused_icon, red, fontsize=8.2)
    arch_node(ax, side_x, 0.620, side_w, 0.075, "Live Sensors", "garden telemetry", orange, "#fef3c7", draw_weather_icon, orange, fontsize=7.5)
    arch_node(ax, decision_x, 0.475, decision_w, 0.075, "IoT Sensor Fusion", "temp, humidity, soil", orange, "#ffedd5", draw_sensor_icon, orange, fontsize=8.1)
    arch_node(ax, decision_x, 0.315, decision_w, 0.075, "Priority Scorer", "confidence + urgency", purple, "#f3e8ff", draw_priority_icon, purple, fontsize=8.1)
    # Centered on the combined span of IoT Sensor Fusion + Priority Scorer (its two
    # inputs), the same way Live Sensors is centered on VLM Re-ranker above.
    arch_node(ax, side_x, 0.3725, side_w, 0.120, "Treatment Plan", "High, Medium, Low", gold, "#fef9c3", draw_treatment_icon, gold, fontsize=7.5)

    arch_arrow(ax, (0.250, 0.658), (0.335, 0.658), color=dark, scale=0.62, shrink=3)
    arch_arrow(ax, (0.450, 0.620), (0.450, 0.550), color=dark, scale=0.55, shrink=3)
    arch_arrow(ax, (0.510, 0.475), (0.510, 0.330), color=dark, scale=0.55, shrink=3)
    routed_arch_arrow(
        ax,
        [(0.565, 0.292), (0.603, 0.292), (0.603, 0.640), (decision_x - 0.004, 0.640)],
        color=dark,
        scale=0.38,
        dashed=False,
        zorder=9,
        shrink=0,
    )
    arch_arrow(ax, (decision_r + 0.004, 0.658), (side_l - 0.004, 0.658), color=dark, scale=0.30, shrink=0)
    arch_arrow(ax, (decision_center_x, 0.616), (decision_center_x, 0.554), color=dark, scale=0.50, shrink=0)
    arch_arrow(ax, (decision_center_x, 0.471), (decision_center_x, 0.394), color=dark, scale=0.50, shrink=0)
    arch_arrow(ax, (decision_r + 0.004, 0.355), (side_l - 0.004, 0.420), color=dark, scale=0.36, shrink=0)
    routed_arch_arrow(
        ax,
        [(side_center_x - 0.005, 0.616), (side_center_x - 0.005, 0.578), (decision_r + 0.004, 0.535)],
        color=orange,
        scale=0.40,
        dashed=True,
        zorder=9,
        shrink=0,
    )

    routed_arch_arrow(
        ax,
        [(0.254, 0.5125), (0.285, 0.5125), (0.285, 0.716), (decision_x + 0.018, 0.716), (decision_x + 0.018, 0.699)],
        color=red,
        scale=0.46,
        dashed=True,
        zorder=9,
        shrink=0,
    )
    ax.text(0.482, 0.695, "optional image path to visual re-ranker", ha="center", va="center", fontsize=6.6, color=red)
    routed_arch_arrow(
        ax,
        [(0.254, 0.292), (0.294, 0.292), (0.294, 0.490), (0.331, 0.490)],
        color=gray,
        scale=0.52,
        dashed=True,
        zorder=9,
        shrink=0,
    )
    routed_arch_arrow(
        ax,
        [(0.390, 0.334), (0.390, 0.430), (0.450, 0.471)],
        color=gold,
        scale=0.52,
        dashed=True,
        zorder=9,
        shrink=0,
    )

    ax.plot([0.060, 0.098], [0.105, 0.105], color=dark, linewidth=3.0)
    ax.text(0.108, 0.105, "main flow", ha="left", va="center", fontsize=7.2, color="#4b5563")
    ax.plot([0.180, 0.218], [0.105, 0.105], color=red, linewidth=2.6, linestyle=(0, (5, 5)))
    ax.text(0.228, 0.105, "optional image path", ha="left", va="center", fontsize=7.2, color="#4b5563")
    ax.plot([0.325, 0.363], [0.105, 0.105], color=gray, linewidth=2.4, linestyle=(0, (5, 5)))
    ax.text(0.373, 0.105, "fallback / support path", ha="left", va="center", fontsize=7.2, color="#4b5563")

    fig.savefig(OUT / "farmfederate_ragpipeline_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_architecture_clipart()
    make_edgecloud_clipart()
    make_vlm_fusion_clipart()
    make_ragpipeline_clipart()
