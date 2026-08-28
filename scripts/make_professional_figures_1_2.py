from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

from clipart_icons import make_icon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"

COLORS = {
    "ink": "#1f2937",
    "muted": "#64748b",
    "line": "#334155",
    "blue": "#2563eb",
    "blue_soft": "#eef6ff",
    "green": "#16a34a",
    "green_soft": "#effaf2",
    "orange": "#d97706",
    "orange_soft": "#fff4df",
    "red": "#dc2626",
    "red_soft": "#fff1f0",
    "gold": "#ca8a04",
    "gold_soft": "#fff8d8",
    "violet": "#7c3aed",
    "violet_soft": "#f4f1ff",
    "slate_soft": "#f8fafc",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def box(ax, x, y, w, h, edge, face, lw=2.0, r=0.012, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


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


def arrow(ax, a, b, color=None, dashed=False, rad=0.0, lw=2.2, z=10, ms=18, shrinkA=7.0, shrinkB=7.0):
    color = color or COLORS["line"]
    style = (0, (5, 4)) if dashed else "solid"
    # On short hops, cap shrink and head size so a visible shaft always remains.
    gap = _dist_pts(ax, a, b)
    if shrinkA + shrinkB + 0.4 * ms > 0.62 * gap:
        shrinkA = min(shrinkA, 5.0)
        shrinkB = min(shrinkB, 5.0)
        inner = max(gap - shrinkA - shrinkB, 4.0)
        ms = min(ms, inner * 1.5)  # head length (0.4*ms) at most ~60% of remaining span
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=style,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=shrinkA,
            shrinkB=shrinkB,
            zorder=z + 1,
            joinstyle="round",
            capstyle="butt",
            path_effects=_halo(lw),
        )
    )


def route_arrow(ax, pts, color=None, dashed=True, lw=2.1, z=10, ms=17, shrinkA=0.0, shrinkB=4.0):
    color = color or COLORS["orange"]
    style = (0, (5, 4)) if dashed else "solid"
    pts = list(pts)
    pts[0] = _trim_toward(ax, pts[1], pts[0], shrinkA)
    pts[-1] = _trim_toward(ax, pts[-2], pts[-1], shrinkB)
    # Keep the head short enough to fit inside the final segment.
    last = _dist_pts(ax, pts[-2], pts[-1])
    ms = min(ms, max(2.0 * last, 8.0))
    path = MplPath(pts, [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=style,
            zorder=z + 1,
            joinstyle="round",
            capstyle="butt",
            path_effects=_halo(lw),
        )
    )


def title(ax, x, y, text, color=None, fs=12):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=color or COLORS["ink"], zorder=20)


def subtitle(ax, x, y, text, fs=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=COLORS["muted"], zorder=20)


# Emoji clipart icons (Segoe UI Emoji, full color).  Signature: fn(ax, x, y, s, color).
draw_leaf = make_icon("\U0001F343")               # leaf fluttering: tea leaf image input
draw_doc = make_icon("\U0001F4DD")                # memo: text / field notes
draw_sensor = make_icon("\U0001F321️")       # thermometer: IoT sensor telemetry
draw_chip = make_icon("\U0001F9E0")               # brain: LLM encoder / local training
draw_vision = make_icon("\U0001F441️")       # eye: ViT vision encoder
draw_network = make_icon("\U0001F310")            # globe with meridians: FedAvg network
draw_db = make_icon("\U0001F5C4️")           # file cabinet: FAISS knowledge base
draw_bars = make_icon("\U0001F4CA")               # bar chart: classifier / disease head
draw_check = make_icon("✅")                  # check mark button: verified output
draw_fusion = make_icon("\U0001F9E9")             # puzzle piece: VLM fusion
draw_search = make_icon("\U0001F50D")             # magnifying glass: Sentence-BERT retrieval
draw_treatment = make_icon("\U0001F48A")          # pill: treatment / advisory action
draw_lock = make_icon("\U0001F512")               # locked padlock: raw data stays local
draw_global = make_icon("\U0001F4E4")             # outbox tray: global model broadcast
draw_phone = make_icon("\U0001F4F1")              # mobile phone: client device C1
draw_gateway = make_icon("\U0001F4E1")            # satellite antenna: client device C2
draw_laptop = make_icon("\U0001F4BB")             # laptop: client device C3
CLIENT_ICONS = (draw_phone, draw_gateway, draw_laptop)


def node(ax, x, y, w, h, heading, sub, edge, face, icon=None, fs=9.0, icon_color=None):
    box(ax, x, y, w, h, edge, face, lw=2.0, r=0.011, z=5)
    if icon:
        s = h * 0.58
        # Center the emoji clipart on a fixed left slot so it never collides with the heading.
        icon(ax, x + 0.17 * w - s / 2, y + (h - s) / 2, s, icon_color or edge)
        tx = x + w * 0.66
    else:
        tx = x + w * 0.50
    title(ax, tx, y + h * 0.60, heading, fs=fs)
    if sub:
        subtitle(ax, tx, y + h * 0.34, sub, fs=max(5.9, fs - 2.5))


def make_figure_1():
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 7.7), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Major regions.
    box(ax, 0.025, 0.120, 0.675, 0.800, COLORS["green"], COLORS["green_soft"], lw=2.5, r=0.018, z=0)
    box(ax, 0.735, 0.540, 0.240, 0.380, COLORS["orange"], COLORS["orange_soft"], lw=2.5, r=0.018, z=0)
    box(ax, 0.200, 0.135, 0.490, 0.210, "#cbd5e1", COLORS["slate_soft"], lw=1.5, r=0.012, z=1)
    title(ax, 0.363, 0.875, "On-Device Multimodal Diagnosis", COLORS["green"], fs=15.5)
    title(ax, 0.855, 0.875, "Federated Coordination", COLORS["orange"], fs=14.0)
    title(ax, 0.445, 0.314, "Local RAG Advisory", COLORS["blue"], fs=12.0)

    # Inputs.
    node(ax, 0.055, 0.665, 0.132, 0.115, "Leaf Image", "field photo", COLORS["green"], "white", draw_leaf, fs=9.5)
    node(ax, 0.055, 0.505, 0.132, 0.115, "Field Notes", "symptoms", COLORS["blue"], "white", draw_doc, fs=9.5)
    node(ax, 0.055, 0.345, 0.132, 0.115, "IoT Signals", "weather + soil", COLORS["orange"], "white", draw_sensor, fs=9.5)

    # Encoders and diagnosis.
    node(ax, 0.235, 0.665, 0.135, 0.115, "ViT Encoder", "visual features", COLORS["green"], "white", draw_vision, fs=9.2)
    node(ax, 0.235, 0.505, 0.135, 0.115, "LLM Encoder", "text features", COLORS["blue"], "white", draw_chip, fs=9.2)
    node(ax, 0.405, 0.585, 0.135, 0.130, "VLM Fusion", "8 fusion strategies", COLORS["red"], COLORS["red_soft"], draw_fusion, fs=9.8)
    node(ax, 0.565, 0.585, 0.120, 0.130, "Disease Head", "5-class softmax", COLORS["red"], "white", draw_bars, fs=9.0)
    node(ax, 0.565, 0.405, 0.120, 0.120, "OBB Visualizer", "explain regions", COLORS["gold"], COLORS["gold_soft"], draw_check, fs=8.7)

    arrow(ax, (0.187, 0.722), (0.235, 0.722), COLORS["line"])
    arrow(ax, (0.187, 0.562), (0.235, 0.562), COLORS["line"])
    arrow(ax, (0.370, 0.722), (0.405, 0.662), COLORS["line"], rad=-0.05)
    arrow(ax, (0.370, 0.562), (0.405, 0.638), COLORS["line"], rad=0.05)
    arrow(ax, (0.540, 0.650), (0.565, 0.650), COLORS["line"])
    arrow(ax, (0.625, 0.585), (0.625, 0.525), COLORS["gold"])

    # RAG advisory. Boxes are spaced with a wider gap than a plain box touch so the
    # connector arrows get a visible shaft + head instead of a cramped chevron.
    node(ax, 0.216, 0.180, 0.092, 0.090, "SBERT", "query embed", COLORS["blue"], "white", draw_search, fs=7.6)
    node(ax, 0.338, 0.180, 0.092, 0.090, "FAISS KB", "local top-k", COLORS["blue"], "white", draw_db, fs=7.6)
    node(ax, 0.460, 0.180, 0.092, 0.090, "IoT Re-rank", "context boost", COLORS["orange"], "white", draw_sensor, fs=7.6)
    node(ax, 0.582, 0.180, 0.092, 0.090, "Treatment", "priority plan", COLORS["gold"], COLORS["gold_soft"], draw_treatment, fs=7.6)
    arrow(ax, (0.308, 0.225), (0.338, 0.225), COLORS["ink"], lw=2.6, ms=21, shrinkA=3.0, shrinkB=3.0)
    arrow(ax, (0.430, 0.225), (0.460, 0.225), COLORS["ink"], lw=2.6, ms=21, shrinkA=3.0, shrinkB=3.0)
    arrow(ax, (0.552, 0.225), (0.582, 0.225), COLORS["ink"], lw=2.6, ms=21, shrinkA=3.0, shrinkB=3.0)
    route_arrow(ax, [(0.625, 0.405), (0.625, 0.292), (0.630, 0.292), (0.630, 0.270)], color=COLORS["gold"], dashed=True, lw=1.8, ms=15, shrinkB=1.5)
    route_arrow(ax, [(0.121, 0.345), (0.121, 0.292), (0.507, 0.292), (0.507, 0.270)], color=COLORS["orange"], dashed=True, lw=1.8, ms=15, shrinkB=1.5)

    # Federated loop.
    node(ax, 0.770, 0.710, 0.165, 0.115, "FedAvg Server", "weight averaging", COLORS["orange"], "white", draw_network, fs=9.8)
    for i, cx in enumerate((0.775, 0.845, 0.915), start=1):
        box(ax, cx - 0.032, 0.600, 0.064, 0.055, COLORS["orange"], "white", lw=1.6, r=0.007, z=5)
        CLIENT_ICONS[i - 1](ax, cx - 0.029, 0.605, 0.045, COLORS["orange"])
        ax.text(cx + 0.020, 0.629, f"C{i}", ha="center", va="center", fontsize=7.0, fontweight="bold", color=COLORS["orange"], zorder=20)
        arrow(ax, (cx, 0.655), (cx, 0.710), COLORS["orange"], dashed=True, lw=1.7, ms=14)
    node(ax, 0.785, 0.405, 0.145, 0.090, "Global Model", "broadcast only", COLORS["orange"], "white", draw_global, fs=8.5)
    # Averaged weights: route around the client cards (right channel) instead of through C2.
    route_arrow(ax, [(0.939, 0.7675), (0.958, 0.7675), (0.958, 0.450), (0.934, 0.450)], color=COLORS["orange"], dashed=True, lw=1.9, ms=16, shrinkA=0.0, shrinkB=0.0)
    # Broadcast back to the on-device model: clean horizontal entry into Disease Head.
    route_arrow(ax, [(0.781, 0.450), (0.720, 0.450), (0.720, 0.650), (0.690, 0.650)], color=COLORS["orange"], dashed=True, lw=1.8, ms=15, shrinkA=0.0, shrinkB=0.0)

    # Privacy callout.
    box(ax, 0.052, 0.150, 0.120, 0.090, COLORS["gold"], COLORS["gold_soft"], lw=1.8, r=0.009, z=5)
    draw_lock(ax, 0.060, 0.164, 0.060, COLORS["gold"])
    title(ax, 0.128, 0.205, "Raw Data", fs=8.2)
    subtitle(ax, 0.128, 0.180, "stays local", fs=7.1)

    # Legend.
    box(ax, 0.755, 0.125, 0.205, 0.055, "#cbd5e1", "white", lw=1.2, r=0.007, z=5)
    ax.plot([0.775, 0.815], [0.153, 0.153], color=COLORS["line"], linewidth=3.0, zorder=20)
    ax.text(0.822, 0.153, "local inference", ha="left", va="center", fontsize=7.6, color=COLORS["muted"], zorder=20)
    ax.plot([0.885, 0.925], [0.153, 0.153], color=COLORS["orange"], linewidth=3.0, linestyle=(0, (5, 4)), zorder=20)
    ax.text(0.932, 0.153, "weights", ha="left", va="center", fontsize=7.6, color=COLORS["muted"], zorder=20)

    fig.savefig(OUT / "farmfederate_architecture_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figure_2():
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(18.8, 7.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Main regions.
    box(ax, 0.030, 0.135, 0.620, 0.780, COLORS["green"], COLORS["green_soft"], lw=2.5, r=0.018, z=0)
    box(ax, 0.705, 0.135, 0.265, 0.780, COLORS["blue"], COLORS["blue_soft"], lw=2.5, r=0.018, z=0)
    ax.plot([0.675, 0.675], [0.145, 0.905], color=COLORS["muted"], linewidth=2.2, linestyle=(0, (5, 5)), zorder=1)
    title(ax, 0.340, 0.865, "Tea Garden Edge Layer", COLORS["green"], fs=15.5)
    title(ax, 0.838, 0.865, "Cloud Coordination Layer", COLORS["blue"], fs=15.5)
    subtitle(ax, 0.675, 0.935, "privacy boundary", fs=8.0)

    # Three client cards.
    client_specs = [
        (0.060, 0.585, "C1", "Garden Phone"),
        (0.260, 0.585, "C2", "Edge Gateway"),
        (0.460, 0.585, "C3", "Field Tablet"),
    ]
    for x, y, cid, label in client_specs:
        box(ax, x, y, 0.155, 0.185, COLORS["green"], "white", lw=2.0, r=0.012, z=5)
        ax.text(x + 0.015, y + 0.158, cid, ha="left", va="center", fontsize=9.5, fontweight="bold", color=COLORS["green"], zorder=20)
        title(ax, x + 0.088, y + 0.142, label, fs=8.4)
        draw_leaf(ax, x + 0.020, y + 0.070, 0.055, COLORS["green"])
        draw_doc(ax, x + 0.062, y + 0.070, 0.055, COLORS["blue"])
        draw_sensor(ax, x + 0.105, y + 0.070, 0.055, COLORS["orange"])
        subtitle(ax, x + 0.078, y + 0.042, "image + notes + IoT", fs=6.8)
        route_arrow(ax, [(x + 0.078, y), (x + 0.078, 0.520), (0.780, 0.520), (0.780, 0.607)], color=COLORS["orange"], dashed=True, lw=1.6, ms=14, shrinkA=0.0, shrinkB=0.0)

    # Local processing blocks.
    node(ax, 0.090, 0.315, 0.145, 0.120, "Local Training", "LLM + ViT + VLM", COLORS["green"], "white", draw_chip, fs=9.0)
    node(ax, 0.275, 0.315, 0.145, 0.120, "Local RAG", "private FAISS KB", COLORS["blue"], "white", draw_db, fs=9.0)
    node(ax, 0.460, 0.315, 0.145, 0.120, "Advisory", "treatment actions", COLORS["gold"], COLORS["gold_soft"], draw_treatment, fs=9.0)
    arrow(ax, (0.235, 0.375), (0.275, 0.375), COLORS["line"], lw=2.0, ms=16)
    arrow(ax, (0.420, 0.375), (0.460, 0.375), COLORS["line"], lw=2.0, ms=16)

    box(ax, 0.075, 0.185, 0.520, 0.070, COLORS["gold"], COLORS["gold_soft"], lw=1.7, r=0.010, z=5)
    draw_lock(ax, 0.088, 0.196, 0.055, COLORS["gold"])
    title(ax, 0.330, 0.232, "Raw leaf images, symptom logs, and knowledge-base documents stay on the edge device", fs=8.6)
    subtitle(ax, 0.330, 0.207, "Only model parameters are transmitted during federated learning", fs=7.4)

    # Cloud blocks.
    node(ax, 0.755, 0.610, 0.165, 0.125, "FedAvg Aggregator", "weighted client updates", COLORS["blue"], "white", draw_network, fs=9.3)
    node(ax, 0.755, 0.355, 0.165, 0.125, "Global Model", "updated weights", COLORS["orange"], "white", draw_global, fs=9.3)
    box(ax, 0.735, 0.205, 0.205, 0.070, "#cbd5e1", "white", lw=1.3, r=0.008, z=5)
    title(ax, 0.838, 0.244, "No Raw Data Upload", fs=9.0)
    subtitle(ax, 0.838, 0.220, "privacy-preserving coordination", fs=7.3)

    # Cross-boundary flows.
    arrow(ax, (0.650, 0.520), (0.755, 0.670), COLORS["orange"], dashed=True, lw=2.2, ms=18)
    arrow(ax, (0.838, 0.610), (0.838, 0.480), COLORS["orange"], dashed=True, lw=2.1, ms=18)
    # Keep the return route off the dashed privacy divider (x=0.675).
    route_arrow(ax, [(0.751, 0.418), (0.692, 0.418), (0.692, 0.500), (0.520, 0.500), (0.520, 0.439)], color=COLORS["orange"], dashed=True, lw=1.9, ms=16, shrinkA=0.0, shrinkB=0.0)

    # Legend.
    box(ax, 0.070, 0.060, 0.440, 0.055, "#cbd5e1", "white", lw=1.2, r=0.007, z=5)
    ax.plot([0.095, 0.135], [0.088, 0.088], color=COLORS["line"], linewidth=3.0, zorder=20)
    ax.text(0.145, 0.088, "local computation and advisory flow", ha="left", va="center", fontsize=7.7, color=COLORS["muted"], zorder=20)
    ax.plot([0.320, 0.360], [0.088, 0.088], color=COLORS["orange"], linewidth=3.0, linestyle=(0, (5, 4)), zorder=20)
    ax.text(0.370, 0.088, "federated weight update / broadcast", ha="left", va="center", fontsize=7.7, color=COLORS["muted"], zorder=20)

    fig.savefig(OUT / "farmfederate_edgecloud_clipart.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
