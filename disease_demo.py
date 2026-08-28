"""
Tea Leaf Disease Detection Demo
- Image input  -> draw OBB annotations (polygon) + disease label + remedies
- Text input   -> show annotated reference image for that disease + remedies
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import gradio as gr
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
MERGED_DIR = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Merged-Dataset")
REAL_IMGS  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\images")
REAL_LABS  = Path(r"C:\Users\USER_HP\Desktop\FarmFederate\Real Dataset\labels")

CLASS_NAMES = ["LEAF_BLIGHT", "LEAF_HOPPERS", "LEAF_RUST",
               "LOOPER_CATERPILLARS", "MOSQUITO_BUG"]

DISPLAY = {
    "LEAF_BLIGHT":         "Leaf Blight",
    "LEAF_HOPPERS":        "Leaf Hoppers",
    "LEAF_RUST":           "Leaf Rust",
    "LOOPER_CATERPILLARS": "Looper Caterpillars",
    "MOSQUITO_BUG":        "Mosquito Bug",
}

# BGR colours for cv2
COLORS = {
    "LEAF_BLIGHT":         (34,  139, 34),
    "LEAF_HOPPERS":        (220, 50,  50),
    "LEAF_RUST":           (30,  144, 255),
    "LOOPER_CATERPILLARS": (148, 0,   211),
    "MOSQUITO_BUG":        (0,   206, 209),
}

REMEDIES = {
    "LEAF_BLIGHT": [
        "Remove and destroy infected leaves immediately.",
        "Apply copper-based fungicide (Bordeaux mixture) every 10-14 days.",
        "Improve air circulation by pruning dense canopy.",
        "Avoid overhead irrigation; water at the base in the morning.",
        "Ensure balanced nutrition — excess nitrogen increases susceptibility.",
    ],
    "LEAF_HOPPERS": [
        "Spray neem oil (5 ml/L) or insecticidal soap on leaf undersides.",
        "Introduce natural predators such as lacewings or parasitic wasps.",
        "Use yellow sticky traps to monitor and reduce adult populations.",
        "Remove heavily infested shoots and destroy them.",
        "Apply imidacloprid only in severe outbreaks.",
    ],
    "LEAF_RUST": [
        "Apply sulphur-based or triazole fungicide at first sign.",
        "Remove heavily infected leaves and dispose off-site.",
        "Avoid wetting foliage; use drip irrigation where possible.",
        "Ensure adequate potassium levels to strengthen cell walls.",
        "Rotate fungicide classes to prevent resistance.",
    ],
    "LOOPER_CATERPILLARS": [
        "Hand-pick caterpillars and egg masses in small-scale gardens.",
        "Spray Bacillus thuringiensis (Bt) — safe biological control.",
        "Apply spinosad or pyrethrin-based spray in the evening.",
        "Install pheromone traps to monitor moth populations.",
        "Encourage birds and parasitic wasps as natural predators.",
    ],
    "MOSQUITO_BUG": [
        "Spray cypermethrin (0.005%) during cooler hours.",
        "Remove weeds and alternate hosts around the plantation.",
        "Use light traps at night to reduce adult populations.",
        "Apply systemic insecticide in severe cases, respecting PHI.",
        "Maintain field hygiene by removing crop debris after harvest.",
    ],
}

TEXT_ALIASES = {
    "leaf blight":        "LEAF_BLIGHT",
    "brown blight":       "LEAF_BLIGHT",
    "gray blight":        "LEAF_BLIGHT",
    "grey blight":        "LEAF_BLIGHT",
    "algal leaf":         "LEAF_BLIGHT",
    "blight":             "LEAF_BLIGHT",
    "leaf hoppers":       "LEAF_HOPPERS",
    "leafhopper":         "LEAF_HOPPERS",
    "helopeltis":         "LEAF_HOPPERS",
    "hopper":             "LEAF_HOPPERS",
    "jassid":             "LEAF_HOPPERS",
    "leaf rust":          "LEAF_RUST",
    "rust":               "LEAF_RUST",
    "looper caterpillar": "LOOPER_CATERPILLARS",
    "looper":             "LOOPER_CATERPILLARS",
    "caterpillar":        "LOOPER_CATERPILLARS",
    "worm":               "LOOPER_CATERPILLARS",
    "mosquito bug":       "MOSQUITO_BUG",
    "mosquito":           "MOSQUITO_BUG",
}

# ── Model ────────────────────────────────────────────────────────────────────
print("Loading ResNet-50...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = torch.nn.Identity()
backbone.eval().to(device)

tfm = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def embed(pil_img: Image.Image) -> torch.Tensor:
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        f = backbone(x)
    return F.normalize(f.squeeze(0), dim=0).cpu()


# Build reference library
print("Building reference embeddings...")
ref_embs: dict[str, list[torch.Tensor]] = {c: [] for c in CLASS_NAMES}
ref_imgs: dict[str, list[Path]]         = {c: [] for c in CLASS_NAMES}

for cls in CLASS_NAMES:
    for p in (MERGED_DIR / cls).glob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                ref_embs[cls].append(embed(Image.open(p)))
                ref_imgs[cls].append(p)
            except Exception:
                pass

ref_mean = {c: F.normalize(torch.stack(e).mean(0), dim=0)
            for c, e in ref_embs.items() if e}

print("Ready.", {c: len(ref_imgs[c]) for c in CLASS_NAMES})


# ── OBB drawing helpers ───────────────────────────────────────────────────────
def draw_obb_label(img_bgr: np.ndarray, pts: list[tuple],
                   label: str, color: tuple, font_scale: float = 0.55,
                   thickness: int = 2) -> None:
    """Draw one OBB polygon + label on img_bgr in-place."""
    H, W = img_bgr.shape[:2]
    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img_bgr, [arr], isClosed=True, color=color, thickness=thickness)

    # Label background at the top-left point of the box
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x0, y0 = arr[0]
    rx = min(max(0, int(x0) - 2), max(0, W - tw - 8))
    ry = int(y0) - th - bl - 6
    if ry < 0:
        ry = min(max(0, int(y0) + 6), max(0, H - th - bl - 8))
    cv2.rectangle(img_bgr, (rx, ry), (rx + tw + 6, ry + th + bl + 6), color, -1)
    cv2.putText(img_bgr, label, (rx + 3, ry + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def parse_yolo_obb(label_path: Path, W: int, H: int) -> list[tuple[int, list]]:
    """Return list of (class_id, [(x,y)x4]) from a YOLO OBB .txt file."""
    results = []
    if not label_path.exists():
        return results
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                cid = int(parts[0])
                coords = list(map(float, parts[1:9]))
                pts = [(int(coords[i] * W), int(coords[i + 1] * H))
                       for i in range(0, 8, 2)]
                results.append((cid, pts))
            except ValueError:
                pass
    return results


def detect_leaf_obb(img_bgr: np.ndarray) -> list[list[tuple]]:
    """
    Detect leaf regions via green-channel masking + contour -> rotated OBB.
    Returns list of polygon point lists (4 pts each).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Broad green/yellow-green range typical of leaves
    mask = cv2.inRange(hsv, (25, 30, 40), (95, 255, 255))
    # Fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img_bgr.shape[:2]
    min_area = 0.01 * W * H   # ignore tiny noise
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        pts  = cv2.boxPoints(rect)
        boxes.append([tuple(map(int, p)) for p in pts])

    # Fallback: whole-image OBB
    if not boxes:
        m = 15
        boxes = [[(m, m), (W - m, m), (W - m, H - m), (m, H - m)]]

    return boxes


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def leaf_context_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Broad plant/lesion context used to suppress background pixels."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (18, 20, 35), (105, 255, 255))
    lesion_tones = cv2.inRange(hsv, (0, 25, 25), (38, 255, 235))
    context = green
    if cv2.countNonZero(context) < 0.02 * W * H:
        context = cv2.bitwise_or(context, lesion_tones)

    k = _odd_kernel(min(W, H) / 80, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    context = cv2.morphologyEx(context, cv2.MORPH_CLOSE, kernel)
    context = cv2.morphologyEx(context, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(context, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80.0, 0.002 * W * H)
    max_dist = max(1.0, (W * W + H * H) ** 0.5 / 2)
    scored = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dist = ((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5
        centrality = 1.0 - min(1.0, dist / max_dist)
        scored.append((area * (0.45 + 0.55 * centrality), contour))
    if scored:
        focused = np.zeros_like(context)
        for _, contour in sorted(scored, key=lambda item: item[0], reverse=True)[:3]:
            cv2.drawContours(focused, [contour], -1, 255, -1)
        if cv2.countNonZero(focused) >= min_area:
            context = focused

    dilate_k = _odd_kernel(min(W, H) / 35, 7)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    return cv2.dilate(context, dilate_kernel, iterations=1)


def color_anomaly_mask(img_bgr: np.ndarray, context: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(context) < 50:
        return np.zeros(context.shape, dtype=np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[context > 0]
    median = np.median(pixels, axis=0)
    dist = np.sqrt(np.sum((lab - median) ** 2, axis=2))
    context_dist = dist[context > 0]
    threshold = max(18.0, float(np.percentile(context_dist, 82)))
    return (((dist >= threshold) & (context > 0)).astype(np.uint8) * 255)


def disease_region_mask(img_bgr: np.ndarray, cls: str) -> np.ndarray:
    """Class-aware disease/pest injury mask for uploaded leaves."""
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = [c.astype(np.int16) for c in cv2.split(img_bgr)]
    context = leaf_context_mask(img_bgr)
    anomaly = color_anomaly_mask(img_bgr, context)

    orange_rust = (
        (h >= 4) & (h <= 34) & (s >= 55) & (v >= 65) &
        (r >= g - 8) & (g >= b + 4)
    )
    brown_necrosis = (
        (((h <= 32) | (h >= 170)) & (s >= 35) & (v >= 35) & (v <= 215) &
         (r >= g + 4) & (r >= b + 8))
    )
    dark_dead_tissue = ((v <= 90) & (s >= 18) & (r >= g - 4) & (r >= b + 5))
    pale_stipple = (
        (((h >= 18) & (h <= 58) & (s >= 18) & (s <= 175) & (v >= 115)) |
         ((s <= 65) & (v >= 145))) &
        (anomaly > 0)
    )
    feeding_holes = (((v <= 70) | ((s <= 38) & (v >= 170))) & (anomaly > 0))

    if cls == "LEAF_RUST":
        mask = orange_rust
    elif cls == "LEAF_BLIGHT":
        mask = brown_necrosis | dark_dead_tissue
    elif cls == "LEAF_HOPPERS":
        mask = pale_stipple | ((brown_necrosis | dark_dead_tissue) & (anomaly > 0))
    elif cls == "LOOPER_CATERPILLARS":
        mask = feeding_holes | brown_necrosis | ((anomaly > 0) & (v <= 185))
    elif cls == "MOSQUITO_BUG":
        mask = brown_necrosis | dark_dead_tissue | ((anomaly > 0) & (v <= 190))
    else:
        mask = anomaly > 0

    mask = ((mask & (context > 0)).astype(np.uint8) * 255)
    open_k = _odd_kernel(min(W, H) / 180, 3)
    close_k = _odd_kernel(min(W, H) / 90, 5)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def boxes_from_mask(mask: np.ndarray, min_area_ratio: float = 0.0005,
                    max_boxes: int = 8) -> list[list[tuple]]:
    H, W = mask.shape[:2]
    min_area = max(12.0, min_area_ratio * W * H)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    boxes = []
    for contour in contours[:max_boxes]:
        pts = cv2.boxPoints(cv2.minAreaRect(contour))
        boxes.append([tuple(map(int, p)) for p in pts])
    return boxes


def detect_disease_obb(img_bgr: np.ndarray, cls: str) -> tuple[list[list[tuple]], np.ndarray, bool]:
    mask = disease_region_mask(img_bgr, cls)
    boxes = boxes_from_mask(mask)
    if boxes:
        return boxes, mask, True

    context = leaf_context_mask(img_bgr)
    fallback_boxes = boxes_from_mask(context, min_area_ratio=0.01, max_boxes=3)
    if fallback_boxes:
        return fallback_boxes, mask, False
    return detect_leaf_obb(img_bgr), mask, False


def annotate_disease_regions(img_bgr: np.ndarray, cls: str, confidence: float,
                             color: tuple, font_scale: float,
                             thickness: int) -> None:
    H, W = img_bgr.shape[:2]
    boxes, mask, used_disease_mask = detect_disease_obb(img_bgr, cls)
    if used_disease_mask:
        overlay = img_bgr.copy()
        overlay[mask > 0] = (
            0.45 * np.array(color, dtype=np.float32) +
            0.55 * overlay[mask > 0].astype(np.float32)
        ).astype(np.uint8)
        blended = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
        img_bgr[:] = blended
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        visible = [c for c in contours if cv2.contourArea(c) >= max(12.0, 0.0005 * W * H)]
        cv2.drawContours(img_bgr, visible, -1, color, max(1, thickness))

    label = f"{DISPLAY[cls]} area {confidence * 100:.0f}%"
    if not used_disease_mask:
        label = f"{DISPLAY[cls]} suspected {confidence * 100:.0f}%"
    for i, pts in enumerate(boxes):
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img_bgr, [arr], isClosed=True, color=color, thickness=thickness)
        if i != 0:
            continue
        draw_obb_label(img_bgr, pts, label, color, font_scale, thickness)


def classify_img(pil: Image.Image) -> tuple[str, float]:
    feat = embed(pil)
    best_cls, best_sim = "", -1.0
    for cls, mean_e in ref_mean.items():
        s = float(torch.dot(feat, mean_e))
        if s > best_sim:
            best_cls, best_sim = cls, s
    return best_cls, best_sim


def match_text(text: str) -> str | None:
    low = text.lower().strip()
    for alias in sorted(TEXT_ALIASES, key=len, reverse=True):
        if alias in low:
            return TEXT_ALIASES[alias]
    for cls in CLASS_NAMES:
        if cls.lower().replace("_", " ") in low:
            return cls
    return None


def format_remedies(cls: str) -> str:
    lines = [f"### {DISPLAY[cls]}\n"]
    for i, r in enumerate(REMEDIES[cls], 1):
        lines.append(f"{i}. {r}")
    return "\n".join(lines)


# ── Inference ────────────────────────────────────────────────────────────────
def infer(image_input, text_input):
    # ── IMAGE PATH ────────────────────────────────────────────────────────
    if image_input is not None:
        pil = Image.fromarray(image_input).convert("RGB")
        W, H = pil.size
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        cls, sim = classify_img(pil)
        color = COLORS[cls]
        font_s = max(0.45, min(0.85, min(W, H) / 900))
        thick  = max(1, int(min(W, H) / 360))

        annotate_disease_regions(img_bgr, cls, sim, color, font_s, thick)

        out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        detail = (f"**Detected:** {DISPLAY[cls]}  \n"
                  f"**Confidence:** {sim*100:.1f}%\n\n"
                  + format_remedies(cls))
        return out, detail

    # ── TEXT PATH ─────────────────────────────────────────────────────────
    if text_input and text_input.strip():
        cls = match_text(text_input)
        if cls is None:
            return None, ("Could not match a disease. Try: leaf blight, leaf hoppers, "
                          "leaf rust, looper caterpillar, mosquito bug.")

        paths = ref_imgs.get(cls, [])
        if not paths:
            return None, f"No reference images for {DISPLAY[cls]}."

        img_path = random.choice(paths)
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        color   = COLORS[cls]
        font_s  = max(0.45, min(0.85, min(W, H) / 900))
        thick   = max(1, int(min(W, H) / 360))

        # For reference images from real dataset — use existing OBB labels if available
        stem = img_path.stem
        # strip "ref_" prefix if it's a Ref-Dataset image
        label_path = REAL_LABS / (stem.replace("ref_", "", 1) + ".txt")
        boxes = parse_yolo_obb(label_path, W, H) if label_path.exists() else []

        if boxes:
            for cid, pts in boxes:
                draw_obb_label(img_bgr, pts, DISPLAY[cls], color, font_s, thick)
        else:
            annotate_disease_regions(img_bgr, cls, 1.0, color, font_s, thick)

        out = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        detail = (f"**Disease:** {DISPLAY[cls]}  \n"
                  f"**Source:** Reference image\n\n"
                  + format_remedies(cls))
        return out, detail

    return None, "Upload an image **or** type a disease name below."


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Tea Leaf Disease Detector") as demo:
    gr.Markdown("# Tea Leaf Disease Detector\n"
                "Upload a leaf image **or** type a disease name to get OBB annotation + remedies.")
    with gr.Row():
        with gr.Column(scale=1):
            img_in  = gr.Image(label="Upload leaf image", type="numpy")
            txt_in  = gr.Textbox(label='Or type a disease name',
                                 placeholder="e.g. leaf rust / leaf blight / mosquito bug ...")
            run_btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=1):
            img_out = gr.Image(label="Annotated Output")
            txt_out = gr.Markdown()

    run_btn.click(fn=infer, inputs=[img_in, txt_in], outputs=[img_out, txt_out])
    img_in.change(fn=infer, inputs=[img_in, txt_in],  outputs=[img_out, txt_out])

    gr.Examples(
        examples=[[None, "leaf rust"], [None, "leaf blight"],
                  [None, "mosquito bug"], [None, "looper caterpillar"],
                  [None, "leaf hoppers"]],
        inputs=[img_in, txt_in], outputs=[img_out, txt_out],
        fn=infer, cache_examples=False, label="Quick examples",
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
