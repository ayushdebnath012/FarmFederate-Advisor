#!/usr/bin/env python3
# server.py — FastAPI backend serving trained FarmFederate models from models/ folder.
# Supports LLM (text-only), ViT (image-only), and VLM (multimodal) lightweight models
# trained by FarmFederate_Colab_Complete.py.

import os
import io
import json
import pathlib
import traceback
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from PIL import Image
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Fix PosixPath on Windows (models saved on Linux/Colab)
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# --- import ModelManager ---
try:
    from model_manager import ModelManager
    HAVE_MODEL_MANAGER = True
except ImportError:
    try:
        from .model_manager import ModelManager
        HAVE_MODEL_MANAGER = True
    except ImportError:
        HAVE_MODEL_MANAGER = False
        ModelManager = None

# ============================================================================
# LIGHTWEIGHT MODEL CLASSES (from FarmFederate_Colab_Complete.py)
# ============================================================================

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
NUM_LABELS = len(ISSUE_LABELS)


class LightweightTextClassifier(nn.Module):
    """Lightweight text classifier matching the Colab training architecture."""

    def __init__(self, vocab_size=30522, embed_dim=256, num_labels=5,
                 max_seq_len=128, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.pre_norm(x)
        mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.post_norm(x)
        logits = self.classifier(x)
        return {"logits": logits}


class LightweightVisionClassifier(nn.Module):
    """Lightweight vision classifier matching the Colab training architecture."""

    def __init__(self, num_labels=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.down1 = nn.Conv2d(64, 128, 1)
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.down2 = nn.Conv2d(128, 256, 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.down3 = nn.Conv2d(256, 512, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values, **kwargs):
        x = self.stem(pixel_values)
        x = self.block1(x) + self.down1(x)
        x = self.block2(x) + self.down2(x)
        x = self.block3(x) + self.down3(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return {"logits": logits}


class MultiModalClassifier(nn.Module):
    """VLM multimodal classifier with 8 fusion architectures."""

    def __init__(self, num_labels=5, fusion_type='concat',
                 text_dim=256, vision_dim=512, projection_dim=256, dropout=0.3):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_labels = num_labels
        self.text_dim = text_dim
        self.vision_dim = vision_dim

        # Text encoder
        self.text_embedding = nn.Embedding(30522, text_dim)
        self.text_encoder = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, dim_feedforward=text_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.text_dropout = nn.Dropout(dropout)

        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.vision_proj_initial = nn.Linear(256 * 7 * 7, vision_dim)
        self.vision_dropout = nn.Dropout(dropout)

        self._build_fusion_layers(fusion_type, text_dim, vision_dim, projection_dim, dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim), nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def _build_fusion_layers(self, fusion_type, text_dim, vision_dim, projection_dim, dropout=0.3):
        if fusion_type == 'concat':
            self.fusion_dim = text_dim + vision_dim
        elif fusion_type == 'attention':
            self.fusion_dim = text_dim
            self.cross_attention = nn.MultiheadAttention(text_dim, 4, dropout=dropout, batch_first=True)
            self.vision_proj = nn.Linear(vision_dim, text_dim)
        elif fusion_type == 'gated':
            self.fusion_dim = text_dim
            self.gate = nn.Sequential(nn.Linear(text_dim + vision_dim, text_dim), nn.Sigmoid())
            self.vision_proj = nn.Linear(vision_dim, text_dim)
        elif fusion_type == 'clip':
            self.fusion_dim = projection_dim * 2
            self.text_proj = nn.Sequential(nn.Linear(text_dim, projection_dim), nn.LayerNorm(projection_dim))
            self.vision_proj = nn.Sequential(nn.Linear(vision_dim, projection_dim), nn.LayerNorm(projection_dim))
        elif fusion_type == 'flamingo':
            self.fusion_dim = text_dim
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.perceiver_latents = nn.Parameter(torch.randn(32, text_dim))
            self.perceiver_attn = nn.MultiheadAttention(text_dim, 4, dropout=dropout, batch_first=True)
            self.gated_xattn = nn.MultiheadAttention(text_dim, 4, dropout=dropout, batch_first=True)
            self.xattn_gate = nn.Parameter(torch.tensor([0.1]))
        elif fusion_type == 'blip2':
            self.fusion_dim = text_dim
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.qformer_queries = nn.Parameter(torch.randn(16, text_dim) * 0.02)
            self.qformer_attn = nn.MultiheadAttention(text_dim, 4, dropout=dropout, batch_first=True)
            self.query_proj = nn.Linear(text_dim, text_dim)
        elif fusion_type == 'coca':
            self.fusion_dim = projection_dim * 2 + text_dim
            self.text_proj = nn.Sequential(nn.Linear(text_dim, projection_dim), nn.LayerNorm(projection_dim))
            self.vision_proj_contrastive = nn.Sequential(nn.Linear(vision_dim, projection_dim), nn.LayerNorm(projection_dim))
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.caption_xattn = nn.MultiheadAttention(text_dim, 4, dropout=dropout, batch_first=True)
        elif fusion_type == 'unified_io':
            self.fusion_dim = text_dim
            self.modality_embeddings = nn.Embedding(3, text_dim)
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.unified_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(text_dim, 4, text_dim * 4, dropout, batch_first=True), 2
            )
        else:
            self.fusion_dim = text_dim + vision_dim

    def encode_text(self, input_ids):
        x = self.text_embedding(input_ids)
        x = self.text_encoder(x)
        x = x.transpose(1, 2)
        x = self.text_pool(x).squeeze(-1)
        x = self.text_dropout(x)
        return x

    def encode_vision(self, pixel_values):
        x = self.vision_encoder(pixel_values)
        x = x.flatten(1)
        x = self.vision_proj_initial(x)
        x = self.vision_dropout(x)
        return x

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        text_features = self.encode_text(input_ids)

        if pixel_values is None:
            pixel_values = torch.zeros((input_ids.size(0), 3, 224, 224), device=input_ids.device)
        vision_features = self.encode_vision(pixel_values)

        if self.fusion_type == 'concat':
            fused = torch.cat([text_features, vision_features], dim=-1)
        elif self.fusion_type == 'attention':
            vision_proj = self.vision_proj(vision_features).unsqueeze(1)
            text_seq = text_features.unsqueeze(1)
            attn_out, _ = self.cross_attention(text_seq, vision_proj, vision_proj)
            fused = (text_features + attn_out.squeeze(1)) / 2
        elif self.fusion_type == 'gated':
            vision_proj = self.vision_proj(vision_features)
            gate = self.gate(torch.cat([text_features, vision_features], dim=-1))
            fused = text_features + gate * vision_proj
        elif self.fusion_type == 'clip':
            text_embeds = F.normalize(self.text_proj(text_features), dim=-1)
            vision_embeds = F.normalize(self.vision_proj(vision_features), dim=-1)
            fused = torch.cat([text_embeds, vision_embeds], dim=-1)
        elif self.fusion_type == 'flamingo':
            B = text_features.size(0)
            vision_proj = self.vision_proj(vision_features).unsqueeze(1).expand(-1, 49, -1)
            latents = self.perceiver_latents.unsqueeze(0).expand(B, -1, -1)
            attn_out, _ = self.perceiver_attn(latents, vision_proj, vision_proj)
            text_seq = text_features.unsqueeze(1)
            xattn_out, _ = self.gated_xattn(text_seq, attn_out, attn_out)
            fused = text_features + torch.tanh(self.xattn_gate) * xattn_out.squeeze(1)
        elif self.fusion_type == 'blip2':
            B = text_features.size(0)
            vision_proj = self.vision_proj(vision_features).unsqueeze(1).expand(-1, 49, -1)
            queries = self.qformer_queries.unsqueeze(0).expand(B, -1, -1)
            cross_out, _ = self.qformer_attn(queries, vision_proj, vision_proj)
            pooled = cross_out.mean(dim=1)
            fused = self.query_proj(pooled) + text_features
        elif self.fusion_type == 'coca':
            text_embeds = F.normalize(self.text_proj(text_features), dim=-1)
            vision_embeds = F.normalize(self.vision_proj_contrastive(vision_features), dim=-1)
            vision_proj = self.vision_proj(vision_features).unsqueeze(1).expand(-1, 49, -1)
            text_seq = text_features.unsqueeze(1)
            caption_out, _ = self.caption_xattn(text_seq, vision_proj, vision_proj)
            fused = torch.cat([text_embeds, vision_embeds, caption_out.squeeze(1)], dim=-1)
        elif self.fusion_type == 'unified_io':
            B = text_features.size(0)
            device = text_features.device
            text_token = self.modality_embeddings(torch.zeros(B, dtype=torch.long, device=device))
            vision_token = self.modality_embeddings(torch.ones(B, dtype=torch.long, device=device))
            fused_token = self.modality_embeddings(torch.full((B,), 2, dtype=torch.long, device=device))
            vision_proj = self.vision_proj(vision_features)
            sequence = torch.stack([fused_token, text_features + text_token, vision_proj + vision_token], dim=1)
            unified_out = self.unified_transformer(sequence)
            fused = unified_out[:, 0]
        else:
            fused = torch.cat([text_features, vision_features], dim=-1)

        logits = self.classifier(fused)
        return {"logits": logits}


# ============================================================================
# SIMPLE WORDPIECE TOKENIZER (no HuggingFace dependency for inference)
# ============================================================================

class SimpleTokenizer:
    """Minimal tokenizer that maps characters to WordPiece-compatible token IDs.
    Uses the same vocab_size=30522 as the training tokenizer.
    """
    def __init__(self, vocab_size=30522, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_id = 0
        self.unk_id = 100
        self.cls_id = 101
        self.sep_id = 102

    def __call__(self, text, max_length=None, truncation=True, padding="max_length",
                 return_tensors="pt"):
        max_len = max_length or self.max_length
        # Simple character-level hashing to token IDs
        tokens = [self.cls_id]
        for ch in text.lower():
            tid = (ord(ch) * 31 + 7) % (self.vocab_size - 200) + 200
            tokens.append(tid)
        tokens.append(self.sep_id)

        if truncation and len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + [self.sep_id]

        attention_mask = [1] * len(tokens)
        if padding == "max_length":
            pad_len = max_len - len(tokens)
            tokens += [self.pad_id] * pad_len
            attention_mask += [0] * pad_len

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": tokens, "attention_mask": attention_mask}


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = int(os.environ.get("MAX_LEN", 128))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
DEMO_MODE = str(os.environ.get("DEMO_MODE", "")).lower() in ("1", "true", "yes")

IMAGE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ADVICE = {
    "water_stress": "Irrigate earlier; mulch; monitor soil moisture AM/PM.",
    "nutrient_def": "Balance NPK (N focus if older leaves yellow); verify with LCC.",
    "pest_risk": "Inspect undersides; sticky traps; early biocontrol or mild soap.",
    "disease_risk": "Improve airflow; avoid late overhead irrigation; prune infected leaves.",
    "heat_stress": "Provide shade at peak heat; keep moisture stable; ensure K sufficiency.",
}

DISPLAY_NAMES = {
    "water_stress": "Water Stress",
    "nutrient_def": "Nutrient Deficiency",
    "pest_risk": "Pest Risk",
    "disease_risk": "Disease Risk",
    "heat_stress": "Heat Stress",
}

# Globals
MODEL: Optional[nn.Module] = None
MODEL_TYPE: str = "VLM"  # 'LLM', 'ViT', 'VLM'
TOKENIZER = SimpleTokenizer(max_length=MAX_LEN)
MODEL_MANAGER: Optional[Any] = None
CURRENT_MODEL_INFO: Optional[Dict[str, Any]] = None

print(f"[server] Device={DEVICE}, DEMO_MODE={DEMO_MODE}")


def advisor_from_mask(mask: List[int]) -> str:
    active = [ISSUE_LABELS[i] for i, v in enumerate(mask) if v == 1]
    if not active:
        return "Conditions look normal. Continue routine monitoring."
    return "Recommended actions:\n" + "\n".join(
        [f"- {DISPLAY_NAMES.get(lab, lab)}: {ADVICE.get(lab, '')}" for lab in active]
    )


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_checkpoint(path: str) -> dict:
    """Load a checkpoint and return the full dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"[server] Loading checkpoint: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def create_model_from_checkpoint(ck: dict) -> nn.Module:
    """Create the correct model architecture based on checkpoint metadata."""
    model_type = ck.get("model_type", "VLM")
    fusion_type = ck.get("fusion_type", "concat")

    if model_type in ("LLM", "text"):
        model = LightweightTextClassifier(num_labels=NUM_LABELS)
    elif model_type in ("ViT", "vision"):
        model = LightweightVisionClassifier(num_labels=NUM_LABELS)
    else:
        model = MultiModalClassifier(num_labels=NUM_LABELS, fusion_type=fusion_type)

    return model, model_type


def load_model(checkpoint_path: str):
    """Load a model from checkpoint file."""
    global MODEL, MODEL_TYPE

    ck = load_checkpoint(checkpoint_path)

    # Extract state dict
    state_dict = None
    if isinstance(ck, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ck:
                state_dict = ck[key]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in list(ck.values())[:3]):
            state_dict = ck
    if state_dict is None:
        raise RuntimeError("Could not find state_dict in checkpoint")

    model, model_type = create_model_from_checkpoint(ck)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[server] Loaded {model_type} model. missing={len(missing)} unexpected={len(unexpected)}")

    model.to(DEVICE)
    model.eval()
    MODEL = model
    MODEL_TYPE = model_type

    f1 = ck.get("f1_score")
    fusion = ck.get("fusion_type")
    print(f"[server] Model ready: type={model_type}, fusion={fusion}, F1={f1}, device={DEVICE}")


# ============================================================================
# LIFESPAN & APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_MANAGER, CURRENT_MODEL_INFO

    if DEMO_MODE:
        print("[server] DEMO_MODE active — no model loading")
        yield
        return

    try:
        if HAVE_MODEL_MANAGER and ModelManager is not None:
            MODEL_MANAGER = ModelManager(models_dir="../models")
            print(f"[server] ModelManager found {len(MODEL_MANAGER.models)} models")

            try:
                MODEL_MANAGER.export_model_catalog("model_catalog.json")
            except Exception:
                pass

            model_id = os.environ.get("MODEL_ID")
            if model_id:
                info = MODEL_MANAGER.get_model_info(model_id)
                if info:
                    load_model(info.path)
                    CURRENT_MODEL_INFO = info.to_dict()
                else:
                    print(f"[server][WARN] MODEL_ID '{model_id}' not found")
            else:
                # Load best VLM by default
                rec = MODEL_MANAGER.get_recommended_model("production")
                if rec:
                    print(f"[server] Loading recommended model: {rec.name} (F1={rec.f1_score})")
                    load_model(rec.path)
                    CURRENT_MODEL_INFO = rec.to_dict()
                else:
                    print("[server][WARN] No models found")
        else:
            print("[server][WARN] ModelManager not available")
    except Exception as e:
        print(f"[server][ERROR] Startup failed: {e}")
        traceback.print_exc()

    yield
    print("[server] Shutting down...")


app = FastAPI(title="FarmFederate Crop Stress API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# UTILITY
# ============================================================================

def preprocess_image(upload: UploadFile) -> Optional[torch.Tensor]:
    if upload is None:
        return None
    try:
        content = upload.file.read()
        upload.file.seek(0)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return IMAGE_TRANSFORM(img)
    except Exception as e:
        print(f"[server][WARN] Image preprocessing failed: {e}")
        return None


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": MODEL is not None,
        "model_type": MODEL_TYPE if MODEL else None,
        "labels": ISSUE_LABELS,
        "demo_mode": DEMO_MODE,
        "current_model": CURRENT_MODEL_INFO,
    }


@app.get("/sensors/latest")
async def get_latest_sensors():
    sensors_dir = os.path.join("checkpoints_paper", "ingest", "sensors")
    abs_dir = os.path.abspath(sensors_dir)
    if not os.path.exists(abs_dir):
        return JSONResponse({"error": "No sensor data available"}, status_code=404)
    try:
        files = [f for f in os.listdir(abs_dir) if f.endswith('.json')]
        if not files:
            return JSONResponse({"error": "No sensor data available"}, status_code=404)
        latest = max([os.path.join(abs_dir, f) for f in files], key=os.path.getmtime)
        with open(latest, 'r') as f:
            data = json.load(f)
        import re
        snake = {re.sub(r'(?<!^)(?=[A-Z])', '_', k).lower(): v for k, v in data.items()}
        return JSONResponse(snake)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(request: Request):
    # Demo mode
    if DEMO_MODE:
        return await _demo_predict(request)

    if MODEL is None:
        return JSONResponse({"error": "No model loaded"}, status_code=503)

    content_type = request.headers.get("content-type", "").lower()
    text = ""
    sensors = ""
    client_id = "unknown"
    image_tensor = None

    if "application/json" in content_type:
        data = await request.json()
        text = str(data.get("text", "") or "")
        sensors = str(data.get("sensors", "") or "")
        client_id = str(data.get("client_id", client_id) or client_id)
    elif "multipart" in content_type or "form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
        sensors = str(form.get("sensors", "") or "")
        client_id = str(form.get("client_id", client_id) or client_id)
        upload = form.get("image", None)
        if upload is not None:
            image_tensor = preprocess_image(upload)
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    if not text and not sensors and image_tensor is None:
        return JSONResponse({"error": "Provide at least text or an image."}, status_code=400)

    # Build combined text
    combined = text.strip() or "(no text)"
    if sensors:
        combined = f"SENSORS: {sensors}\nLOG: {combined}"

    # Tokenize
    enc = TOKENIZER(combined, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Image
    img_batch = None
    if image_tensor is not None:
        img_batch = image_tensor.unsqueeze(0).to(DEVICE)

    # Forward
    with torch.no_grad():
        try:
            if MODEL_TYPE in ("LLM", "text"):
                out = MODEL(input_ids=input_ids, attention_mask=attention_mask)
            elif MODEL_TYPE in ("ViT", "vision"):
                if img_batch is None:
                    img_batch = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
                out = MODEL(pixel_values=img_batch)
            else:  # VLM
                if img_batch is None:
                    img_batch = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
                out = MODEL(input_ids=input_ids, attention_mask=attention_mask, pixel_values=img_batch)

            logits = out["logits"] if isinstance(out, dict) else out
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": "Inference failed", "detail": str(e)}, status_code=500)

    # Convert to probabilities (softmax for multi-class)
    probs = F.softmax(logits, dim=-1).cpu().numpy().ravel().tolist()
    pred_class = int(np.argmax(probs))
    mask = [1 if i == pred_class else 0 for i in range(NUM_LABELS)]
    # Also flag high secondary probabilities
    for i, p in enumerate(probs):
        if p > 0.3:
            mask[i] = 1

    return JSONResponse({
        "client_id": client_id,
        "text_used": combined,
        "labels": ISSUE_LABELS,
        "probs": probs,
        "predicted_class": ISSUE_LABELS[pred_class],
        "predicted_display": DISPLAY_NAMES.get(ISSUE_LABELS[pred_class], ISSUE_LABELS[pred_class]),
        "confidence": float(probs[pred_class]),
        "mask": mask,
        "advice": advisor_from_mask(mask),
        "model_type": MODEL_TYPE,
        "model_info": CURRENT_MODEL_INFO,
    })


async def _demo_predict(request: Request):
    """Keyword-based demo predictions."""
    content_type = request.headers.get("content-type", "").lower()
    text = ""
    if "application/json" in content_type:
        data = await request.json()
        text = str(data.get("text", "") or "")
    elif "multipart" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")

    kw_map = {
        0: ["water", "dry", "wilt", "drought", "moisture"],
        1: ["yellow", "nutrient", "fertilizer", "pale", "deficiency"],
        2: ["bug", "insect", "pest", "hole", "eaten", "aphid"],
        3: ["spot", "disease", "fungus", "rot", "blight", "rust"],
        4: ["heat", "hot", "burn", "scorch", "temperature"],
    }
    text_lower = text.lower()
    probs = [0.1] * 5
    for i, kws in kw_map.items():
        matches = sum(1 for kw in kws if kw in text_lower)
        if matches > 0:
            probs[i] = min(0.5 + matches * 0.15, 0.95)
    if max(probs) <= 0.1:
        probs = [0.25, 0.12, 0.08, 0.10, 0.05]

    pred = int(np.argmax(probs))
    mask = [1 if p > 0.3 else 0 for p in probs]
    return JSONResponse({
        "labels": ISSUE_LABELS, "probs": probs, "mask": mask,
        "predicted_class": ISSUE_LABELS[pred],
        "confidence": float(probs[pred]),
        "advice": advisor_from_mask(mask), "demo_mode": True,
    })


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/models")
async def list_models():
    if not MODEL_MANAGER:
        return JSONResponse({"models": [], "current_model": CURRENT_MODEL_INFO})
    models = MODEL_MANAGER.list_models()
    recs = {}
    for uc in ("production", "privacy", "fast", "vision", "text"):
        r = MODEL_MANAGER.get_recommended_model(uc)
        recs[uc] = r.name if r else None
    return JSONResponse({"models": models, "recommendations": recs, "current_model": CURRENT_MODEL_INFO})


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    if not MODEL_MANAGER:
        return JSONResponse({"error": "ModelManager not available"}, status_code=400)
    info = MODEL_MANAGER.get_model_info(model_id)
    if not info:
        return JSONResponse({"error": f"Model '{model_id}' not found"}, status_code=404)
    return JSONResponse({"model": info.to_dict(), "is_current": CURRENT_MODEL_INFO and CURRENT_MODEL_INFO.get("name") == model_id})


@app.post("/models/switch/{model_id}")
async def switch_model(model_id: str):
    global CURRENT_MODEL_INFO
    if not MODEL_MANAGER:
        return JSONResponse({"error": "ModelManager not available"}, status_code=400)
    info = MODEL_MANAGER.get_model_info(model_id)
    if not info:
        return JSONResponse({"error": f"Model '{model_id}' not found"}, status_code=404)
    try:
        load_model(info.path)
        CURRENT_MODEL_INFO = info.to_dict()
        return JSONResponse({"success": True, "message": f"Switched to {model_id}", "model": CURRENT_MODEL_INFO})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/models/current")
async def get_current_model():
    return JSONResponse({"model": CURRENT_MODEL_INFO, "model_type": MODEL_TYPE if MODEL else None})


@app.post("/control/{device}")
async def control_device(device: str, request: Request):
    try:
        body = await request.json()
        state = body.get("state", False)
        return JSONResponse({
            "success": True, "device": device, "state": state,
            "message": f"{device} {'activated' if state else 'deactivated'}"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print(f"[server] Starting on port {os.environ.get('PORT', 8000)}")
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
