#!/usr/bin/env python3
# server.py — FastAPI backend to serve the full multimodal model checkpoint
# Place this file in backend/ and run with your venv active.

import os
import io
import json
import time
import traceback
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- try to import training helpers (priors, labels) if available ---
try:
    from farm_advisor import ISSUE_LABELS as TRAIN_ISSUE_LABELS, apply_priors_to_logits
    HAVE_TRAIN_HELPERS = True
    ISSUE_LABELS = list(TRAIN_ISSUE_LABELS)
    print("[server] Found farm_advisor helpers (absolute)")
except ImportError:
    try:
        from .farm_advisor import ISSUE_LABELS as TRAIN_ISSUE_LABELS, apply_priors_to_logits
        HAVE_TRAIN_HELPERS = True
        ISSUE_LABELS = list(TRAIN_ISSUE_LABELS)
        print("[server] Found farm_advisor helpers (relative)")
    except ImportError:
        pass
if not globals().get('HAVE_TRAIN_HELPERS'):
    HAVE_TRAIN_HELPERS = False
    ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
    def apply_priors_to_logits(logits: torch.Tensor, texts: Optional[List[str]]):
        # fallback: no-op (no priors)
        return logits
    print("[server] farm_advisor helpers NOT found — using fallback labels and no priors.")

NUM_LABELS = len(ISSUE_LABELS)

# --- import your multimodal model & tokenizer builders ---
# This file expects multimodal_model.py to define MultimodalClassifier,
# build_tokenizer() and (optionally) build_image_processor().
# Supports both relative imports (uvicorn backend.server:app) and
# absolute imports (cd backend && uvicorn server:app).
_DEMO_MODE_EARLY = str(os.environ.get("DEMO_MODE", "")).lower() in ("1", "true", "yes")
HAVE_MODEL_CLASSES = False
try:
    from .multimodal_model import MultimodalClassifier, build_tokenizer, build_image_processor
    HAVE_MODEL_CLASSES = True
    print("[server] Imported MultimodalClassifier (relative import)")
except ImportError:
    try:
        from multimodal_model import MultimodalClassifier, build_tokenizer, build_image_processor
        HAVE_MODEL_CLASSES = True
        print("[server] Imported MultimodalClassifier (absolute import)")
    except ImportError:
        try:
            from .multimodal_model import MultimodalModel as MultimodalClassifier
            from .multimodal_model import build_tokenizer, build_image_processor
            HAVE_MODEL_CLASSES = True
            print("[server] Imported MultimodalModel alias (relative)")
        except ImportError:
            try:
                from multimodal_model import MultimodalModel as MultimodalClassifier
                from multimodal_model import build_tokenizer, build_image_processor
                HAVE_MODEL_CLASSES = True
                print("[server] Imported MultimodalModel alias (absolute)")
            except ImportError:
                if _DEMO_MODE_EARLY:
                    print("[server][WARN] multimodal_model not available — DEMO_MODE active, continuing without model")
                    MultimodalClassifier = None
                    def build_tokenizer(*a, **kw): return None
                    def build_image_processor(*a, **kw): return None
                else:
                    print("[server][ERROR] Could not import multimodal_model definitions.")
                    traceback.print_exc()
                    raise

# --- import ModelManager for model selection ---
try:
    from model_manager import ModelManager
    print("[server] Imported ModelManager for model management")
    HAVE_MODEL_MANAGER = True
except ImportError:
    try:
        from .model_manager import ModelManager
        print("[server] Imported ModelManager (relative)")
        HAVE_MODEL_MANAGER = True
    except ImportError:
        print("[server][WARN] ModelManager not available - dynamic model switching disabled")
        HAVE_MODEL_MANAGER = False
        ModelManager = None

# ---------------- Configuration ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join("checkpoints", "global_central.pt"))
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "roberta-base")
IMAGE_MODEL_NAME = os.environ.get("IMAGE_MODEL_NAME", "google/vit-base-patch16-224-in21k")
MAX_LEN = int(os.environ.get("MAX_LEN", 160))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 224))
# Demo mode: when set, server returns canned lightweight responses (no heavy models needed)
DEMO_MODE = str(os.environ.get("DEMO_MODE", "")).lower() in ("1", "true", "yes")
print(f"[server] DEMO_MODE={DEMO_MODE}")

# simple image transforms compatible with ViT / most training pipelines
IMAGE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ---------------- Advice mapping (same as training) ----------------
ADVICE = {
    "water_stress": "Irrigate earlier; mulch; monitor soil moisture AM/PM.",
    "nutrient_def": "Balance NPK (N focus if older leaves yellow); verify with LCC.",
    "pest_risk": "Inspect undersides; sticky traps; early biocontrol or mild soap.",
    "disease_risk": "Improve airflow; avoid late overhead irrigation; prune infected leaves.",
    "heat_stress": "Provide shade at peak heat; keep moisture stable; ensure K sufficiency.",
}

def advisor_from_mask(mask: List[int]) -> str:
    active = [ISSUE_LABELS[i] for i,v in enumerate(mask) if v==1]
    if not active: 
        return "Conditions look normal. Continue routine monitoring."
    return "Recommended actions:\n" + "\n".join([f"- {lab}: {ADVICE.get(lab, '')}" for lab in active])

# ---------------- Globals ----------------
TOKENIZER = None
IMAGE_PROCESSOR = None
MODEL: Optional[nn.Module] = None
THRESHOLDS = np.array([0.3]*NUM_LABELS, dtype=np.float32)
MODEL_MANAGER: Optional[Any] = None  # ModelManager instance
CURRENT_MODEL_INFO: Optional[Dict[str, Any]] = None  # Current loaded model metadata



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize ModelManager and load model
    global MODEL_MANAGER, CURRENT_MODEL_INFO

    # In DEMO_MODE without model classes, skip model loading entirely
    if DEMO_MODE and not HAVE_MODEL_CLASSES:
        print("[server] DEMO_MODE active, no model classes — skipping model loading")
        yield
        print("[server] Shutting down...")
        return

    try:
        # Initialize ModelManager if available
        if HAVE_MODEL_MANAGER and ModelManager is not None:
            MODEL_MANAGER = ModelManager(models_dir="../models")
            print(f"[server] ModelManager initialized with {len(MODEL_MANAGER.models)} models")
            
            # Export model catalog for frontend
            try:
                MODEL_MANAGER.export_model_catalog("model_catalog.json")
            except Exception as e:
                print(f"[server][WARN] Failed to export model catalog: {e}")
            
            # Determine which model to load
            model_id = os.environ.get("MODEL_ID")
            if model_id:
                # Load specific model from environment variable
                model_info = MODEL_MANAGER.get_model_info(model_id)
                if model_info:
                    print(f"[server] Loading model from MODEL_ID env: {model_id}")
                    checkpoint_path = model_info.path
                    CURRENT_MODEL_INFO = model_info.to_dict()
                else:
                    print(f"[server][WARN] MODEL_ID '{model_id}' not found, using default")
                    checkpoint_path = CHECKPOINT_PATH
            elif CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
                # Use CHECKPOINT_PATH if it exists
                print(f"[server] Using CHECKPOINT_PATH: {CHECKPOINT_PATH}")
                checkpoint_path = CHECKPOINT_PATH
            else:
                # Use recommended federated VLM (privacy + performance balance)
                rec_model = MODEL_MANAGER.get_recommended_model("privacy")
                if rec_model:
                    print(f"[server] Loading recommended model: {rec_model.name}")
                    checkpoint_path = rec_model.path
                    CURRENT_MODEL_INFO = rec_model.to_dict()
                else:
                    print(f"[server][ERROR] No models available and no CHECKPOINT_PATH specified")
                    checkpoint_path = None
        else:
            # Fallback to CHECKPOINT_PATH when ModelManager not available
            checkpoint_path = CHECKPOINT_PATH
            print(f"[server] ModelManager not available, using CHECKPOINT_PATH: {checkpoint_path}")
        
        if checkpoint_path:
            load_model_and_tokenizer(checkpoint_path)
        else:
            print("[server][WARN] No model loaded - server running without inference capability")
            
    except Exception as e:
        print("[server][ERROR] Failed startup model load:", e)
        traceback.print_exc()



    yield
    # Shutdown: cleanup if needed
    print("[server] Shutting down...")

app = FastAPI(title="FarmFederate-Advisor (full model server)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Model loader ----------------
def safe_load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a checkpoint in various common formats and return a dict-like object.
    Accepts:
      - raw state_dict (saved via torch.save(model.state_dict()))
      - dict with key 'model_state_dict'
      - dict with 'state_dict' or similar
      - whole model (rare) -> will return its state_dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"[server] Loading checkpoint: {path}")
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict):
        # common keys
        for k in ("model_state_dict", "state_dict", "model"):
            if k in ck:
                val = ck[k]
                print(f"[server] Found checkpoint key '{k}' -> using it as state_dict")
                return val
        # maybe it's already the raw state_dict
        # check if values are tensors
        if all(isinstance(v, torch.Tensor) for v in ck.values()):
            print("[server] Checkpoint appears to be a raw state_dict")
            return ck
        # If it's a wrapper with nested 'model', try heuristics
        # pick largest tensor-like entry
        for v in ck.values():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                print("[server] Found nested state_dict inside checkpoint; using it")
                return v
    # if it's a nn.Module (rare)
    if hasattr(ck, "state_dict"):
        print("[server] Checkpoint is a model object; extracting state_dict()")
        return ck.state_dict()
    raise RuntimeError("Unknown checkpoint format")

def load_model_and_tokenizer(checkpoint_path: str = CHECKPOINT_PATH):
    global TOKENIZER, IMAGE_PROCESSOR, MODEL, THRESHOLDS

    # tokenizer from multimodal_model builder (keeps same tokenization as training)
    TOKENIZER = build_tokenizer(TEXT_MODEL_NAME)
    print("[server] Tokenizer loaded.")

    # create architecture identical to training
    # Note: Set use_cross_attention=False for compatibility with old checkpoints
    # Set to True after retraining with enhanced architecture
    MODEL = MultimodalClassifier(
        text_model_name=TEXT_MODEL_NAME,
        image_model_name=IMAGE_MODEL_NAME,
        num_labels=NUM_LABELS,
        freeze_backbones=False,  # inference only — no training
        use_cross_attention=False,  # TODO: Set to True after retraining
    )
    print("[server] Multimodal model instance created.")

    # try to load checkpoint
    try:
        state_dict = safe_load_checkpoint(checkpoint_path)
        # Ensure key name shapes compatible — allow strict=False
        missing, unexpected = MODEL.load_state_dict(state_dict, strict=False)
        print(f"[server] Loaded weights (strict=False). missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    except Exception as e:
        print("[server][WARN] Failed to load checkpoint with strict heuristics:", e)
        traceback.print_exc()
        # continue with random init (not ideal)
    
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"[server] Model moved to {DEVICE} and set to eval().")

    # optional: if you saved thresholds.npy near checkpoint, attempt to load
    thr_path = os.path.join(os.path.dirname(checkpoint_path), "thresholds.npy")
    if os.path.exists(thr_path):
        try:
            THRESHOLDS = np.load(thr_path)
            print(f"[server] Loaded thresholds from {thr_path}: {THRESHOLDS}")
        except Exception:
            print("[server][WARN] Failed to load thresholds.npy — using defaults 0.5.")

    # image processor: try to use builder if present, else fallback to torchvision transforms
    try:
        IMAGE_PROCESSOR = build_image_processor(IMAGE_MODEL_NAME)
        print("[server] Image processor loaded from multimodal_model (if defined).")
    except Exception:
        IMAGE_PROCESSOR = None
        print("[server] No image processor builder available — using torchvision transform fallback.")

# ---------------- utility helpers ----------------
def build_text_with_sensors(text: str, sensors: str) -> str:
    text = (text or "").strip()
    sensors = (sensors or "").strip()
    if sensors:
        sensors_line = sensors if sensors.upper().startswith("SENSORS:") else f"SENSORS: {sensors}"
    else:
        sensors_line = "SENSORS: (not provided)."
    if not text: text = "(no free-text log)."
    return f"{sensors_line}\nLOG: {text}"

def preprocess_image_upload(upload: UploadFile) -> Optional[torch.Tensor]:
    if upload is None:
        return None
    try:
        content = upload.file.read()
        upload.file.seek(0)
        img = Image.open(io.BytesIO(content)).convert("RGB")
        if IMAGE_PROCESSOR is not None:
            # if user provided AutoImageProcessor in multimodal_model, use it
            try:
                # AutoImageProcessor returns numpy or torch tensor depending on config
                proc = IMAGE_PROCESSOR(images=img, return_tensors="pt")
                # expected key may be 'pixel_values'
                if "pixel_values" in proc:
                    t = proc["pixel_values"]
                else:
                    # fallback: assume output is a torch tensor
                    t = torch.tensor(proc).permute(0,3,1,2) if isinstance(proc, np.ndarray) else proc
                return t.squeeze(0) if t.ndim==4 else t
            except Exception:
                pass
        # fallback torchvision transforms
        t = IMAGE_TRANSFORM(img)  # [C,H,W]
        return t
    except Exception as e:
        print("[server][WARN] Failed to preprocess image:", e)
        traceback.print_exc()
        return None

def logits_to_response(text: str, logits: torch.Tensor, thresholds: np.ndarray):
    # logits: [1, C] torch tensor on device
    # apply priors (in training module style) — priors expect (logits, [texts]) according to training helper
    try:
        logits = apply_priors_to_logits(logits, [text])
    except Exception:
        # fallback no-op
        pass
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
    thr = list(map(float, thresholds.tolist()))
    mask = [1 if p >= t else 0 for p,t in zip(probs, thr)]
    active_labels = []
    all_scores = []
    for i, lab in enumerate(ISSUE_LABELS):
        entry = {"label": lab, "prob": float(probs[i]), "threshold": float(thr[i])}
        all_scores.append(entry)
        if mask[i] == 1:
            active_labels.append(entry)
    advice = advisor_from_mask(mask)
    return {
        "active_labels": active_labels,
        "all_scores": all_scores,
        "raw_probs": probs,
        "advice": advice,
        "debug": {"probs": probs, "thresholds": thr, "mask": mask}
    }

# ----------------- Routes -----------------
@app.get("/health")
async def health():
    model_loaded = MODEL is not None
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": bool(model_loaded),
        "labels": ISSUE_LABELS,
        "demo_mode": DEMO_MODE,
    }

@app.get("/sensors/latest")
async def get_latest_sensors():
    """
    Return the latest sensor data from MQTT listener saved files.
    Looks in checkpoints_paper/ingest/sensors/ for *.json files and returns the most recent.
    """
    import os
    print("[DEBUG] Current working directory:", os.getcwd())
    sensors_dir = os.path.join("checkpoints_paper", "ingest", "sensors")
    abs_sensors_dir = os.path.abspath(sensors_dir)
    print("[DEBUG] Looking for sensor files in:", abs_sensors_dir)
    if not os.path.exists(abs_sensors_dir):
        print("[DEBUG] Directory does not exist.")
        return JSONResponse({"error": "No sensor data available"}, status_code=404)

    # Find all sensor JSON files
    try:
        files = [f for f in os.listdir(abs_sensors_dir) if f.endswith('.json')]
        print("[DEBUG] Found files:", files)
        if not files:
            print("[DEBUG] No .json files found.")
            return JSONResponse({"error": "No sensor data available"}, status_code=404)

        # Get the most recently modified file
        latest_file = max([os.path.join(abs_sensors_dir, f) for f in files], key=os.path.getmtime)
        print("[DEBUG] Latest file:", latest_file)

        with open(latest_file, 'r') as f:
            sensor_data = json.load(f)

        # Convert camelCase keys to snake_case for frontend compatibility
        def to_snake_case(s):
            import re
            return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

        snake_case_data = {}
        for k, v in sensor_data.items():
            snake_case_data[to_snake_case(k)] = v

        return JSONResponse(snake_case_data, status_code=200)
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        return JSONResponse({"error": f"Failed to read sensor data: {str(e)}"}, status_code=500)

@app.post("/predict")
async def predict(request: Request):
    """
    Accepts JSON:
      { "text": "...", "sensors": "soil_moisture=...", "client_id": "..." }
    Or multipart/form-data:
      text: str, sensors: str (optional), image: file (optional)
    """
    global MODEL, TOKENIZER, THRESHOLDS

    # Demo mode: return simulated predictions without loading ML models
    if DEMO_MODE:
        content_type = request.headers.get("content-type", "").lower()
        text = ""
        if "application/json" in content_type:
            data = await request.json()
            text = str(data.get("text", "") or "")
        elif "multipart/form-data" in content_type:
            form = await request.form()
            text = str(form.get("text", "") or "")

        # Simple keyword-based demo predictions
        text_lower = text.lower()
        probs = [0.1, 0.1, 0.1, 0.1, 0.1]
        if "water" in text_lower or "dry" in text_lower or "wilt" in text_lower:
            probs[0] = 0.85  # water_stress
        elif "yellow" in text_lower or "nutrient" in text_lower or "fertilizer" in text_lower:
            probs[1] = 0.82  # nutrient_def
        elif "bug" in text_lower or "insect" in text_lower or "pest" in text_lower or "hole" in text_lower:
            probs[2] = 0.78  # pest_risk
        elif "spot" in text_lower or "disease" in text_lower or "fungus" in text_lower or "rot" in text_lower:
            probs[3] = 0.88  # disease_risk
        elif "heat" in text_lower or "hot" in text_lower or "burn" in text_lower:
            probs[4] = 0.75  # heat_stress
        else:
            # Default to healthy-ish with slight water stress
            probs = [0.3, 0.15, 0.1, 0.1, 0.05]

        mask = [1 if p > 0.3 else 0 for p in probs]
        return JSONResponse({
            "labels": ISSUE_LABELS,
            "probs": probs,
            "mask": mask,
            "advice": advisor_from_mask(mask),
            "demo_mode": True,
        })

    if MODEL is None or TOKENIZER is None:
        # Attempt to reload if startup failed earlier
        try:
            load_model_and_tokenizer(CHECKPOINT_PATH)
        except Exception:
            return JSONResponse({"error": "Model not available on server"}, status_code=503)

    content_type = request.headers.get("content-type", "").lower()
    text = ""
    sensors = ""
    client_id = "unknown"
    image_tensor = None

    # JSON
    if "application/json" in content_type:
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=400)
        text = str(data.get("text", "") or "")
        sensors = str(data.get("sensors", "") or "")
        client_id = str(data.get("client_id", client_id) or client_id)

    # multipart/form-data
    elif "multipart/form-data" in content_type or "form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
        sensors = str(form.get("sensors", "") or "")
        client_id = str(form.get("client_id", client_id) or client_id)
        upload = form.get("image", None)
        if upload is not None:
            # upload is starlette UploadFile
            image_tensor = preprocess_image_upload(upload)

    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    if not text and not sensors and image_tensor is None:
        return JSONResponse({"error": "Provide at least text or an image."}, status_code=400)

    combined_text = build_text_with_sensors(text, sensors)

    # Tokenize text
    enc = TOKENIZER(
        combined_text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # prepare image batch if available
    if image_tensor is not None:
        # ensure batch dim
        if isinstance(image_tensor, torch.Tensor):
            img_batch = image_tensor.unsqueeze(0).to(DEVICE)
        else:
            # if it's numpy etc.
            img_batch = torch.tensor(image_tensor).unsqueeze(0).to(DEVICE)
    else:
        # If your multimodal model **requires** image always, send a zero tensor
        img_batch = None

    # Forward pass (no grad)
    with torch.no_grad():
        try:
            # Match the forward signature from multimodal_model.MultimodalClassifier:
            # def forward(self, input_ids, attention_mask, pixel_values)
            if img_batch is not None:
                logits = MODEL(input_ids=input_ids, attention_mask=attention_mask, pixel_values=img_batch)
            else:
                # If model expects pixel_values (non-optional) — create zero image tensor with correct feature size
                # We attempt call without pixel_values first
                try:
                    out_logits = MODEL(input_ids=input_ids, attention_mask=attention_mask, pixel_values=None)
                    logits = out_logits
                except TypeError:
                    # fallback: zero image
                    dummy_img = torch.zeros((1,3,IMG_SIZE,IMG_SIZE), device=DEVICE)
                    logits = MODEL(input_ids=input_ids, attention_mask=attention_mask, pixel_values=dummy_img)
            # If model returns a tensor directly (logits) or an object with .logits
            if isinstance(logits, dict) and "logits" in logits:
                logits = logits["logits"]
            elif hasattr(logits, "logits"):
                logits = logits.logits
            # ensure shape [1,C]
            if isinstance(logits, torch.Tensor) and logits.dim()==1:
                logits = logits.unsqueeze(0)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": "model inference failed", "detail": str(e)}, status_code=500)

    result = logits_to_response(combined_text, logits, THRESHOLDS)
    # Add client_id & text used for debugging
    out = {
        "client_id": client_id,
        "text_used": combined_text,
        "result": result,
        "advice": result.get("advice", "")
    }
    return JSONResponse(out, status_code=200)

# ---------------- Model Management Endpoints ----------------
@app.get("/models")
async def list_models():
    """
    List all available trained models with metadata
    """
    if not HAVE_MODEL_MANAGER or MODEL_MANAGER is None:
        return JSONResponse({
            "error": "ModelManager not available",
            "current_model": CHECKPOINT_PATH if CHECKPOINT_PATH else None
        }, status_code=200)
    
    try:
        models = MODEL_MANAGER.list_models()
        recommendations = {
            "production": MODEL_MANAGER.get_recommended_model("production").name if MODEL_MANAGER.get_recommended_model("production") else None,
            "privacy": MODEL_MANAGER.get_recommended_model("privacy").name if MODEL_MANAGER.get_recommended_model("privacy") else None,
            "fast": MODEL_MANAGER.get_recommended_model("fast").name if MODEL_MANAGER.get_recommended_model("fast") else None,
            "vision": MODEL_MANAGER.get_recommended_model("vision").name if MODEL_MANAGER.get_recommended_model("vision") else None,
            "text": MODEL_MANAGER.get_recommended_model("text").name if MODEL_MANAGER.get_recommended_model("text") else None,
        }
        
        return JSONResponse({
            "models": models,
            "recommendations": recommendations,
            "current_model": CURRENT_MODEL_INFO
        }, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get detailed information about a specific model
    """
    if not HAVE_MODEL_MANAGER or MODEL_MANAGER is None:
        return JSONResponse({"error": "ModelManager not available"}, status_code=400)
    
    try:
        model_info = MODEL_MANAGER.get_model_info(model_id)
        if model_info:
            return JSONResponse({
                "model": model_info.to_dict(),
                "is_current": CURRENT_MODEL_INFO and CURRENT_MODEL_INFO.get("name") == model_id
            }, status_code=200)
        else:
            return JSONResponse({"error": f"Model '{model_id}' not found"}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/models/switch/{model_id}")
async def switch_model(model_id: str):
    """
    Switch to a different trained model
    """
    global CURRENT_MODEL_INFO
    
    if not HAVE_MODEL_MANAGER or MODEL_MANAGER is None:
        return JSONResponse({"error": "ModelManager not available - model switching disabled"}, status_code=400)
    
    try:
        model_info = MODEL_MANAGER.get_model_info(model_id)
        if not model_info:
            return JSONResponse({"error": f"Model '{model_id}' not found"}, status_code=404)
        
        # Check if model file exists
        if not os.path.exists(model_info.path):
            return JSONResponse({
                "error": f"Model file not found: {model_info.path}"
            }, status_code=404)
        
        # Load the new model
        print(f"[server] Switching to model: {model_id}")
        try:
            load_model_and_tokenizer(model_info.path)
            CURRENT_MODEL_INFO = model_info.to_dict()
            return JSONResponse({
                "success": True,
                "message": f"Successfully switched to model: {model_id}",
                "model": CURRENT_MODEL_INFO
            }, status_code=200)
        except Exception as load_error:
            traceback.print_exc()
            return JSONResponse({
                "error": f"Failed to load model: {str(load_error)}",
                "detail": traceback.format_exc()
            }, status_code=500)
            
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/models/current")
async def get_current_model():
    """
    Get information about the currently loaded model
    """
    if CURRENT_MODEL_INFO:
        return JSONResponse({
            "model": CURRENT_MODEL_INFO,
            "checkpoint_path": CHECKPOINT_PATH
        }, status_code=200)
    else:
        return JSONResponse({
            "model": None,
            "checkpoint_path": CHECKPOINT_PATH,
            "message": "No model metadata available (loaded from CHECKPOINT_PATH)"
        }, status_code=200)

# ---------------- Control endpoint for IoT devices ----------------
@app.post("/control/{device}")
async def control_device(device: str, request: Request):
    """
    Control endpoint for IoT devices (water pump, heater, pest control, etc.)
    """
    try:
        body = await request.json()
        state = body.get("state", False)
        
        # Log the control command
        print(f"[control] Device: {device}, State: {state}")
        
        # Here you would integrate with MQTT or other IoT protocol
        # For now, just return success
        return JSONResponse({
            "success": True,
            "device": device,
            "state": state,
            "message": f"{device} {'activated' if state else 'deactivated'}"
        }, status_code=200)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ---------------- Entry point for direct run ----------------
if __name__ == "__main__":
    import uvicorn
    print(f"[server] Starting uvicorn (device={DEVICE}). If you want to change port, set PORT env var.")
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
