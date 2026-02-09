#!/usr/bin/env python3
"""
demo_server.py — Lightweight FastAPI backend for Render free tier (512MB).
No PyTorch/transformers imports. Uses keyword-based crop stress predictions.
For full ML inference, use server.py with local GPU/CPU.
"""

import os
import json
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Labels and advice (same as training) ---
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

ADVICE = {
    "water_stress": "Irrigate earlier; mulch around plants; monitor soil moisture morning and evening.",
    "nutrient_def": "Balance NPK (nitrogen focus if older leaves yellow); verify with leaf color chart.",
    "pest_risk": "Inspect leaf undersides; use sticky traps; try biocontrol or mild soap spray.",
    "disease_risk": "Improve airflow; avoid late overhead irrigation; prune infected leaves immediately.",
    "heat_stress": "Provide shade at peak heat; keep moisture stable; ensure potassium sufficiency.",
}

DISPLAY_NAMES = {
    "water_stress": "Needs Water",
    "nutrient_def": "Needs Fertilizer",
    "pest_risk": "Bug Problem",
    "disease_risk": "May Be Sick",
    "heat_stress": "Heat Stress",
}

# --- Keyword matching for demo predictions ---
KEYWORD_MAP = {
    "water_stress": ["water", "dry", "wilt", "wilting", "drooping", "thirsty", "drought", "irrigation", "moisture"],
    "nutrient_def": ["yellow", "nutrient", "fertilizer", "pale", "stunted", "deficiency", "nitrogen", "npk", "growth"],
    "pest_risk": ["bug", "insect", "pest", "hole", "eaten", "chewed", "aphid", "caterpillar", "worm", "mite", "fly"],
    "disease_risk": ["spot", "disease", "fungus", "rot", "mold", "blight", "rust", "lesion", "brown", "black", "sick", "infected", "red"],
    "heat_stress": ["heat", "hot", "burn", "scorch", "sunburn", "temperature", "crispy", "curl"],
}


def predict_from_text(text: str) -> dict:
    """Keyword-based prediction for demo mode."""
    text_lower = text.lower().strip()
    probs = [0.1] * len(ISSUE_LABELS)

    # Score each label based on keyword matches
    for i, label in enumerate(ISSUE_LABELS):
        keywords = KEYWORD_MAP.get(label, [])
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            probs[i] = min(0.5 + matches * 0.15, 0.95)

    # If no keywords matched, give a balanced "mostly healthy" response
    if max(probs) <= 0.1:
        probs = [0.25, 0.12, 0.08, 0.10, 0.05]

    mask = [1 if p > 0.3 else 0 for p in probs]

    active_labels = []
    all_scores = []
    for i, label in enumerate(ISSUE_LABELS):
        entry = {"label": label, "prob": probs[i], "threshold": 0.3}
        all_scores.append(entry)
        if mask[i] == 1:
            active_labels.append(entry)

    # Build advice
    active = [ISSUE_LABELS[i] for i, v in enumerate(mask) if v == 1]
    if not active:
        advice = "Conditions look normal. Continue routine monitoring."
    else:
        advice = "Recommended actions:\n" + "\n".join(
            [f"- {DISPLAY_NAMES.get(lab, lab)}: {ADVICE.get(lab, '')}" for lab in active]
        )

    return {
        "active_labels": active_labels,
        "all_scores": all_scores,
        "raw_probs": probs,
        "advice": advice,
    }


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[demo_server] Starting in DEMO_MODE (lightweight, no ML models)")
    yield
    print("[demo_server] Shutting down...")


app = FastAPI(title="FarmFederate-Advisor (Demo Mode)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": "cpu",
        "model_loaded": False,
        "labels": ISSUE_LABELS,
        "demo_mode": True,
    }


@app.get("/sensors/latest")
async def get_latest_sensors():
    return JSONResponse({"error": "No sensor data available (demo mode)"}, status_code=404)


@app.post("/predict")
async def predict(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    text = ""

    if "application/json" in content_type:
        data = await request.json()
        text = str(data.get("text", "") or "")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        text = str(form.get("text", "") or "")
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    if not text.strip():
        text = "general crop check"

    result = predict_from_text(text)
    return JSONResponse({
        "client_id": "demo",
        "text_used": text,
        "result": result,
        "advice": result["advice"],
        "demo_mode": True,
    })


@app.get("/models")
async def list_models():
    return JSONResponse({
        "models": [
            {"id": "centralized_vlm", "name": "Centralized VLM (RoBERTa + Swin + BLIP-2)", "type": "vlm", "accuracy": 0.874},
            {"id": "federated_vlm", "name": "Federated VLM (Privacy-Preserving)", "type": "vlm", "accuracy": 0.798},
            {"id": "centralized_llm", "name": "Centralized LLM (RoBERTa)", "type": "llm", "accuracy": 0.72},
            {"id": "centralized_vit", "name": "Centralized ViT (Swin)", "type": "vit", "accuracy": 0.68},
        ],
        "recommendations": {
            "production": "Centralized VLM (RoBERTa + Swin + BLIP-2)",
            "privacy": "Federated VLM (Privacy-Preserving)",
        },
        "current_model": {"name": "Demo Mode (keyword-based)", "type": "demo"},
        "demo_mode": True,
    })


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    return JSONResponse({
        "model": {"id": model_id, "name": model_id, "type": "demo"},
        "is_current": False,
        "demo_mode": True,
    })


@app.get("/models/current")
async def get_current_model():
    return JSONResponse({
        "model": {"name": "Demo Mode", "type": "demo"},
        "demo_mode": True,
    })


@app.post("/models/switch/{model_id}")
async def switch_model(model_id: str):
    return JSONResponse({
        "success": False,
        "message": "Model switching not available in demo mode. Run the full server locally with your models.",
        "demo_mode": True,
    })


@app.post("/control/{device}")
async def control_device(device: str, request: Request):
    try:
        body = await request.json()
        state = body.get("state", False)
        return JSONResponse({
            "success": True,
            "device": device,
            "state": state,
            "message": f"{device} {'activated' if state else 'deactivated'} (demo)",
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
