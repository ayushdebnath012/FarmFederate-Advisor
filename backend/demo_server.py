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


# --- Knowledge base for RAG (demo mode) ---
KB_DOCS = [
    {"id": "ws_01", "stress": "water_stress", "text": "Water stress occurs when crops cannot access sufficient soil moisture. Symptoms include wilting, leaf roll, and stomatal closure. Immediate irrigation is critical; drip systems reduce evaporation losses by up to 40%."},
    {"id": "ws_02", "stress": "water_stress", "text": "Drought-stressed plants show reduced photosynthesis and early senescence. Mulching with organic material 5–10 cm deep retains soil moisture and reduces irrigation frequency by 25–30%."},
    {"id": "ws_03", "stress": "water_stress", "text": "Wilting and drooping leaves in the afternoon that recover overnight indicate moderate water stress. Apply 25–30 mm of water per week; use soil moisture sensors to avoid overwatering."},
    {"id": "nd_01", "stress": "nutrient_def", "text": "Nitrogen deficiency causes yellowing (chlorosis) of older leaves first, stunted growth, and reduced yield. Apply urea (46-0-0) at 50–100 kg/ha or top-dress with ammonium nitrate."},
    {"id": "nd_02", "stress": "nutrient_def", "text": "Pale green to yellow leaves across the whole plant indicate low N or S. A complete NPK foliar spray (20-20-20) at 2 g/L provides immediate relief within 5–7 days."},
    {"id": "nd_03", "stress": "nutrient_def", "text": "Iron chlorosis shows as yellowing between leaf veins on young leaves. Apply chelated iron (Fe-EDTA) as foliar spray or soil drench at pH < 7. Avoid alkaline soils without acidification."},
    {"id": "pr_01", "stress": "pest_risk", "text": "Aphid infestations cause leaf curl, sticky honeydew deposits, and sooty mold. Introduce natural predators (ladybirds, lacewings) or apply insecticidal soap solution at 2% concentration."},
    {"id": "pr_02", "stress": "pest_risk", "text": "Chewed leaf margins and frass deposits indicate caterpillar or beetle damage. Hand-pick egg masses, apply Bacillus thuringiensis (Bt) spray, and use pheromone traps to monitor adult populations."},
    {"id": "pr_03", "stress": "pest_risk", "text": "Spider mite damage appears as fine stippling on leaves with webbing. Maintain humidity above 60%, apply acaricide (abamectin) at label rate, and remove severely affected leaves."},
    {"id": "dr_01", "stress": "disease_risk", "text": "Fungal blight and leaf spot diseases spread rapidly in warm, humid conditions. Remove infected leaves, improve canopy airflow, and apply copper-based fungicide every 7–10 days preventively."},
    {"id": "dr_02", "stress": "disease_risk", "text": "Brown or black lesions with yellow halos indicate bacterial or fungal disease. Avoid overhead irrigation, apply mancozeb or chlorothalonil at 2 g/L, and rotate crops to break the disease cycle."},
    {"id": "dr_03", "stress": "disease_risk", "text": "Powdery mildew presents as white powdery patches on leaves. Apply sulfur-based fungicide or potassium bicarbonate spray; ensure plants are not over-fertilized with nitrogen."},
    {"id": "hs_01", "stress": "heat_stress", "text": "Heat stress above 35°C causes pollen sterility, flower drop, and fruit blistering. Apply shade nets (30–50% shading) during peak hours, and foliar-spray calcium to strengthen cell walls."},
    {"id": "hs_02", "stress": "heat_stress", "text": "Leaf scorch and crispy margins occur when temperatures exceed evaporative cooling capacity. Irrigate in early morning, apply kaolin clay particle film to reflect solar radiation, and ensure adequate potassium."},
    {"id": "hs_03", "stress": "heat_stress", "text": "Sunscald on fruits causes bleached or white patches. Maintain consistent soil moisture, use reflective mulch, and avoid excessive nitrogen that promotes tender tissues vulnerable to heat."},
]


def _retrieve(query: str, top_k: int = 5) -> list:
    """Keyword-based retrieval from KB_DOCS (demo mode, no FAISS)."""
    query_lower = query.lower()
    scored = []
    for doc in KB_DOCS:
        # Count overlapping words (simple BM25-style proxy)
        doc_words = set(doc["text"].lower().split() + [doc["stress"].replace("_", " ")])
        query_words = set(query_lower.split())
        overlap = len(doc_words & query_words)
        # Boost by matching stress class keywords
        stress_kws = KEYWORD_MAP.get(doc["stress"], [])
        stress_hits = sum(1 for kw in stress_kws if kw in query_lower)
        score = overlap * 0.1 + stress_hits * 0.3
        if score > 0:
            scored.append({"id": doc["id"], "text": doc["text"], "stress_class": doc["stress"], "score": round(score, 3)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _rag_advisory(description: str, top_k: int = 5) -> dict:
    """Build RAG advisory from knowledge retrieval + prediction."""
    # Classify
    pred = predict_from_text(description if description.strip() else "general crop check")
    active = [s["label"] for s in pred["active_labels"]]
    top_class = max(pred["all_scores"], key=lambda x: x["prob"])["label"] if pred["all_scores"] else "disease_risk"

    # Build enriched query: description + top predicted class
    query = f"{description} {top_class.replace('_', ' ')} {' '.join(active)}"
    retrieved = _retrieve(query, top_k=top_k)

    # Assemble context from retrieved docs
    context_parts = [f"[{i+1}] {doc['text']}" for i, doc in enumerate(retrieved)]
    context = "\n".join(context_parts) if context_parts else "No specific guidance retrieved."

    # Construct prompt
    stress_str = ", ".join(active) if active else "general monitoring"
    prompt = (
        f"Crop stress diagnosis: {stress_str}\n\n"
        f"Retrieved agronomic guidance:\n{context}\n\n"
        f"Based on the diagnosis and retrieved guidance, provide a treatment recommendation."
    )

    # Generate offline treatment advisory
    if not active:
        treatment = "No active stress detected. Continue routine monitoring every 3–5 days."
    else:
        lines = [f"Detected stress: {', '.join(DISPLAY_NAMES.get(a, a) for a in active)}.\n"]
        for doc in retrieved[:3]:
            lines.append(f"• {doc['text']}")
        lines.append("\nMonitor crop response over 3–5 days and consult a local agronomist if symptoms persist.")
        treatment = "\n".join(lines)

    return {
        "retrieved": retrieved,
        "prompt": prompt,
        "treatment": treatment,
        "active_labels": pred["active_labels"],
        "all_scores": pred["all_scores"],
        "kb_docs_searched": len(KB_DOCS),
        "recall_at_k": round(len(retrieved) / max(len(KB_DOCS), 1), 3),
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
        "endpoints": ["/predict", "/rag", "/rag/metrics", "/models", "/sensors/latest", "/health"],
        "rag": {"kb_size": len(KB_DOCS), "retrieval_mode": "keyword_bm25_proxy"},
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


@app.post("/rag")
async def rag_advisory(request: Request):
    """Federated RAG advisory endpoint: retrieves from knowledge base and generates treatment."""
    content_type = request.headers.get("content-type", "").lower()
    description = ""

    if "application/json" in content_type:
        data = await request.json()
        description = str(data.get("description", "") or data.get("text", "") or "")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        description = str(form.get("description", "") or form.get("text", "") or "")
    else:
        return JSONResponse({"error": f"Unsupported Content-Type: {content_type}"}, status_code=415)

    result = _rag_advisory(description, top_k=5)
    return JSONResponse({
        "client_id": "demo",
        "description_used": description,
        "result": result,
        "demo_mode": True,
        "rag_mode": "offline_keyword_retrieval",
    })


@app.get("/rag/metrics")
async def rag_metrics():
    """Return the RAG evaluation metrics from the last training run."""
    import os
    metrics_path = os.path.join(os.path.dirname(__file__), "RAG", "rag_results (1).json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            import json as _json
            return JSONResponse(_json.load(f))
    return JSONResponse({
        "classification": {"macro_f1": 1.0, "micro_f1": 1.0},
        "retrieval": {"recall_at_5": 0.129, "mrr": 0.100, "ndcg_at_5": 0.172, "kb_coverage": 1.0},
        "robustness": {"embedding_drift": 0.156},
        "config": {"num_farms": 3, "fed_rounds": 5, "rag_rounds": 10, "top_k": 5},
        "demo_mode": True,
    })


@app.get("/models")
async def list_models():
    return JSONResponse({
        "models": [
            {"id": "centralized_vlm", "name": "Centralized VLM (RoBERTa + Swin + BLIP-2)", "type": "vlm", "accuracy": 0.874},
            {"id": "federated_vlm", "name": "Federated VLM (Privacy-Preserving)", "type": "vlm", "accuracy": 0.798},
            {"id": "federated_rag", "name": "Federated RAG Advisory (DistilRoBERTa + FAISS)", "type": "rag", "accuracy": 1.0, "kb_coverage": 1.0, "recall_at_5": 0.129},
            {"id": "centralized_llm", "name": "Centralized LLM (RoBERTa)", "type": "llm", "accuracy": 0.72},
            {"id": "centralized_vit", "name": "Centralized ViT (Swin)", "type": "vit", "accuracy": 0.68},
        ],
        "recommendations": {
            "production": "Centralized VLM (RoBERTa + Swin + BLIP-2)",
            "privacy": "Federated VLM (Privacy-Preserving)",
            "advisory": "Federated RAG Advisory (DistilRoBERTa + FAISS)",
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
