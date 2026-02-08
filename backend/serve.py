#!/usr/bin/env python3
"""
FarmFederate REST API Server

FastAPI-based production server for crop stress detection inference.

Usage:
    python serve.py --port 8000
    uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4
"""

import argparse
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine = None


# ============================================================================
# Pydantic Models for API
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for text-only prediction."""
    text: str = Field(..., description="Text description of crop symptoms", min_length=1)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str = Field(..., description="Predicted stress type")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    description: str = Field(..., description="Human-readable description")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    high_confidence: bool = Field(..., description="Whether confidence exceeds threshold")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch text predictions."""
    items: List[str] = Field(..., description="List of text descriptions")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[PredictionResponse]
    total_items: int
    successful: int
    failed: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    device: str
    uptime_seconds: float


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global inference_engine
    start_time = time.time()

    # Load model on startup
    model_path = os.getenv('MODEL_PATH', 'models/best_model.pt')
    device = os.getenv('DEVICE', None)
    threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

    try:
        from inference import CropStressInference
        inference_engine = CropStressInference(
            model_path=model_path,
            device=device,
            confidence_threshold=threshold,
        )
        logger.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.warning(f"Model not found at {model_path}, running in demo mode")
        inference_engine = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        inference_engine = None

    app.state.start_time = start_time
    yield

    # Cleanup on shutdown
    logger.info("Shutting down FarmFederate API server")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="FarmFederate Crop Stress Detection API",
    description="""
    AI-powered crop stress detection using multimodal deep learning.

    This API provides endpoints for:
    - Single image/text prediction
    - Batch predictions
    - Health monitoring

    Supported stress types:
    - **water_stress**: Drought or overwatering symptoms
    - **nutrient_def**: Nutrient deficiency indicators
    - **pest_risk**: Pest damage or infestation signs
    - **disease_risk**: Fungal, bacterial, or viral infections
    - **heat_stress**: Thermal damage or sun scorch
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FarmFederate Crop Stress Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    uptime = time.time() - getattr(app.state, 'start_time', time.time())
    return HealthResponse(
        status="healthy" if inference_engine else "degraded",
        model_loaded=inference_engine is not None,
        model_type=getattr(inference_engine, 'model_type', None) if inference_engine else None,
        device=getattr(inference_engine, 'device', 'unknown') if inference_engine else 'none',
        uptime_seconds=uptime,
    )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: PredictionRequest):
    """Predict crop stress from text description only."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    try:
        result = inference_engine.predict(text=request.text)
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            description=result['description'],
            all_probabilities=result['all_probabilities'],
            recommendations=result['recommendations'],
            high_confidence=result['high_confidence'],
            processing_time_ms=processing_time,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    image: UploadFile = File(..., description="Crop image file (JPG/PNG)"),
    text: Optional[str] = Form(None, description="Optional text description"),
):
    """Predict crop stress from image (optionally with text)."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image file type")

    start_time = time.time()
    try:
        # Read image data
        image_data = await image.read()
        from PIL import Image
        pil_image = Image.open(io.BytesIO(image_data))

        result = inference_engine.predict(image=pil_image, text=text)
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            description=result['description'],
            all_probabilities=result['all_probabilities'],
            recommendations=result['recommendations'],
            high_confidence=result['high_confidence'],
            processing_time_ms=processing_time,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple text descriptions."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    for text in request.items:
        item_start = time.time()
        try:
            result = inference_engine.predict(text=text)
            item_time = (time.time() - item_start) * 1000
            results.append(PredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                description=result['description'],
                all_probabilities=result['all_probabilities'],
                recommendations=result['recommendations'],
                high_confidence=result['high_confidence'],
                processing_time_ms=item_time,
            ))
            successful += 1
        except Exception as e:
            failed += 1
            logger.warning(f"Batch item failed: {e}")

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        results=results,
        total_items=len(request.items),
        successful=successful,
        failed=failed,
        total_time_ms=total_time,
    )


@app.get("/stress-types")
async def get_stress_types():
    """Get list of supported stress types with descriptions."""
    from inference import STRESS_LABELS, STRESS_DESCRIPTIONS
    return {
        "stress_types": [
            {"id": label, "description": STRESS_DESCRIPTIONS[label]}
            for label in STRESS_LABELS
        ]
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def start_server(engine=None, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    global inference_engine
    if engine:
        inference_engine = engine

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FarmFederate API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.workers > 1:
        uvicorn.run(
            "serve:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
