# FarmFederate API Documentation

## Overview

FarmFederate provides a REST API for crop stress detection using multimodal AI (LLM + ViT + VLM fusion). The API supports text-based, image-based, and multimodal predictions.

**Base URL**: `http://localhost:8000`

---

## Authentication

Currently, the API does not require authentication. For production deployment, implement API key or OAuth2 authentication.

---

## Endpoints

### Health Check

```http
GET /health
```

Check API server health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "vlm",
  "device": "cuda",
  "uptime_seconds": 3600.5
}
```

---

### Text-Based Prediction

```http
POST /predict/text
Content-Type: application/json
```

Predict crop stress from text description.

**Request Body:**
```json
{
  "text": "The tomato plant leaves are yellowing and curling at the edges with brown spots"
}
```

**Response:**
```json
{
  "prediction": "disease_risk",
  "confidence": 0.87,
  "description": "Disease risk detected - fungal, bacterial, or viral infection likely",
  "all_probabilities": {
    "water_stress": 0.05,
    "nutrient_def": 0.03,
    "pest_risk": 0.02,
    "disease_risk": 0.87,
    "heat_stress": 0.03
  },
  "recommendations": [
    "Isolate affected plants",
    "Improve air circulation",
    "Remove and destroy infected tissue",
    "Apply fungicide/bactericide as appropriate"
  ],
  "high_confidence": true,
  "processing_time_ms": 45.2
}
```

---

### Image-Based Prediction

```http
POST /predict/image
Content-Type: multipart/form-data
```

Predict crop stress from image (optionally with text).

**Request Parameters:**
- `image` (file, required): Crop image file (JPG/PNG)
- `text` (string, optional): Additional text description

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "image=@crop_photo.jpg" \
  -F "text=yellowing leaves observed"
```

**Response:** Same format as text prediction.

---

### Batch Prediction

```http
POST /predict/batch
Content-Type: application/json
```

Batch prediction for multiple text descriptions.

**Request Body:**
```json
{
  "items": [
    "Wilted leaves with dry brown edges",
    "White powdery coating on leaves",
    "Small holes in leaves with visible insects"
  ]
}
```

**Response:**
```json
{
  "results": [
    {"prediction": "water_stress", "confidence": 0.82, ...},
    {"prediction": "disease_risk", "confidence": 0.91, ...},
    {"prediction": "pest_risk", "confidence": 0.88, ...}
  ],
  "total_items": 3,
  "successful": 3,
  "failed": 0,
  "total_time_ms": 134.5
}
```

---

### Get Stress Types

```http
GET /stress-types
```

Get list of supported stress types with descriptions.

**Response:**
```json
{
  "stress_types": [
    {"id": "water_stress", "description": "Water stress detected - plant shows signs of drought or overwatering"},
    {"id": "nutrient_def", "description": "Nutrient deficiency detected - plant lacks essential minerals"},
    {"id": "pest_risk", "description": "Pest risk detected - signs of insect damage or infestation"},
    {"id": "disease_risk", "description": "Disease risk detected - fungal, bacterial, or viral infection likely"},
    {"id": "heat_stress", "description": "Heat stress detected - thermal damage or sun scorch present"}
  ]
}
```

---

## Stress Types

| Type | Description | Common Symptoms |
|------|-------------|-----------------|
| `water_stress` | Drought or overwatering | Wilting, drooping, dry/soggy soil |
| `nutrient_def` | Mineral deficiency | Yellowing, chlorosis, stunted growth |
| `pest_risk` | Insect damage | Holes, bite marks, visible pests |
| `disease_risk` | Pathogen infection | Spots, lesions, mold, rot |
| `heat_stress` | Thermal damage | Leaf scorch, curling, bleaching |

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid image file type"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction error: <error message>"
}
```

---

## Rate Limits

Default rate limits (configurable):
- 100 requests per minute per IP
- 1000 requests per hour per IP

---

## SDK Examples

### Python

```python
import requests

# Text prediction
response = requests.post(
    "http://localhost:8000/predict/text",
    json={"text": "yellowing leaves with brown spots"}
)
result = response.json()
print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")

# Image prediction
with open("crop.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/image",
        files={"image": f},
        data={"text": "outdoor tomato plant"}
    )
result = response.json()
```

### JavaScript

```javascript
// Text prediction
const response = await fetch('http://localhost:8000/predict/text', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'yellowing leaves with brown spots'})
});
const result = await response.json();
console.log(`Prediction: ${result.prediction} (${(result.confidence * 100).toFixed(1)}%)`);
```

### Flutter/Dart

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictStress(String text) async {
  final response = await http.post(
    Uri.parse('http://localhost:8000/predict/text'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'text': text}),
  );
  return jsonDecode(response.body);
}
```

---

## Deployment

### Docker

```bash
# Build and run
docker-compose up -d

# With GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MODEL_PATH=models/best_model.pt
export DEVICE=cuda
export CONFIDENCE_THRESHOLD=0.5

# Start server
python serve.py --port 8000 --workers 4
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best_model.pt` | Path to trained model |
| `DEVICE` | auto-detect | `cuda` or `cpu` |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for high-confidence flag |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PORT` | `8000` | API server port |

---

## OpenAPI/Swagger

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
