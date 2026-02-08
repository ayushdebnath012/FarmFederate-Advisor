#!/usr/bin/env python3
"""
================================================================================
CROP STRESS DETECTION & RECOMMENDATION SYSTEM
================================================================================
A complete system for detecting plant stress from:
- Text descriptions (farmer observations, logs)
- Images (plant photos)
- Sensor data (soil moisture, temperature, humidity, pH, VPD, rainfall)

With actionable recommendations for each detected stress type.

For Kaggle/Colab: Copy this code and run!
================================================================================
"""

import os
import re
import json
import random
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {DEVICE}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# STRESS LABELS & RECOMMENDATIONS
# ============================================================================
ISSUE_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']
NUM_LABELS = len(ISSUE_LABELS)

STRESS_INFO = {
    'water_stress': {
        'name': 'Water Stress',
        'keywords': ['dry', 'wilting', 'wilt', 'parched', 'drought', 'moisture',
                     'irrigation', 'droop', 'cracking soil', 'water stress'],
        'symptoms': ['Wilting leaves', 'Drooping stems', 'Dry/cracked soil', 'Curled foliage', 'Leaf drop'],
        'immediate_actions': [
            'Irrigate immediately - deep watering preferred',
            'Apply mulch (3-4 inches) to retain soil moisture',
            'Check and repair irrigation systems',
            'Water early morning or late evening',
            'Reduce plant spacing to minimize evaporation'
        ],
        'preventive': [
            'Install soil moisture sensors for monitoring',
            'Use drought-resistant crop varieties',
            'Implement drip irrigation system',
            'Add organic matter to improve water retention',
            'Schedule deficit irrigation during vegetative stage'
        ],
        'sensor_triggers': {'soil_moisture': '<20%', 'vpd': '>2.0 kPa'}
    },
    'nutrient_def': {
        'name': 'Nutrient Deficiency',
        'keywords': ['nitrogen', 'phosphorus', 'potassium', 'npk', 'fertilizer',
                     'chlorosis', 'yellowing', 'interveinal', 'deficiency', 'spad'],
        'symptoms': ['Yellowing leaves (chlorosis)', 'Stunted growth', 'Interveinal chlorosis',
                     'Purple/red discoloration', 'Necrotic leaf margins'],
        'immediate_actions': [
            'Apply balanced NPK fertilizer (ratio based on soil test)',
            'Foliar spray with micronutrients (Fe, Mn, Zn)',
            'Apply urea for quick nitrogen boost',
            'Side-dress with potassium sulfate if K deficient',
            'Adjust soil pH if outside 6.0-7.0 range'
        ],
        'preventive': [
            'Regular soil testing every 6 months',
            'Use slow-release fertilizers',
            'Implement crop rotation with legumes',
            'Add compost/organic matter annually',
            'Monitor with SPAD meter or leaf color charts'
        ],
        'sensor_triggers': {'soil_ph': '<5.8 or >7.4', 'spad': '<35'}
    },
    'pest_risk': {
        'name': 'Pest Infestation',
        'keywords': ['pest', 'aphid', 'whitefly', 'borer', 'caterpillar', 'larvae',
                     'insect', 'mites', 'thrips', 'chewed', 'holes', 'frass'],
        'symptoms': ['Holes in leaves', 'Chewed leaf margins', 'Webbing on plants',
                     'Sticky honeydew residue', 'Visible insects/larvae', 'Frass (insect droppings)'],
        'immediate_actions': [
            'Identify pest species before treatment',
            'Apply neem oil spray (2-3ml/L water)',
            'Use targeted insecticide based on pest type',
            'Remove heavily infested plant parts',
            'Set up yellow sticky traps for monitoring'
        ],
        'preventive': [
            'Introduce beneficial insects (ladybugs, lacewings)',
            'Practice companion planting (marigolds, basil)',
            'Weekly field scouting and monitoring',
            'Maintain field hygiene - remove crop residues',
            'Use pheromone traps for early detection'
        ],
        'sensor_triggers': {'humidity': '45-70%', 'temperature': '25-35C'}
    },
    'disease_risk': {
        'name': 'Disease Risk',
        'keywords': ['blight', 'rust', 'mildew', 'rot', 'fungal', 'bacterial',
                     'lesion', 'spots', 'mosaic', 'necrosis', 'pathogen'],
        'symptoms': ['Leaf spots/lesions', 'Rust pustules', 'Powdery/downy mildew',
                     'Wilting despite adequate water', 'Rotting stems/roots', 'Mosaic patterns'],
        'immediate_actions': [
            'Remove and destroy infected plant material',
            'Apply copper-based fungicide for bacterial diseases',
            'Use systemic fungicide for fungal infections',
            'Improve air circulation around plants',
            'Reduce overhead irrigation immediately'
        ],
        'preventive': [
            'Use disease-resistant varieties',
            'Practice 3-4 year crop rotation',
            'Maintain proper plant spacing',
            'Apply preventive fungicide before monsoon',
            'Avoid working in wet fields'
        ],
        'sensor_triggers': {'humidity': '>80%', 'rainfall_24h': '>5mm'}
    },
    'heat_stress': {
        'name': 'Heat Stress',
        'keywords': ['heatwave', 'hot', 'scorch', 'sunburn', 'thermal stress',
                     'high temperature', 'desiccation', 'heat stress', 'burning'],
        'symptoms': ['Leaf scorching/browning', 'Wilting in afternoon', 'Flower/fruit drop',
                     'Reduced fruit set', 'Premature senescence', 'Sunburn on fruits'],
        'immediate_actions': [
            'Provide temporary shade (shade cloth 30-50%)',
            'Increase irrigation frequency',
            'Apply kaolin clay spray for cooling',
            'Mulch heavily to cool soil',
            'Spray water on foliage in early morning'
        ],
        'preventive': [
            'Use heat-tolerant crop varieties',
            'Schedule planting to avoid peak summer',
            'Install permanent shade structures',
            'Use white plastic mulch to reflect heat',
            'Maintain adequate potassium levels'
        ],
        'sensor_triggers': {'temperature': '>38C', 'vpd': '>2.5 kPa'}
    }
}

# ============================================================================
# SENSOR PRIORS - Adjust predictions based on sensor data
# ============================================================================
class SensorPriors:
    """Calculate stress priors from sensor readings."""

    SENSOR_PATTERN = re.compile(
        r"soil_moisture=(?P<sm>\d+(?:\.\d+)?)%.*?"
        r"soil_pH=(?P<ph>\d+(?:\.\d+)?).*?"
        r"temp=(?P<t>\d+(?:\.\d+)?).*?"
        r"humidity=(?P<h>\d+(?:\.\d+)?)%.*?"
        r"VPD=(?P<vpd>\d+(?:\.\d+)?).*?"
        r"rainfall_24h=(?P<rf>\d+(?:\.\d+)?)",
        re.I | re.S
    )

    @staticmethod
    def parse_sensors(text: str) -> Optional[Dict[str, float]]:
        """Extract sensor values from text."""
        match = SensorPriors.SENSOR_PATTERN.search(text)
        if not match:
            return None
        try:
            return {
                'soil_moisture': float(match.group('sm')),
                'soil_ph': float(match.group('ph')),
                'temperature': float(match.group('t')),
                'humidity': float(match.group('h')),
                'vpd': float(match.group('vpd')),
                'rainfall_24h': float(match.group('rf'))
            }
        except:
            return None

    @staticmethod
    def generate_sensor_string(sensors: Dict[str, float] = None) -> str:
        """Generate sensor summary string."""
        if sensors is None:
            sensors = {
                'soil_moisture': round(np.clip(np.random.normal(30, 8), 10, 60), 1),
                'soil_ph': round(np.clip(np.random.normal(6.5, 0.5), 5.0, 8.0), 1),
                'temperature': round(np.clip(np.random.normal(28, 5), 15, 45), 1),
                'humidity': round(np.clip(np.random.normal(60, 15), 20, 95), 0),
                'vpd': round(np.clip(np.random.normal(1.5, 0.5), 0.5, 3.0), 2),
                'rainfall_24h': round(np.clip(np.random.exponential(2), 0, 20), 1)
            }

        return (f"SENSORS: soil_moisture={sensors['soil_moisture']}%, "
                f"soil_pH={sensors['soil_ph']}, temp={sensors['temperature']}C, "
                f"humidity={sensors['humidity']}%, VPD={sensors['vpd']} kPa, "
                f"rainfall_24h={sensors['rainfall_24h']}mm")

    @staticmethod
    def calculate_priors(sensors: Dict[str, float]) -> np.ndarray:
        """Calculate stress probability adjustments from sensor data."""
        priors = np.zeros(NUM_LABELS, dtype=np.float32)

        sm = sensors.get('soil_moisture', 30)
        ph = sensors.get('soil_ph', 6.5)
        temp = sensors.get('temperature', 28)
        hum = sensors.get('humidity', 60)
        vpd = sensors.get('vpd', 1.5)
        rain = sensors.get('rainfall_24h', 0)

        # Water stress
        if sm < 20:
            priors[0] += 0.25
        elif sm < 25:
            priors[0] += 0.15
        if vpd > 2.0:
            priors[0] += 0.15
        if sm > 35 and vpd < 1.5:
            priors[0] -= 0.20

        # Nutrient deficiency
        if ph < 5.5 or ph > 7.5:
            priors[1] += 0.20
        elif ph < 6.0 or ph > 7.0:
            priors[1] += 0.10

        # Pest risk
        if 45 <= hum <= 75 and 25 <= temp <= 35:
            priors[2] += 0.12
        if rain < 1.0 and hum < 50:
            priors[2] += 0.08

        # Disease risk
        if hum > 80:
            priors[3] += 0.25
        elif hum > 70:
            priors[3] += 0.15
        if rain > 5:
            priors[3] += 0.20
        if hum < 50 and rain < 1:
            priors[3] -= 0.15

        # Heat stress
        if temp > 38:
            priors[4] += 0.30
        elif temp > 35:
            priors[4] += 0.20
        if vpd > 2.5:
            priors[4] += 0.15
        if temp < 25:
            priors[4] -= 0.20

        return np.clip(priors, -0.3, 0.4)


# ============================================================================
# TEXT-BASED STRESS DETECTION (Keyword + ML hybrid)
# ============================================================================
def detect_stress_from_keywords(text: str) -> Tuple[List[int], List[float]]:
    """Detect stress using keyword matching."""
    text_lower = text.lower()
    predictions = []
    confidences = []

    for i, label in enumerate(ISSUE_LABELS):
        info = STRESS_INFO[label]
        keywords = info['keywords']

        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in text_lower)

        if matches >= 3:
            predictions.append(1)
            confidences.append(min(0.85, 0.5 + matches * 0.1))
        elif matches >= 2:
            predictions.append(1)
            confidences.append(min(0.7, 0.4 + matches * 0.1))
        elif matches >= 1:
            predictions.append(0)
            confidences.append(0.3 + matches * 0.1)
        else:
            predictions.append(0)
            confidences.append(0.1)

    return predictions, confidences


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================
class TextStressClassifier(nn.Module):
    """LLM-style text classifier for stress detection."""

    def __init__(self, vocab_size=15000, embed_dim=256, hidden_dim=256, num_labels=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

        # Build vocabulary from stress keywords
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for info in STRESS_INFO.values():
            for kw in info['keywords']:
                for word in kw.split():
                    if word not in self.word2idx:
                        self.word2idx[word] = idx
                        idx += 1

    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        words = text.lower().split()
        tokens = [self.word2idx.get(w, 1) for w in words]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.encoder(x)

        # Attention pooling
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = (x * attn_weights).sum(dim=1)

        return self.classifier(x)


class ImageStressClassifier(nn.Module):
    """ViT-style image classifier for stress detection."""

    def __init__(self, num_labels=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, pixel_values):
        x = self.features(pixel_values)
        return self.classifier(x)


class MultimodalStressClassifier(nn.Module):
    """VLM-style multimodal classifier combining text + image + sensors."""

    def __init__(self, vocab_size=15000, num_labels=5):
        super().__init__()

        # Text encoder
        self.text_embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.text_encoder = nn.LSTM(128, 128, batch_first=True, bidirectional=True)

        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Fusion & classifier
        # text: 256, image: 128, sensor: 64 = 448
        self.fusion = nn.Sequential(
            nn.LayerNorm(448),
            nn.Linear(448, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

        # Vocabulary
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for info in STRESS_INFO.values():
            for kw in info['keywords']:
                for word in kw.split():
                    if word not in self.word2idx:
                        self.word2idx[word] = idx
                        idx += 1

    def tokenize(self, text: str, max_len: int = 64) -> torch.Tensor:
        words = text.lower().split()
        tokens = [self.word2idx.get(w, 1) for w in words]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def forward(self, input_ids, pixel_values, sensor_values):
        # Text
        text_emb = self.text_embed(input_ids)
        text_out, _ = self.text_encoder(text_emb)
        text_feat = text_out.mean(dim=1)  # [B, 256]

        # Image
        img_feat = self.image_encoder(pixel_values)  # [B, 128]

        # Sensors
        sensor_feat = self.sensor_encoder(sensor_values)  # [B, 64]

        # Fuse
        fused = torch.cat([text_feat, img_feat, sensor_feat], dim=-1)
        return self.fusion(fused)


# ============================================================================
# CROP STRESS ADVISOR - Main Interface
# ============================================================================
class CropStressAdvisor:
    """Main class for crop stress detection and recommendations."""

    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self.device = DEVICE

        if use_ml:
            self.text_model = TextStressClassifier().to(self.device)
            self.image_model = ImageStressClassifier().to(self.device)
            self.multimodal_model = MultimodalStressClassifier().to(self.device)

            # Set to eval mode (would load trained weights in production)
            self.text_model.eval()
            self.image_model.eval()
            self.multimodal_model.eval()

    def analyze_text(self, text: str, sensors: Dict[str, float] = None) -> Dict:
        """Analyze text description for stress detection."""

        # 1. Keyword-based detection
        kw_preds, kw_confs = detect_stress_from_keywords(text)

        # 2. Parse sensors from text or use provided
        if sensors is None:
            sensors = SensorPriors.parse_sensors(text)

        # 3. Calculate sensor priors
        if sensors:
            priors = SensorPriors.calculate_priors(sensors)
        else:
            priors = np.zeros(NUM_LABELS)

        # 4. Combine keyword + sensor priors
        final_probs = np.array(kw_confs) + priors * 0.3
        final_probs = np.clip(final_probs, 0, 1)

        # 5. ML prediction (if enabled)
        if self.use_ml:
            with torch.no_grad():
                tokens = self.text_model.tokenize(text).unsqueeze(0).to(self.device)
                ml_logits = self.text_model(tokens)
                ml_probs = torch.sigmoid(ml_logits).cpu().numpy()[0]
                # Combine: 60% ML, 40% keyword+sensor
                final_probs = 0.6 * ml_probs + 0.4 * final_probs

        # 6. Generate predictions
        predictions = (final_probs > 0.4).astype(int).tolist()

        return {
            'predictions': predictions,
            'probabilities': final_probs.tolist(),
            'sensors': sensors,
            'detected_stresses': [ISSUE_LABELS[i] for i, p in enumerate(predictions) if p == 1]
        }

    def analyze_image(self, image_tensor: torch.Tensor) -> Dict:
        """Analyze plant image for stress detection."""

        if not self.use_ml:
            # Return random predictions for demo
            probs = np.random.uniform(0.1, 0.6, NUM_LABELS)
            predictions = (probs > 0.4).astype(int).tolist()
            return {
                'predictions': predictions,
                'probabilities': probs.tolist(),
                'detected_stresses': [ISSUE_LABELS[i] for i, p in enumerate(predictions) if p == 1]
            }

        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            logits = self.image_model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = (probs > 0.4).astype(int).tolist()

        return {
            'predictions': predictions,
            'probabilities': probs.tolist(),
            'detected_stresses': [ISSUE_LABELS[i] for i, p in enumerate(predictions) if p == 1]
        }

    def analyze_multimodal(self, text: str, image_tensor: torch.Tensor,
                           sensors: Dict[str, float] = None) -> Dict:
        """Analyze using text + image + sensors."""

        # Parse sensors
        if sensors is None:
            sensors = SensorPriors.parse_sensors(text)
        if sensors is None:
            sensors = {
                'soil_moisture': 30.0, 'soil_ph': 6.5, 'temperature': 28.0,
                'humidity': 60.0, 'vpd': 1.5, 'rainfall_24h': 0.0
            }

        sensor_tensor = torch.tensor([
            sensors['soil_moisture'], sensors['soil_ph'], sensors['temperature'],
            sensors['humidity'], sensors['vpd'], sensors['rainfall_24h']
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tokens = self.multimodal_model.tokenize(text).unsqueeze(0).to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            logits = self.multimodal_model(tokens, image_tensor, sensor_tensor)
            ml_probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Add sensor priors
        priors = SensorPriors.calculate_priors(sensors)
        final_probs = np.clip(ml_probs + priors * 0.2, 0, 1)

        predictions = (final_probs > 0.4).astype(int).tolist()

        return {
            'predictions': predictions,
            'probabilities': final_probs.tolist(),
            'sensors': sensors,
            'detected_stresses': [ISSUE_LABELS[i] for i, p in enumerate(predictions) if p == 1]
        }

    def get_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate recommendations based on analysis results."""

        recommendations = []
        predictions = analysis['predictions']
        probabilities = analysis['probabilities']

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1 or prob > 0.35:  # Include near-misses
                stress_type = ISSUE_LABELS[i]
                info = STRESS_INFO[stress_type]

                # Determine severity
                if prob >= 0.7:
                    severity = 'SEVERE'
                    urgency = 'IMMEDIATE ACTION REQUIRED'
                    color = 'RED'
                elif prob >= 0.5:
                    severity = 'MODERATE'
                    urgency = 'Action within 24-48 hours'
                    color = 'ORANGE'
                elif prob >= 0.35:
                    severity = 'MILD'
                    urgency = 'Monitor and take preventive action'
                    color = 'YELLOW'
                else:
                    continue

                rec = {
                    'stress_type': info['name'],
                    'label': stress_type,
                    'severity': severity,
                    'urgency': urgency,
                    'color': color,
                    'confidence': f"{prob*100:.1f}%",
                    'probability': prob,
                    'symptoms': info['symptoms'],
                    'immediate_actions': info['immediate_actions'],
                    'preventive_measures': info['preventive'],
                    'sensor_triggers': info['sensor_triggers']
                }
                recommendations.append(rec)

        # Sort by probability (highest first)
        recommendations.sort(key=lambda x: x['probability'], reverse=True)

        return recommendations

    def generate_report(self, text: str = None, image: torch.Tensor = None,
                        sensors: Dict[str, float] = None) -> str:
        """Generate a complete stress analysis report."""

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CROP STRESS ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)

        # Determine analysis type
        if text and image is not None:
            report_lines.append("\nAnalysis Type: MULTIMODAL (Text + Image + Sensors)")
            analysis = self.analyze_multimodal(text, image, sensors)
        elif text:
            report_lines.append("\nAnalysis Type: TEXT-BASED (with Sensor Priors)")
            analysis = self.analyze_text(text, sensors)
        elif image is not None:
            report_lines.append("\nAnalysis Type: IMAGE-BASED")
            analysis = self.analyze_image(image)
        else:
            return "ERROR: No input provided"

        # Input summary
        if text:
            report_lines.append(f"\nInput Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        if analysis.get('sensors'):
            report_lines.append(f"\nSensor Data:")
            for k, v in analysis['sensors'].items():
                report_lines.append(f"  {k}: {v}")

        # Detection results
        report_lines.append("\n" + "-" * 50)
        report_lines.append("DETECTION RESULTS")
        report_lines.append("-" * 50)

        for i, label in enumerate(ISSUE_LABELS):
            prob = analysis['probabilities'][i]
            pred = analysis['predictions'][i]
            status = "DETECTED" if pred else "Not detected"
            bar = "#" * int(prob * 20) + "-" * (20 - int(prob * 20))
            report_lines.append(f"  {STRESS_INFO[label]['name']:20s} [{bar}] {prob*100:5.1f}% - {status}")

        # Recommendations
        recommendations = self.get_recommendations(analysis)

        if recommendations:
            report_lines.append("\n" + "-" * 50)
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 50)

            for rec in recommendations:
                report_lines.append(f"\n[{rec['severity']}] {rec['stress_type']} ({rec['confidence']})")
                report_lines.append(f"  Urgency: {rec['urgency']}")
                report_lines.append(f"\n  Common Symptoms:")
                for symptom in rec['symptoms'][:3]:
                    report_lines.append(f"    - {symptom}")
                report_lines.append(f"\n  Immediate Actions:")
                for action in rec['immediate_actions'][:3]:
                    report_lines.append(f"    1. {action}")
                report_lines.append(f"\n  Preventive Measures:")
                for measure in rec['preventive_measures'][:2]:
                    report_lines.append(f"    * {measure}")
        else:
            report_lines.append("\n[OK] No significant stress detected!")
            report_lines.append("  Continue regular monitoring and maintenance.")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)


# ============================================================================
# DEMO / MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("CROP STRESS DETECTION & RECOMMENDATION SYSTEM")
    print("=" * 70)

    # Initialize advisor
    advisor = CropStressAdvisor(use_ml=True)

    # Demo inputs
    demo_cases = [
        {
            'name': 'Case 1: Multiple Stresses',
            'text': ("SENSORS: soil_moisture=15%, soil_pH=6.2, temp=38C, humidity=45%, "
                     "VPD=2.3 kPa, rainfall_24h=0mm\n"
                     "LOG: Maize field showing severe wilting and yellowing on older leaves. "
                     "Soil is cracked and dry. Plants drooping during midday heat."),
        },
        {
            'name': 'Case 2: Disease Risk',
            'text': ("SENSORS: soil_moisture=45%, soil_pH=6.5, temp=26C, humidity=85%, "
                     "VPD=0.8 kPa, rainfall_24h=12mm\n"
                     "LOG: Tomato plants have brown lesions and spots on leaves. "
                     "White powdery mildew observed on lower canopy. Very humid conditions."),
        },
        {
            'name': 'Case 3: Pest Infestation',
            'text': ("SENSORS: soil_moisture=30%, soil_pH=6.8, temp=30C, humidity=55%, "
                     "VPD=1.5 kPa, rainfall_24h=0mm\n"
                     "LOG: Cotton plants have aphids on undersides of leaves. "
                     "Sticky honeydew residue observed. Some leaves have holes and chewed margins."),
        },
        {
            'name': 'Case 4: Healthy Crop',
            'text': ("SENSORS: soil_moisture=35%, soil_pH=6.5, temp=28C, humidity=60%, "
                     "VPD=1.2 kPa, rainfall_24h=2mm\n"
                     "LOG: Rice field looks healthy with good green color. "
                     "No visible issues. Plants growing well with proper tillering."),
        }
    ]

    for case in demo_cases:
        print(f"\n{'='*70}")
        print(f"  {case['name']}")
        print(f"{'='*70}")

        report = advisor.generate_report(text=case['text'])
        print(report)
        print("\n")

    # Interactive mode hint
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE:")
    print("=" * 70)
    print("""
# Python usage:
advisor = CropStressAdvisor(use_ml=True)

# Text-only analysis
analysis = advisor.analyze_text("Wheat leaves showing rust and yellowing")
recommendations = advisor.get_recommendations(analysis)

# With sensor data
sensors = {'soil_moisture': 20, 'soil_ph': 6.5, 'temperature': 35,
           'humidity': 50, 'vpd': 2.0, 'rainfall_24h': 0}
analysis = advisor.analyze_text("Plants wilting under sun", sensors)

# Generate full report
report = advisor.generate_report(text="Your observation here", sensors=sensors)
print(report)
    """)


if __name__ == '__main__':
    main()
