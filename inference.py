#!/usr/bin/env python3
"""
FarmFederate Production Inference Script

This script provides a production-ready interface for running crop stress detection
inference using trained multimodal models.

Usage:
    python inference.py --model_path models/best_vlm.pt --image crop.jpg --text "yellowing leaves"
    python inference.py --model_path models/best_vlm.pt --image_dir ./images/
    python inference.py --serve --port 8000  # Start REST API server
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Stress labels
STRESS_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']

# Stress descriptions for human-readable output
STRESS_DESCRIPTIONS = {
    'water_stress': 'Water stress detected - plant shows signs of drought or overwatering',
    'nutrient_def': 'Nutrient deficiency detected - plant lacks essential minerals',
    'pest_risk': 'Pest risk detected - signs of insect damage or infestation',
    'disease_risk': 'Disease risk detected - fungal, bacterial, or viral infection likely',
    'heat_stress': 'Heat stress detected - thermal damage or sun scorch present',
}


class CropStressInference:
    """Production inference class for crop stress detection."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        self.image_processor = None

        self._load_model(model_path)
        self._setup_preprocessors()

    def _load_model(self, model_path: str):
        """Load the trained model from checkpoint."""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine model type from checkpoint
        self.model_type = checkpoint.get('model_type', 'vlm')
        self.model_config = checkpoint.get('config', {})

        # Load model architecture
        if self.model_type == 'vlm':
            from FarmFederate_Colab import MultiModalClassifier
            self.model = MultiModalClassifier(
                num_labels=len(STRESS_LABELS),
                fusion_type=self.model_config.get('fusion_type', 'coca'),
            )
        elif self.model_type == 'vision':
            from FarmFederate_Colab import LightweightVisionClassifier
            self.model = LightweightVisionClassifier(num_labels=len(STRESS_LABELS))
        else:
            from FarmFederate_Colab import LightweightTextClassifier
            self.model = LightweightTextClassifier(num_labels=len(STRESS_LABELS))

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully ({self.model_type} type)")

    def _setup_preprocessors(self):
        """Setup image and text preprocessors."""
        try:
            from transformers import AutoTokenizer, AutoImageProcessor
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        except Exception:
            print("Warning: Using basic preprocessors (HuggingFace not available)")
            self.tokenizer = None
            self.image_processor = None

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # Resize and normalize
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0).to(self.device)

    def preprocess_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input.

        Args:
            text: Input text description
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            return {k: v.to(self.device) for k, v in encoded.items()}
        else:
            # Basic tokenization fallback
            words = text.lower().split()[:max_length]
            vocab = {w: i + 1 for i, w in enumerate(set(words))}
            input_ids = [vocab.get(w, 0) for w in words]
            input_ids = input_ids + [0] * (max_length - len(input_ids))
            attention_mask = [1 if i > 0 else 0 for i in input_ids]
            return {
                'input_ids': torch.tensor([input_ids]).to(self.device),
                'attention_mask': torch.tensor([attention_mask]).to(self.device),
            }

    @torch.no_grad()
    def predict(
        self,
        image: Optional[Union[str, Path, Image.Image]] = None,
        text: Optional[str] = None,
    ) -> Dict:
        """Run inference on input data.

        Args:
            image: Input image (path, PIL Image, or array)
            text: Input text description

        Returns:
            Dictionary with predictions, confidence scores, and recommendations
        """
        if image is None and text is None:
            raise ValueError("At least one of 'image' or 'text' must be provided")

        inputs = {}

        if image is not None:
            inputs['pixel_values'] = self.preprocess_image(image)

        if text is not None:
            text_inputs = self.preprocess_text(text)
            inputs.update(text_inputs)

        # Run model
        outputs = self.model(**inputs)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get predictions
        pred_idx = np.argmax(probs)
        pred_label = STRESS_LABELS[pred_idx]
        confidence = float(probs[pred_idx])

        # Build result
        result = {
            'prediction': pred_label,
            'confidence': confidence,
            'description': STRESS_DESCRIPTIONS[pred_label],
            'all_probabilities': {
                label: float(probs[i]) for i, label in enumerate(STRESS_LABELS)
            },
            'recommendations': self._get_recommendations(pred_label, confidence),
            'high_confidence': confidence >= self.confidence_threshold,
        }

        return result

    def _get_recommendations(self, stress_type: str, confidence: float) -> List[str]:
        """Get actionable recommendations based on prediction."""
        recommendations = {
            'water_stress': [
                'Check soil moisture levels',
                'Adjust irrigation schedule',
                'Consider mulching to retain moisture',
                'Monitor for signs of root rot if overwatered',
            ],
            'nutrient_def': [
                'Conduct soil nutrient analysis',
                'Apply balanced fertilizer',
                'Check soil pH levels',
                'Consider foliar feeding for quick recovery',
            ],
            'pest_risk': [
                'Inspect plants for visible pests',
                'Set up pest traps for monitoring',
                'Consider integrated pest management (IPM)',
                'Apply appropriate pesticide if confirmed',
            ],
            'disease_risk': [
                'Isolate affected plants',
                'Improve air circulation',
                'Remove and destroy infected tissue',
                'Apply fungicide/bactericide as appropriate',
            ],
            'heat_stress': [
                'Provide shade during peak hours',
                'Increase watering frequency',
                'Apply anti-transpirant sprays',
                'Mulch to keep roots cool',
            ],
        }

        recs = recommendations.get(stress_type, ['Consult agricultural expert'])

        if confidence < self.confidence_threshold:
            recs.insert(0, f'Low confidence ({confidence:.1%}) - consider manual inspection')

        return recs

    def batch_predict(self, items: List[Dict]) -> List[Dict]:
        """Run batch inference on multiple items.

        Args:
            items: List of dicts with 'image' and/or 'text' keys

        Returns:
            List of prediction results
        """
        results = []
        for item in items:
            try:
                result = self.predict(
                    image=item.get('image'),
                    text=item.get('text'),
                )
                result['status'] = 'success'
            except Exception as e:
                result = {'status': 'error', 'error': str(e)}
            results.append(result)
        return results


def main():
    """Main entry point for CLI inference."""
    parser = argparse.ArgumentParser(description='FarmFederate Crop Stress Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--text', type=str, help='Text description of symptoms')
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch inference')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--serve', action='store_true', help='Start REST API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for REST API server')

    args = parser.parse_args()

    # Initialize inference engine
    engine = CropStressInference(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.threshold,
    )

    if args.serve:
        # Start REST API server
        from serve import start_server
        start_server(engine, port=args.port)
    elif args.image_dir:
        # Batch inference on directory
        image_dir = Path(args.image_dir)
        items = [{'image': str(p)} for p in image_dir.glob('*.jpg')]
        items.extend([{'image': str(p)} for p in image_dir.glob('*.png')])

        print(f"Running batch inference on {len(items)} images...")
        results = engine.batch_predict(items)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for item, result in zip(items, results):
                print(f"\n{item['image']}:")
                print(f"  Prediction: {result.get('prediction', 'error')}")
                print(f"  Confidence: {result.get('confidence', 0):.1%}")
    else:
        # Single inference
        result = engine.predict(image=args.image, text=args.text)

        print("\n" + "=" * 60)
        print("CROP STRESS DETECTION RESULT")
        print("=" * 60)
        print(f"Prediction:  {result['prediction']}")
        print(f"Confidence:  {result['confidence']:.1%}")
        print(f"Description: {result['description']}")
        print("\nAll Probabilities:")
        for label, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
            bar = '*' * int(prob * 20)
            print(f"  {label:<15} {prob:6.1%} {bar}")
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
