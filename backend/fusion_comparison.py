# ============================================================================
# FUSION ARCHITECTURE COMPARISON - Train and compare all VLM-style fusion models
# Using REAL HuggingFace datasets (not synthetic)
# ============================================================================
"""
This script trains and compares all 8 fusion architectures using real datasets:
- Text: CGIAR/gardian, argilla/farming, ag_news, localmini
- Images: PlantVillage, BD-Crop, Plant-Pathology, PlantWild

Fusion types: concat, attention, gated, clip, flamingo, blip2, coca, unified_io
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from PIL import Image

# Import from project
try:
    from datasets_loader import (
        build_text_corpus_mix,
        load_stress_image_datasets_hf,
        ISSUE_LABELS,
        NUM_LABELS,
        weak_labels
    )
    HAS_LOADER = True
except ImportError:
    print("[WARNING] Could not import datasets_loader, using fallback")
    HAS_LOADER = False
    ISSUE_LABELS = ['water_stress', 'nutrient_def', 'pest_risk', 'disease_risk', 'heat_stress']
    NUM_LABELS = 5

from transformers import AutoTokenizer, AutoImageProcessor

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'text_model': 'prajjwal1/bert-tiny',  # Small for faster comparison
    'vit_model': 'google/vit-base-patch16-224-in21k',
    'num_labels': NUM_LABELS,
    'batch_size': 8,
    'epochs': 10,  # More epochs for real data
    'learning_rate': 2e-4,
    'max_text_samples': 1000,  # Limit for faster comparison
    'max_image_samples': 1000,
    'train_split': 0.8,
    'seed': 42,
    'projection_dim': 256,
    'num_perceiver_latents': 32,
    'num_qformer_queries': 16,
    'max_seq_len': 128,
}

# Fusion types to compare
FUSION_TYPES = ['concat', 'attention', 'gated', 'clip', 'flamingo', 'blip2', 'coca', 'unified_io']

# Set seeds
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# ============================================================================
# REAL MULTIMODAL DATASET
# ============================================================================
class RealMultiModalDataset(Dataset):
    """Dataset using real HuggingFace text and image data."""

    def __init__(self, texts, labels, image_dataset, tokenizer, image_processor,
                 max_seq_len=128):
        self.texts = texts
        self.labels = labels
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len

        # Create index mapping (cycle images if fewer than texts)
        self.num_images = len(image_dataset) if image_dataset else 0
        print(f"[Dataset] {len(texts)} texts, {self.num_images} images")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Text processing
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Label processing (multi-label)
        label_indices = self.labels[idx]
        label_tensor = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for l in label_indices:
            if 0 <= l < NUM_LABELS:
                label_tensor[l] = 1.0

        # Image processing (cycle through images)
        if self.num_images > 0:
            img_idx = idx % self.num_images
            try:
                img = self.image_dataset[img_idx]['image']
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert('RGB')
                else:
                    img = img.convert('RGB')
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
            except Exception as e:
                # Fallback to gray image
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            # No images available - use gray placeholder
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': label_tensor
        }


# ============================================================================
# LIGHTWEIGHT FUSION MODEL (same as before)
# ============================================================================
class LightweightFusionModel(nn.Module):
    """Lightweight fusion model for faster comparison."""

    def __init__(self, fusion_type='concat', text_dim=128, vision_dim=768,
                 num_labels=5, projection_dim=256, num_perceiver_latents=32,
                 num_qformer_queries=16):
        super().__init__()

        self.fusion_type = fusion_type
        self.num_labels = num_labels
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.projection_dim = projection_dim

        # Text encoder
        self.text_embedding = nn.Embedding(30522, text_dim)  # BERT vocab size
        self.text_encoder = nn.TransformerEncoderLayer(
            d_model=text_dim, nhead=4, dim_feedforward=text_dim*4,
            dropout=0.1, batch_first=True
        )
        self.text_pool = nn.AdaptiveAvgPool1d(1)

        # Vision encoder (simple CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.vision_proj_initial = nn.Linear(256 * 7 * 7, vision_dim)

        # Fusion-specific components
        if fusion_type == 'concat':
            fusion_dim = text_dim + vision_dim
        elif fusion_type == 'attention':
            fusion_dim = text_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
            self.vision_proj = nn.Linear(vision_dim, text_dim)
        elif fusion_type == 'gated':
            fusion_dim = text_dim
            self.gate = nn.Sequential(
                nn.Linear(text_dim + vision_dim, text_dim),
                nn.Sigmoid()
            )
            self.vision_proj = nn.Linear(vision_dim, text_dim)
        elif fusion_type == 'clip':
            fusion_dim = projection_dim * 2
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_dim, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        elif fusion_type == 'flamingo':
            fusion_dim = text_dim
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.perceiver_latents = nn.Parameter(torch.randn(num_perceiver_latents, text_dim))
            self.perceiver_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
            self.perceiver_norm = nn.LayerNorm(text_dim)
            self.perceiver_ff = nn.Sequential(
                nn.Linear(text_dim, text_dim * 4),
                nn.GELU(),
                nn.Linear(text_dim * 4, text_dim)
            )
            self.gated_xattn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
            self.xattn_gate = nn.Parameter(torch.tensor([0.1]))
            self.xattn_norm = nn.LayerNorm(text_dim)
        elif fusion_type == 'blip2':
            fusion_dim = text_dim
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.qformer_queries = nn.Parameter(torch.randn(num_qformer_queries, text_dim) * 0.02)
            self.qformer_self_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
            self.qformer_cross_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
            self.qformer_norm1 = nn.LayerNorm(text_dim)
            self.qformer_norm2 = nn.LayerNorm(text_dim)
            self.qformer_ff = nn.Sequential(
                nn.Linear(text_dim, text_dim * 4),
                nn.GELU(),
                nn.Linear(text_dim * 4, text_dim)
            )
            self.query_proj = nn.Linear(text_dim, text_dim)
        elif fusion_type == 'coca':
            fusion_dim = projection_dim * 2 + text_dim
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.vision_proj_contrastive = nn.Sequential(
                nn.Linear(vision_dim, projection_dim),
                nn.LayerNorm(projection_dim)
            )
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.caption_xattn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=4, dropout=0.1, batch_first=True
            )
        elif fusion_type == 'unified_io':
            fusion_dim = text_dim
            self.modality_embeddings = nn.Embedding(3, text_dim)
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.unified_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=text_dim, nhead=4,
                    dim_feedforward=text_dim * 4,
                    dropout=0.1, batch_first=True
                ),
                num_layers=2
            )
        else:
            fusion_dim = text_dim + vision_dim

        self.fusion_dim = fusion_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def encode_text(self, input_ids):
        x = self.text_embedding(input_ids)
        x = self.text_encoder(x)
        x = x.transpose(1, 2)
        x = self.text_pool(x).squeeze(-1)
        return x

    def encode_vision(self, pixel_values):
        x = self.vision_encoder(pixel_values)
        x = x.flatten(1)
        x = self.vision_proj_initial(x)
        x_seq = x.unsqueeze(1).expand(-1, 49, -1)
        return x, x_seq

    def forward(self, input_ids, attention_mask, pixel_values, return_contrastive=False):
        text_features = self.encode_text(input_ids)
        vision_features, vision_seq = self.encode_vision(pixel_values)

        contrastive_outputs = None

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
            contrastive_outputs = {
                'text_embeds': text_embeds,
                'vision_embeds': vision_embeds,
                'logit_scale': self.logit_scale.exp()
            }
        elif self.fusion_type == 'flamingo':
            batch_size = text_features.size(0)
            vision_proj = self.vision_proj(vision_seq)
            latents = self.perceiver_latents.unsqueeze(0).expand(batch_size, -1, -1)
            attn_out, _ = self.perceiver_attn(latents, vision_proj, vision_proj)
            latents = self.perceiver_norm(latents + attn_out)
            latents = latents + self.perceiver_ff(latents)
            text_seq = text_features.unsqueeze(1)
            xattn_out, _ = self.gated_xattn(text_seq, latents, latents)
            fused = text_features + torch.tanh(self.xattn_gate) * self.xattn_norm(xattn_out.squeeze(1))
        elif self.fusion_type == 'blip2':
            batch_size = text_features.size(0)
            vision_proj = self.vision_proj(vision_seq)
            queries = self.qformer_queries.unsqueeze(0).expand(batch_size, -1, -1)
            self_out, _ = self.qformer_self_attn(queries, queries, queries)
            queries = self.qformer_norm1(queries + self_out)
            cross_out, _ = self.qformer_cross_attn(queries, vision_proj, vision_proj)
            queries = self.qformer_norm2(queries + cross_out)
            queries = queries + self.qformer_ff(queries)
            pooled_queries = queries.mean(dim=1)
            fused = self.query_proj(pooled_queries) + text_features
        elif self.fusion_type == 'coca':
            text_embeds = F.normalize(self.text_proj(text_features), dim=-1)
            vision_embeds = F.normalize(self.vision_proj_contrastive(vision_features), dim=-1)
            contrastive_outputs = {
                'text_embeds': text_embeds,
                'vision_embeds': vision_embeds,
                'logit_scale': self.logit_scale.exp()
            }
            vision_proj = self.vision_proj(vision_seq)
            text_seq = text_features.unsqueeze(1)
            caption_out, _ = self.caption_xattn(text_seq, vision_proj, vision_proj)
            caption_features = caption_out.squeeze(1)
            fused = torch.cat([text_embeds, vision_embeds, caption_features], dim=-1)
        elif self.fusion_type == 'unified_io':
            batch_size = text_features.size(0)
            text_token = self.modality_embeddings(torch.zeros(batch_size, dtype=torch.long, device=text_features.device))
            vision_token = self.modality_embeddings(torch.ones(batch_size, dtype=torch.long, device=text_features.device))
            fused_token = self.modality_embeddings(torch.full((batch_size,), 2, dtype=torch.long, device=text_features.device))
            vision_proj = self.vision_proj(vision_features)
            sequence = torch.stack([fused_token, text_features + text_token, vision_proj + vision_token], dim=1)
            unified_out = self.unified_transformer(sequence)
            fused = unified_out[:, 0]
        else:
            fused = torch.cat([text_features, vision_features], dim=-1)

        logits = self.classifier(fused)

        if return_contrastive and contrastive_outputs is not None:
            return {'logits': logits, 'contrastive': contrastive_outputs}
        return logits


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def compute_contrastive_loss(text_embeds, vision_embeds, logit_scale):
    logits_per_text = logit_scale * text_embeds @ vision_embeds.T
    logits_per_image = logits_per_text.T
    batch_size = text_embeds.size(0)
    target = torch.arange(batch_size, device=text_embeds.device)
    loss_t2i = F.cross_entropy(logits_per_text, target)
    loss_i2t = F.cross_entropy(logits_per_image, target)
    return (loss_t2i + loss_i2t) / 2


def train_epoch(model, dataloader, optimizer, fusion_type, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        use_contrastive = fusion_type in ['clip', 'coca']
        outputs = model(input_ids, attention_mask, pixel_values, return_contrastive=use_contrastive)

        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        if use_contrastive and isinstance(outputs, dict) and 'contrastive' in outputs:
            cont = outputs['contrastive']
            cont_loss = compute_contrastive_loss(
                cont['text_embeds'], cont['vision_embeds'], cont['logit_scale']
            )
            loss = cls_loss + 0.5 * cont_loss
        else:
            loss = cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, fusion_type, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, pixel_values)

            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = {
        'loss': total_loss / num_batches,
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'exact_match': np.mean(np.all(all_preds == all_labels, axis=1))
    }

    return metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, dataloader, device, num_batches=10):
    model.eval()
    times = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            if i == 0:
                _ = model(input_ids, attention_mask, pixel_values)

            start = time.time()
            _ = model(input_ids, attention_mask, pixel_values)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

    return np.mean(times) * 1000


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    print("=" * 70)
    print("FUSION ARCHITECTURE COMPARISON (REAL DATASETS)")
    print("=" * 70)

    # Load real datasets
    print("\n[1/5] Loading REAL datasets from HuggingFace...")

    if HAS_LOADER:
        # Load text data
        print("\n--- Loading Text Data ---")
        df_text = build_text_corpus_mix(
            mix_sources="localmini",  # Use local for speed, can add "gardian,argilla,agnews"
            max_per_source=CONFIG['max_text_samples'],
            max_samples=CONFIG['max_text_samples']
        )
        texts = df_text['text'].tolist()
        labels = df_text['labels'].tolist()
        print(f"Loaded {len(texts)} text samples")

        # Load image data
        print("\n--- Loading Image Data ---")
        image_dataset = load_stress_image_datasets_hf(
            max_total_images=CONFIG['max_image_samples'],
            max_per_dataset=CONFIG['max_image_samples'] // 2
        )
        if image_dataset:
            print(f"Loaded {len(image_dataset)} images")
        else:
            print("No images loaded, using placeholders")
    else:
        # Fallback to local captions
        print("Using local captions.csv fallback...")
        try:
            df = pd.read_csv('data/captions.csv')
            texts = df['caption'].tolist()[:CONFIG['max_text_samples']]
            # Generate weak labels
            labels = [weak_labels(t) if HAS_LOADER else [0] for t in texts]
        except:
            texts = ["Plant showing signs of water stress and wilting."] * 100
            labels = [[0]] * 100
        image_dataset = None

    # Load tokenizer and image processor
    print("\n[2/5] Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['text_model'])
    image_processor = AutoImageProcessor.from_pretrained(CONFIG['vit_model'])

    # Create dataset
    print("\n[3/5] Creating multimodal dataset...")
    dataset = RealMultiModalDataset(
        texts=texts,
        labels=labels,
        image_dataset=image_dataset,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_seq_len=CONFIG['max_seq_len']
    )

    # Split
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Results storage
    results = {}
    training_histories = {}

    # Train each fusion type
    print("\n[4/5] Training each fusion architecture...")
    print("-" * 70)

    for fusion_type in FUSION_TYPES:
        print(f"\n>>> Training: {fusion_type.upper()}")

        model = LightweightFusionModel(
            fusion_type=fusion_type,
            text_dim=128,
            vision_dim=768,
            num_labels=CONFIG['num_labels'],
            projection_dim=CONFIG['projection_dim'],
            num_perceiver_latents=CONFIG['num_perceiver_latents'],
            num_qformer_queries=CONFIG['num_qformer_queries']
        ).to(DEVICE)

        num_params = count_parameters(model)
        print(f"    Parameters: {num_params:,}")

        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0

        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, fusion_type, DEVICE)
            val_metrics = evaluate(model, val_loader, fusion_type, DEVICE)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_micro'])

            if val_metrics['f1_micro'] > best_f1:
                best_f1 = val_metrics['f1_micro']

            print(f"    Epoch {epoch+1}/{CONFIG['epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                  f"val_f1={val_metrics['f1_micro']:.4f}")

        final_metrics = evaluate(model, val_loader, fusion_type, DEVICE)
        inference_time = measure_inference_time(model, val_loader, DEVICE)

        results[fusion_type] = {
            'num_params': num_params,
            'inference_time_ms': inference_time,
            'final_loss': final_metrics['loss'],
            'f1_micro': final_metrics['f1_micro'],
            'f1_macro': final_metrics['f1_macro'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'exact_match': final_metrics['exact_match'],
            'best_f1': best_f1
        }

        training_histories[fusion_type] = history

        print(f"    Final F1: {final_metrics['f1_micro']:.4f}, "
              f"Inference: {inference_time:.2f}ms")

    # Generate report
    print("\n[5/5] Generating comparison report...")
    print("=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    df = pd.DataFrame(results).T
    df = df.round(4)
    print("\n" + df.to_string())

    # Rankings
    print("\n" + "=" * 70)
    print("RANKINGS")
    print("=" * 70)

    f1_ranking = sorted(results.items(), key=lambda x: x[1]['f1_micro'], reverse=True)
    print("\nBy F1 Score:")
    for i, (name, metrics) in enumerate(f1_ranking, 1):
        print(f"  {i}. {name:12s} - F1: {metrics['f1_micro']:.4f}")

    time_ranking = sorted(results.items(), key=lambda x: x[1]['inference_time_ms'])
    print("\nBy Inference Time:")
    for i, (name, metrics) in enumerate(time_ranking, 1):
        print(f"  {i}. {name:12s} - {metrics['inference_time_ms']:.2f}ms")

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fusion Architecture Comparison (Real Data)', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(FUSION_TYPES)))

    # Training loss
    ax1 = axes[0, 0]
    for i, fusion_type in enumerate(FUSION_TYPES):
        ax1.plot(training_histories[fusion_type]['train_loss'], label=fusion_type, color=colors[i])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Val F1
    ax2 = axes[0, 1]
    for i, fusion_type in enumerate(FUSION_TYPES):
        ax2.plot(training_histories[fusion_type]['val_f1'], label=fusion_type, color=colors[i])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Bar chart F1
    ax3 = axes[0, 2]
    f1_scores = [results[ft]['f1_micro'] for ft in FUSION_TYPES]
    bars = ax3.bar(FUSION_TYPES, f1_scores, color=colors)
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Final F1')
    ax3.set_xticklabels(FUSION_TYPES, rotation=45, ha='right')

    # Inference time
    ax4 = axes[1, 0]
    inf_times = [results[ft]['inference_time_ms'] for ft in FUSION_TYPES]
    ax4.bar(FUSION_TYPES, inf_times, color=colors)
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Inference Time')
    ax4.set_xticklabels(FUSION_TYPES, rotation=45, ha='right')

    # Parameters
    ax5 = axes[1, 1]
    params = [results[ft]['num_params'] / 1e6 for ft in FUSION_TYPES]
    ax5.bar(FUSION_TYPES, params, color=colors)
    ax5.set_ylabel('Parameters (M)')
    ax5.set_title('Parameter Count')
    ax5.set_xticklabels(FUSION_TYPES, rotation=45, ha='right')

    # Efficiency scatter
    ax6 = axes[1, 2]
    for i, fusion_type in enumerate(FUSION_TYPES):
        ax6.scatter(results[fusion_type]['inference_time_ms'],
                   results[fusion_type]['f1_micro'],
                   s=100, c=[colors[i]], label=fusion_type, alpha=0.7)
    ax6.set_xlabel('Inference Time (ms)')
    ax6.set_ylabel('F1 Score')
    ax6.set_title('Efficiency: F1 vs Speed')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fusion_comparison_real_data.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved to: fusion_comparison_real_data.png")

    with open('fusion_comparison_real_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to: fusion_comparison_real_data.json")

    # Recommendations
    best_f1_type = max(results.items(), key=lambda x: x[1]['f1_micro'])[0]
    fastest_type = min(results.items(), key=lambda x: x[1]['inference_time_ms'])[0]

    print(f"""
    ========================================
    RECOMMENDATIONS (Real Data)
    ========================================
    Best F1 Score:      {best_f1_type.upper()} ({results[best_f1_type]['f1_micro']:.4f})
    Fastest Inference:  {fastest_type.upper()} ({results[fastest_type]['inference_time_ms']:.2f}ms)
    """)

    return results, training_histories


if __name__ == '__main__':
    results, histories = run_comparison()
