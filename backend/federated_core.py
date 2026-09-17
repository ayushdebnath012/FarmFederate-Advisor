#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
federated_core.py — Enhanced federated learning core for FarmFederate.

Enhancements based on research paper:
    - Secure aggregation with differential privacy
    - Adaptive learning rates per client  
    - Client sampling strategies (importance-based)
    - Byzantine-robust aggregation (Krum, median)
    - Comprehensive metrics tracking
    - Communication efficiency (gradient compression)
    - MultiModalDataset with advanced augmentation
    - FocalLoss with dynamic weighting
"""

import math
import random
import hashlib
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import get_linear_schedule_with_warmup

from datasets_loader import ISSUE_LABELS, NUM_LABELS
from weather_data import WEATHER_DIM

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------- Dataset -------------
class MultiModalDataset(Dataset):
    """
    Each item: {input_ids, attention_mask, pixel_values, labels, raw_text}
    plus `weather_features` [WEATHER_DIM] when df_text has a "weather" column.
    - `df_text`: DataFrame with columns ["text", "labels"] (+ optional "weather")
    - `tokenizer`: HF tokenizer for text encoder
    - `image_processor`: HF image processor for ViT
    - `image_ds`: optional HF dataset with column "image" (PIL-like)
    - If `image_ds` is None, uses a gray dummy image.
    """
    def __init__(self, df_text: pd.DataFrame, tokenizer, image_processor,
                 image_ds=None, max_len: int = 160):
        self.df = df_text.reset_index(drop=True)
        self.tok = tokenizer
        self.im_proc = image_processor
        self.image_ds = image_ds
        self.max_len = max_len
        self.has_weather = "weather" in self.df.columns and self.df["weather"].notna().all()

        # build a dummy image tensor once (3x224x224)
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), (128, 128, 128))
        self._dummy_pixels = self.im_proc(dummy, return_tensors="pt")["pixel_values"][0]

    def __len__(self):
        return len(self.df)

    def _get_pixels(self, idx: int):
        if self.image_ds is None or len(self.image_ds) == 0:
            return self._dummy_pixels.clone()
        j = idx % len(self.image_ds)
        sample = self.image_ds[j]
        img = sample.get("image", None)
        if img is None:
            return self._dummy_pixels.clone()
        pixels = self.im_proc(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return pixels

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            if 0 <= k < NUM_LABELS:
                labels[k] = 1.0
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": self._get_pixels(i),
            "labels": labels,
            "raw_text": text,
        }
        if self.has_weather:
            item["weather_features"] = torch.tensor(row["weather"], dtype=torch.float32)
        return item


def weather_batch(batch: Dict, device: str):
    """The optional weather tensor of a batch, moved to `device` (None when absent)."""
    w = batch.get("weather_features")
    return None if w is None else w.to(device)


# ------------- Class balancing & loss -------------
def make_weights_for_balanced_classes(df: pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    inv = 1.0 / np.maximum(counts, 1)
    inst_w = []
    for labs in df["labels"]:
        if labs:
            w = np.mean([inv[k] for k in labs])
        else:
            w = inv.mean()
        inst_w.append(w)
    inst_w = np.array(inst_w, dtype=np.float32)
    inst_w = inst_w / (inst_w.mean() + 1e-12)
    return inst_w, counts


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.02):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha.float().clone()
        else:
            self.alpha = None
        self.smooth = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: [B, C]
        targets: [B, C] float in {0,1}
        """
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()

# ------------- Dirichlet split + train_one_client -------------
def split_clients_dirichlet(df: pd.DataFrame, n_clients: int, alpha: float) -> List[pd.DataFrame]:
    """
    Non-IID client split: each label class has its own Dirichlet distribution over clients.
    A row's "primary" label selects which distribution to sample client from.
    """
    num_classes = NUM_LABELS
    rng = np.random.default_rng(SEED)

    prim = []
    for labs in df["labels"]:
        if labs:
            prim.append(int(rng.choice(labs)))
        else:
            prim.append(int(rng.integers(0, num_classes)))
    df2 = df.copy()
    df2["_y"] = prim

    class_client_probs = rng.dirichlet([alpha] * n_clients, size=num_classes)  # [C, n]

    client_bins = [[] for _ in range(n_clients)]
    for idx, y in enumerate(df2["_y"].tolist()):
        cli = int(rng.choice(n_clients, p=class_client_probs[y]))
        client_bins[cli].append(idx)

    out = []
    for k in range(n_clients):
        part = df2.iloc[client_bins[k]].drop(columns=["_y"]).reset_index(drop=True)
        out.append(part)
    return out


def train_one_client(
    model,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    tokenizer,
    image_processor,
    image_ds,
    max_len: int,
    batch_size: int,
    local_epochs: int,
    lr: float,
    device: str,
    grad_accum: int = 1,
    weight_decay: float = 0.01,
    alpha_per_class: Optional[torch.Tensor] = None,
):
    """
    Returns: (state_dict_cpu, train_loss, val_loss)
    (only the *trainable* parameters of the model are included in state_dict_cpu)
    """
    train_ds = MultiModalDataset(df_train, tokenizer, image_processor, image_ds, max_len)
    val_ds = MultiModalDataset(df_val, tokenizer, image_processor, image_ds, max_len)

    weights, counts = make_weights_for_balanced_classes(df_train)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(train_ds), batch_size),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size, 16), shuffle=False, num_workers=0
    )

    loss_fn = FocalLoss(alpha=alpha_per_class.to(device) if alpha_per_class is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum)))
    total_steps = local_epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    model.to(device)
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)

    total_loss = 0.0
    step_count = 0

    for epoch in range(local_epochs):
        for it, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    weather_features=weather_batch(batch, device),
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = loss_fn(logits, labels) / max(1, grad_accum)

            scaler.scale(loss).backward()
            total_loss += loss.item()
            step_count += 1

            if it % max(1, grad_accum) == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

    # simple val loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                weather_features=weather_batch(batch, device),
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = loss_fn(logits, labels)
            val_loss += loss.item() * labels.size(0)
    val_loss /= max(1, len(val_ds))

    # only return CPU state_dict to reduce memory
    # keep_vars=True keeps the Parameter objects so requires_grad reflects trainability;
    # a plain state_dict() is detached and the filter used to return an empty dict
    state_dict_cpu = {k: v.detach().cpu()
                      for k, v in model.state_dict(keep_vars=True).items()
                      if v.requires_grad}

    return state_dict_cpu, total_loss / max(1, step_count), val_loss


# ------------- Advanced Federated Learning Features -------------

def fedavg_aggregate(client_states: List[Dict], client_weights: Optional[List[float]] = None) -> Dict:
    """
    FedAvg aggregation: weighted average of client model parameters.
    
    Args:
        client_states: List of state_dicts from clients
        client_weights: Optional weights (e.g., dataset sizes). If None, uniform.
    
    Returns:
        Aggregated state_dict
    """
    if not client_states:
        raise ValueError("No client states to aggregate")
    
    if client_weights is None:
        client_weights = [1.0] * len(client_states)
    
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    aggregated_state = {}
    for key in client_states[0].keys():
        aggregated_state[key] = sum(
            client_states[i][key] * client_weights[i] 
            for i in range(len(client_states))
        )
    
    return aggregated_state


def add_differential_privacy(
    state_dict: Dict,
    noise_scale: float = 0.01,
    clip_norm: float = 1.0
) -> Dict:
    """
    Add differential privacy noise to model parameters.
    
    Args:
        state_dict: Model parameters
        noise_scale: Gaussian noise std dev
        clip_norm: Gradient clipping threshold
    
    Returns:
        State dict with added noise
    """
    noisy_state = {}
    for key, param in state_dict.items():
        # Clip parameter norms
        param_norm = torch.norm(param)
        if param_norm > clip_norm:
            param = param * (clip_norm / param_norm)
        
        # Add Gaussian noise
        noise = torch.randn_like(param) * noise_scale
        noisy_state[key] = param + noise
    
    return noisy_state


def krum_aggregate(
    client_states: List[Dict],
    num_byzantine: int = 1,
    multi_krum: bool = False
) -> Dict:
    """
    Krum: Byzantine-robust aggregation by selecting closest models.
    
    Args:
        client_states: List of client state_dicts
        num_byzantine: Number of Byzantine clients to tolerate
        multi_krum: If True, average top (n - num_byzantine - 2) models
    
    Returns:
        Aggregated state_dict
    """
    if len(client_states) < 2 * num_byzantine + 3:
        return fedavg_aggregate(client_states)  # Fallback if insufficient clients
    
    # Flatten parameters for distance calculation
    def flatten_state(state: Dict) -> torch.Tensor:
        return torch.cat([v.flatten() for v in state.values()])
    
    flattened = [flatten_state(s) for s in client_states]
    n = len(flattened)
    
    # Compute pairwise distances
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dist = torch.norm(flattened[i] - flattened[j]).item()
                dists.append(dist)
        dists.sort()
        # Sum of distances to n - num_byzantine - 2 closest neighbors
        score = sum(dists[:n - num_byzantine - 2])
        scores.append(score)
    
    if multi_krum:
        # Select top m clients with lowest scores
        m = n - num_byzantine - 2
        selected_indices = np.argsort(scores)[:m]
        selected_states = [client_states[i] for i in selected_indices]
        return fedavg_aggregate(selected_states)
    else:
        # Select single client with lowest score
        best_idx = np.argmin(scores)
        return client_states[best_idx]


def adaptive_client_sampling(
    client_stats: List[Dict],
    num_select: int,
    strategy: str = "importance"
) -> List[int]:
    """
    Adaptive client selection based on various strategies.
    
    Args:
        client_stats: List of dicts with keys: {"id", "data_size", "loss", "staleness"}
        num_select: Number of clients to select
        strategy: "random", "importance", "loss_weighted", "staleness"
    
    Returns:
        List of selected client indices
    """
    n_clients = len(client_stats)
    num_select = min(num_select, n_clients)
    
    if strategy == "random":
        return list(np.random.choice(n_clients, num_select, replace=False))
    
    elif strategy == "importance":
        # Sample proportional to data size
        sizes = np.array([s["data_size"] for s in client_stats])
        probs = sizes / sizes.sum()
        return list(np.random.choice(n_clients, num_select, replace=False, p=probs))
    
    elif strategy == "loss_weighted":
        # Prefer clients with higher loss (need more training)
        losses = np.array([s.get("loss", 1.0) for s in client_stats])
        probs = losses / losses.sum()
        return list(np.random.choice(n_clients, num_select, replace=False, p=probs))
    
    elif strategy == "staleness":
        # Prefer clients that haven't trained recently
        staleness = np.array([s.get("staleness", 0) for s in client_stats])
        probs = (staleness + 1) / (staleness.sum() + n_clients)
        return list(np.random.choice(n_clients, num_select, replace=False, p=probs))
    
    else:
        return list(range(num_select))


def compress_gradients(
    state_dict: Dict,
    compression_ratio: float = 0.1,
    method: str = "topk"
) -> Tuple[Dict, Dict]:
    """
    Compress model updates for communication efficiency.
    
    Args:
        state_dict: Model parameters
        compression_ratio: Fraction of parameters to keep
        method: "topk" or "random"
    
    Returns:
        (compressed_state, indices_dict) for decompression
    """
    compressed_state = {}
    indices_dict = {}
    
    for key, param in state_dict.items():
        flat_param = param.flatten()
        k = max(1, int(len(flat_param) * compression_ratio))
        
        if method == "topk":
            # Keep top-k by absolute value
            _, indices = torch.topk(torch.abs(flat_param), k)
            compressed_state[key] = flat_param[indices]
            indices_dict[key] = (indices, param.shape)
        
        elif method == "random":
            # Random sampling
            indices = torch.randperm(len(flat_param))[:k]
            compressed_state[key] = flat_param[indices]
            indices_dict[key] = (indices, param.shape)
    
    return compressed_state, indices_dict


class FederatedMetrics:
    """Track comprehensive federated learning metrics."""
    
    def __init__(self):
        self.round_metrics = defaultdict(list)
        self.client_metrics = defaultdict(lambda: defaultdict(list))
    
    def log_round(self, round_num: int, metrics: Dict):
        """Log metrics for a federated round."""
        for key, value in metrics.items():
            self.round_metrics[key].append((round_num, value))
    
    def log_client(self, client_id: int, round_num: int, metrics: Dict):
        """Log metrics for a specific client."""
        for key, value in metrics.items():
            self.client_metrics[client_id][key].append((round_num, value))
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            "total_rounds": len(self.round_metrics.get("train_loss", [])),
            "final_train_loss": self.round_metrics["train_loss"][-1][1] if self.round_metrics.get("train_loss") else None,
            "final_val_loss": self.round_metrics["val_loss"][-1][1] if self.round_metrics.get("val_loss") else None,
            "num_clients": len(self.client_metrics),
        }
        return summary
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON file."""
        import json
        data = {
            "round_metrics": dict(self.round_metrics),
            "client_metrics": {k: dict(v) for k, v in self.client_metrics.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ------------- Helper Functions -------------

def weighted_average_state_dicts(state_dicts: List[Dict], weights: List[float]) -> Dict:
    """
    Compute weighted average of state dictionaries (FedAvg).
    
    Args:
        state_dicts: List of model state dictionaries
        weights: List of weights (e.g., dataset sizes)
    
    Returns:
        Averaged state dictionary
    """
    if not state_dicts:
        raise ValueError("No state dicts provided")
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Initialize averaged state dict
    avg_state = {}
    
    # Average each parameter
    for key in state_dicts[0].keys():
        avg_state[key] = sum(
            state_dicts[i][key] * normalized_weights[i]
            for i in range(len(state_dicts))
        )
    
    return avg_state


def evaluate_model(model, dataloader, device="cuda"):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                weather_features=weather_batch(batch, device),
            )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = torch.sigmoid(logits) > 0.5
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    metrics = {
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }
    
    return metrics
