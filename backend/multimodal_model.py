#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multimodal_model.py — Enhanced Roberta + ViT multimodal classifier for crop stress.

Research paper enhancements:
- Text encoder: roberta-base with attention pooling
- Image encoder: google/vit-base-patch16-224-in21k with spatial features
- Cross-modal attention mechanism for better fusion
- Multi-scale feature extraction
- Projection to 512-d embeddings, cross-attention → fusion → MLP → 5 labels
- Uncertainty estimation via Monte Carlo dropout
- Attention visualization for explainability

This file provides:
 - class MultiModalModel : the enhanced server model
 - alias MultimodalClassifier -> MultiModalModel for compatibility
 - .text_encoder, .vision_encoder attributes (used by PEFT / adapter loading)
 - .classifier attribute (used by server fallback)
 - forward(...) accepts image=None (text-only), returns object with .logits and .attention_weights
 - helper methods: forward_text_image(...), predict(...), get_uncertainty(...)
"""

from typing import Optional, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace

from transformers import AutoModel, AutoTokenizer
from transformers import ViTModel, AutoImageProcessor

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
NUM_LABELS = len(ISSUE_LABELS)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for text-image fusion."""
    
    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, Q, D]
            key_value: [B, K, D]
        Returns:
            (attended_output [B, Q, D], attention_weights [B, Q, K])
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output, attn_weights


class MultiModalModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        num_labels: int = NUM_LABELS,
        freeze_backbones: bool = False,
        use_cross_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args
        - text_model_name: HuggingFace name for text encoder (AutoModel)
        - image_model_name: HuggingFace name for ViT encoder
        - freeze_backbones: if True, freeze encoder weights (useful for LoRA)
        - use_cross_attention: if True, use cross-modal attention fusion
        - dropout: dropout rate for fusion layers
        """
        super().__init__()

        # ----- backbones -----
        # Keep attribute names expected by external code / peft utilities:
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vision_encoder = ViTModel.from_pretrained(image_model_name)

        if freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        hidden_t = self.text_encoder.config.hidden_size
        hidden_i = self.vision_encoder.config.hidden_size

        # ----- projections to common dimension -----
        self.use_cross_attention = use_cross_attention
        self.projection_dim = 512
        
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_t, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(hidden_i, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(dropout)
        )

        # ----- cross-modal attention -----
        if self.use_cross_attention:
            self.cross_attn_t2i = CrossModalAttention(self.projection_dim, num_heads=8, dropout=dropout)
            self.cross_attn_i2t = CrossModalAttention(self.projection_dim, num_heads=8, dropout=dropout)

        # ----- fusion head -----
        fusion_input_dim = self.projection_dim * 2 if not use_cross_attention else self.projection_dim * 4
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # final classifier head (exposed as `.classifier` to help server fallback)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

        # convenience alias for backwards compatibility
        self.text_backbone = self.text_encoder
        self.image_backbone = self.vision_encoder
        
        # for uncertainty estimation
        self.mc_dropout = nn.Dropout(0.2)

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Return pooled text features [B, hidden_t] (CLS pooling).
        """
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use pooler_output if present (some models), else CLS token
        if hasattr(t_out, "pooler_output") and t_out.pooler_output is not None:
            t_cls = t_out.pooler_output
        else:
            # last_hidden_state [B, L, H]
            t_cls = t_out.last_hidden_state[:, 0, :]
        return t_cls

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Return pooled image features [B, hidden_i] (CLS pooling).
        """
        i_out = self.vision_encoder(pixel_values=pixel_values)
        if hasattr(i_out, "pooler_output") and i_out.pooler_output is not None:
            i_cls = i_out.pooler_output
        else:
            i_cls = i_out.last_hidden_state[:, 0, :]
        return i_cls

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_features: bool = False,
    ) -> SimpleNamespace:
        """
        Enhanced forward with cross-modal attention and attention weights.

        Args:
          input_ids: [B, L]
          attention_mask: [B, L]
          pixel_values: [B, 3, H, W] or None (text-only)
          return_attention: if True, return attention weights

        Returns:
          SimpleNamespace with attributes:
            - logits: Tensor [B, num_labels]
            - attention_weights: Optional dict of attention weights
        """
        # Text features
        t_cls = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # [B, Ht]
        t_feat = self.text_proj(t_cls)  # [B, 512]
        t_feat = t_feat.unsqueeze(1)  # [B, 1, 512] for attention

        # Image features (if present)
        if pixel_values is not None:
            i_cls = self.get_image_features(pixel_values=pixel_values)  # [B, Hi]
            i_feat = self.image_proj(i_cls)  # [B, 512]
            i_feat = i_feat.unsqueeze(1)  # [B, 1, 512] for attention
        else:
            # if no image provided, use zeros for image feature
            device = t_feat.device
            i_feat = torch.zeros_like(t_feat, device=device)  # [B, 1, 512]

        attention_weights = {}
        
        # Cross-modal attention fusion
        if self.use_cross_attention:
            # Text attending to image
            t_attended, attn_t2i = self.cross_attn_t2i(t_feat, i_feat)  # [B, 1, 512]
            # Image attending to text
            i_attended, attn_i2t = self.cross_attn_i2t(i_feat, t_feat)  # [B, 1, 512]
            
            if return_attention:
                attention_weights["text_to_image"] = attn_t2i
                attention_weights["image_to_text"] = attn_i2t
            
            # Concatenate attended features
            fused_input = torch.cat([
                t_feat.squeeze(1),
                i_feat.squeeze(1),
                t_attended.squeeze(1),
                i_attended.squeeze(1)
            ], dim=-1)  # [B, 512*4=2048]
        else:
            # Simple concatenation
            fused_input = torch.cat([t_feat.squeeze(1), i_feat.squeeze(1)], dim=-1)  # [B, 512*2=1024]

        # Fusion and classification
        fused = self.fusion(fused_input)  # [B, 512]  — h_f used by RAG query builder
        logits = self.classifier(fused)  # [B, num_labels]

        result = SimpleNamespace(logits=logits)
        if return_features:
            result.fused_features = fused   # h_f [B, 512]
        if return_attention:
            result.attention_weights = attention_weights

        return result

    # explicit helper used by server fallback
    def forward_text_image(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        """
        Same as forward but matches server naming used in fallback attempts.
        Accepts `image` (not pixel_values) for convenience.
        """
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)

    # convenience predict wrapper (tokenize outside typically)
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return logits Tensor [B, num_labels] directly (no namespace).
        """
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
        return out.logits
    
    def get_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            pixel_values: [B, 3, H, W] or None
            n_samples: Number of forward passes
        
        Returns:
            (mean_logits [B, num_labels], std_logits [B, num_labels])
        """
        self.train()  # Enable dropout
        
        logits_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(input_ids, attention_mask, pixel_values)
                logits_list.append(out.logits)
        
        logits_stack = torch.stack(logits_list, dim=0)  # [n_samples, B, num_labels]
        mean_logits = logits_stack.mean(dim=0)
        std_logits = logits_stack.std(dim=0)
        
        self.eval()
        return mean_logits, std_logits


# Provide original name as alias for compatibility (if some code imports that)
MultimodalClassifier = MultiModalModel


# Tokenizer / image processor helpers (kept for convenience)
def build_tokenizer(model_name: str = "roberta-base"):
    """
    Returns AutoTokenizer (kept signature similar to earlier code).
    """
    return AutoTokenizer.from_pretrained(model_name)


def build_image_processor(model_name: str = "google/vit-base-patch16-224-in21k"):
    """
    Returns AutoImageProcessor for ViT preprocessing.
    """
    return AutoImageProcessor.from_pretrained(model_name)
