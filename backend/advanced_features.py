# ============================================================================
# ADVANCED FEATURES: Fusion Model, FocalLoss, Sensor Fusion, Weak Labeling
# ============================================================================
print("="*70)
print("LOADING ADVANCED FEATURES")
print("="*70)

# ============================================================================
# 1. SENSOR FUSION - Simulate IoT sensor data and fuse with text
# ============================================================================
def simulate_sensor_summary():
    """Generate realistic IoT sensor readings for agricultural monitoring."""
    soil_m = round(np.clip(np.random.normal(30, 8), 10, 50), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.5), 5.0, 8.0), 1)
    temp = round(np.clip(np.random.normal(28, 5), 15, 42), 1)
    humidity = round(np.clip(np.random.normal(60, 15), 25, 95), 0)
    vpd = round(np.clip(np.random.normal(1.4, 0.5), 0.4, 2.8), 2)
    rainfall = round(np.clip(np.random.exponential(2.0), 0.0, 15.0), 1)
    light = round(np.clip(np.random.normal(45000, 15000), 5000, 80000), 0)

    trend = np.random.choice(["rising", "falling", "stable"], p=[0.3, 0.3, 0.4])

    return {
        'soil_moisture': soil_m,
        'soil_ph': soil_ph,
        'temperature': temp,
        'humidity': humidity,
        'vpd': vpd,
        'rainfall_24h': rainfall,
        'light_intensity': light,
        'trend': trend
    }

def format_sensor_text(sensors: dict) -> str:
    """Format sensor readings as text for model input."""
    return (f"SENSORS: soil_moisture={sensors['soil_moisture']}%, "
            f"soil_pH={sensors['soil_ph']}, temp={sensors['temperature']}°C, "
            f"humidity={sensors['humidity']}%, VPD={sensors['vpd']} kPa, "
            f"rainfall_24h={sensors['rainfall_24h']}mm, "
            f"light={sensors['light_intensity']} lux (trend: {sensors['trend']})")

def fuse_sensor_with_text(text: str, sensors: dict = None) -> str:
    """Fuse sensor data with text description."""
    if sensors is None:
        sensors = simulate_sensor_summary()
    sensor_str = format_sensor_text(sensors)
    return f"{sensor_str}\nLOG: {text.strip()}"

def sensor_to_priors(sensors: dict) -> np.ndarray:
    """Convert sensor readings to label prior probabilities."""
    priors = np.zeros(NUM_LABELS, dtype=np.float32)

    # Water stress indicators
    if sensors['soil_moisture'] < 20 or sensors['vpd'] > 2.0:
        priors[0] += 0.3  # water_stress
    if sensors['soil_moisture'] > 45:
        priors[0] -= 0.2

    # Nutrient deficiency indicators
    if sensors['soil_ph'] < 5.5 or sensors['soil_ph'] > 7.5:
        priors[1] += 0.2  # nutrient_def

    # Pest risk indicators
    if 20 < sensors['temperature'] < 32 and 50 < sensors['humidity'] < 75:
        priors[2] += 0.15  # pest_risk (favorable conditions)

    # Disease risk indicators
    if sensors['humidity'] > 80 or sensors['rainfall_24h'] > 5:
        priors[3] += 0.25  # disease_risk
    if sensors['humidity'] < 40:
        priors[3] -= 0.15

    # Heat stress indicators
    if sensors['temperature'] > 35 or sensors['vpd'] > 2.2:
        priors[4] += 0.3  # heat_stress
    if sensors['temperature'] < 25:
        priors[4] -= 0.2

    return np.clip(priors, -0.5, 0.5)

# ============================================================================
# 2. WEAK LABELING - Keyword-based automatic labeling
# ============================================================================
STRESS_KEYWORDS = {
    'water_stress': [
        'dry', 'wilting', 'wilt', 'drought', 'parched', 'moisture', 'irrigation',
        'drooping', 'cracking soil', 'water stress', 'dehydration', 'thirsty'
    ],
    'nutrient_def': [
        'nitrogen', 'phosphorus', 'potassium', 'npk', 'fertilizer', 'chlorosis',
        'yellowing', 'pale leaves', 'stunted', 'deficiency', 'nutrient', 'spad',
        'interveinal', 'necrotic margin', 'micronutrient'
    ],
    'pest_risk': [
        'pest', 'aphid', 'whitefly', 'borer', 'caterpillar', 'larvae', 'insect',
        'thrips', 'mites', 'weevil', 'hopper', 'chewed', 'holes', 'webbing',
        'honeydew', 'frass', 'infestation'
    ],
    'disease_risk': [
        'blight', 'rust', 'mildew', 'rot', 'fungal', 'bacterial', 'viral',
        'lesion', 'spot', 'necrosis', 'pathogen', 'infection', 'disease',
        'powdery', 'downy', 'canker', 'wilt disease', 'mosaic'
    ],
    'heat_stress': [
        'heat', 'hot', 'scorch', 'sunburn', 'thermal', 'high temperature',
        'heatwave', 'burning', 'desiccation', 'heat stress', 'blistering'
    ]
}

def weak_label_text(text: str) -> List[int]:
    """Generate weak labels from text using keyword matching."""
    text_lower = text.lower()
    labels = []

    for label_name, keywords in STRESS_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            labels.append(ISSUE_LABELS.index(label_name))

    return sorted(set(labels))

def enhance_labels_with_sensors(text_labels: List[int], sensors: dict, threshold: float = 0.2) -> List[int]:
    """Enhance text-based labels with sensor-based priors."""
    priors = sensor_to_priors(sensors)
    enhanced = set(text_labels)

    for i, prior in enumerate(priors):
        if prior > threshold and i not in enhanced:
            enhanced.add(i)

    return sorted(enhanced)

# ============================================================================
# 3. FOCAL LOSS - Handle class imbalance
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.02, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute focal weight
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================================
# 4. VLM-STYLE FUSION COMPONENTS - Advanced Vision-Language Model architectures
# ============================================================================

class PerceiverResampler(nn.Module):
    """
    Flamingo-style Perceiver Resampler for compressing visual features.
    Maps variable-length visual tokens to fixed-length latent queries.
    """
    def __init__(self, dim: int, num_latents: int = 64, num_heads: int = 8,
                 depth: int = 2, ff_mult: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * ff_mult, dim),
                    nn.Dropout(0.1)
                ),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        """x: [B, seq_len, dim] -> [B, num_latents, dim]"""
        batch_size = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for attn, norm1, ff, norm2 in self.layers:
            # Cross-attention: latents attend to visual features
            attn_out, _ = attn(latents, x, x)
            latents = norm1(latents + attn_out)
            latents = norm2(latents + ff(latents))

        return latents


class GatedCrossAttentionBlock(nn.Module):
    """
    Flamingo-style Gated Cross-Attention for interleaving visual info into text.
    Uses tanh gating initialized near zero for stable training.
    """
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(dim)

        # Gating parameter initialized to small value (tanh(0.1) ≈ 0.1)
        self.cross_attn_gate = nn.Parameter(torch.tensor([0.1]))

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(0.1)
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff_gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, text_features, visual_features):
        """
        text_features: [B, text_len, dim]
        visual_features: [B, vis_len, dim]
        """
        # Gated cross-attention
        attn_out, attn_weights = self.cross_attn(text_features, visual_features, visual_features)
        text_features = text_features + torch.tanh(self.cross_attn_gate) * self.cross_attn_norm(attn_out)

        # Gated feed-forward
        text_features = text_features + torch.tanh(self.ff_gate) * self.ff_norm(self.ff(text_features))

        return text_features, attn_weights


class ContrastiveHead(nn.Module):
    """
    CLIP-style contrastive learning head with learnable temperature.
    Projects both modalities to shared embedding space.
    """
    def __init__(self, text_dim: int, vision_dim: int, projection_dim: int = 512):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        # Learnable temperature (log scale for stability)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

    def forward(self, text_features, vision_features):
        """Project and normalize features for contrastive learning."""
        text_embeds = F.normalize(self.text_proj(text_features), dim=-1)
        vision_embeds = F.normalize(self.vision_proj(vision_features), dim=-1)
        return text_embeds, vision_embeds, self.logit_scale.exp()


class ImageTextMatchingHead(nn.Module):
    """
    BLIP-style Image-Text Matching (ITM) head.
    Binary classification for matching/non-matching pairs.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.itm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # match / no-match
        )

    def forward(self, fused_features):
        return self.itm_head(fused_features)


class QFormerBlock(nn.Module):
    """
    BLIP-2 style Q-Former block with learnable query tokens.
    Bridges frozen encoders with learnable queries.
    """
    def __init__(self, dim: int, num_queries: int = 32, num_heads: int = 8,
                 num_layers: int = 2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers * 4)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, visual_features, text_features=None):
        """
        visual_features: [B, vis_len, dim]
        text_features: [B, text_len, dim] (optional for text-conditioned queries)
        """
        batch_size = visual_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        for i, (self_attn, cross_attn, ffn) in enumerate(
            zip(self.self_attn_layers, self.cross_attn_layers, self.ffns)
        ):
            # Self-attention among queries (optionally with text)
            if text_features is not None:
                combined = torch.cat([queries, text_features], dim=1)
                self_out, _ = self_attn(queries, combined, combined)
            else:
                self_out, _ = self_attn(queries, queries, queries)
            queries = self.norms[i*4](queries + self_out)

            # Cross-attention to visual features
            cross_out, _ = cross_attn(queries, visual_features, visual_features)
            queries = self.norms[i*4 + 1](queries + cross_out)

            # Feed-forward
            queries = self.norms[i*4 + 2](queries + ffn(queries))

        return queries


# ============================================================================
# 4b. MULTIMODAL FUSION MODEL - Enhanced with VLM-style fusion options
# ============================================================================
class MultiModalFusionModel(nn.Module):
    """
    Unified multimodal model that fuses text (LLM) and image (ViT) features.

    Supports multiple VLM-style fusion strategies:
    - 'concat': Simple concatenation (baseline)
    - 'attention': Cross-attention fusion
    - 'gated': Gated fusion with learned weights
    - 'clip': CLIP-style contrastive + concat fusion
    - 'flamingo': Flamingo-style gated cross-attention with perceiver
    - 'blip2': BLIP-2 style Q-Former fusion
    - 'coca': CoCa-style dual objective (contrastive + generative)
    - 'unified_io': Unified IO style with modality tokens
    """

    FUSION_TYPES = ['concat', 'attention', 'gated', 'clip', 'flamingo', 'blip2', 'coca', 'unified_io']

    def __init__(self, text_model_name: str, vit_model_name: str, num_labels: int,
                 fusion_type: str = 'concat', use_lora: bool = True,
                 freeze_text: bool = True, freeze_vision: bool = True,
                 projection_dim: int = 512, num_perceiver_latents: int = 64,
                 num_qformer_queries: int = 32):
        super().__init__()

        self.fusion_type = fusion_type
        self.num_labels = num_labels
        self.projection_dim = projection_dim

        # Text Encoder (LLM)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.text_encoder.config.hidden_size

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA to text encoder
        if use_lora and HAS_PEFT:
            lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=get_lora_target_modules(text_model_name),
                lora_dropout=0.1, bias="none"
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        # Vision Encoder (ViT)
        self.vision_encoder = ViTModel.from_pretrained(vit_model_name)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA to vision encoder
        if use_lora and HAS_PEFT:
            vit_lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1, bias="none"
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora_config)

        # ============================================================
        # FUSION-SPECIFIC COMPONENTS
        # ============================================================

        if fusion_type == 'concat':
            fusion_dim = self.text_hidden_size + self.vision_hidden_size

        elif fusion_type == 'attention':
            fusion_dim = self.text_hidden_size
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.text_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)

        elif fusion_type == 'gated':
            fusion_dim = self.text_hidden_size
            self.gate = nn.Sequential(
                nn.Linear(self.text_hidden_size + self.vision_hidden_size, self.text_hidden_size),
                nn.Sigmoid()
            )
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)

        elif fusion_type == 'clip':
            # CLIP-style: contrastive alignment + concatenated classification
            fusion_dim = projection_dim * 2
            self.contrastive_head = ContrastiveHead(
                self.text_hidden_size, self.vision_hidden_size, projection_dim
            )

        elif fusion_type == 'flamingo':
            # Flamingo-style: perceiver resampler + gated cross-attention
            fusion_dim = self.text_hidden_size
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
            self.perceiver = PerceiverResampler(
                dim=self.text_hidden_size,
                num_latents=num_perceiver_latents,
                num_heads=8,
                depth=2
            )
            self.gated_cross_attn = GatedCrossAttentionBlock(
                dim=self.text_hidden_size,
                num_heads=8
            )

        elif fusion_type == 'blip2':
            # BLIP-2 style: Q-Former bridges frozen encoders
            fusion_dim = self.text_hidden_size
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
            self.qformer = QFormerBlock(
                dim=self.text_hidden_size,
                num_queries=num_qformer_queries,
                num_heads=8,
                num_layers=2
            )
            self.query_proj = nn.Linear(self.text_hidden_size, self.text_hidden_size)

        elif fusion_type == 'coca':
            # CoCa-style: dual contrastive + captioning objectives
            fusion_dim = projection_dim * 2 + self.text_hidden_size
            self.contrastive_head = ContrastiveHead(
                self.text_hidden_size, self.vision_hidden_size, projection_dim
            )
            # Cross-attention for captioning-style fusion
            self.caption_cross_attn = nn.MultiheadAttention(
                embed_dim=self.text_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)

        elif fusion_type == 'unified_io':
            # Unified-IO style: modality-specific tokens + shared transformer
            fusion_dim = self.text_hidden_size
            self.modality_embeddings = nn.Embedding(3, self.text_hidden_size)  # text, vision, fused
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
            self.unified_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.text_hidden_size,
                    nhead=8,
                    dim_feedforward=self.text_hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            )
        else:
            fusion_dim = self.text_hidden_size + self.vision_hidden_size

        self.fusion_dim = fusion_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

        # Sensor prior integration
        self.sensor_proj = nn.Linear(num_labels, num_labels)

        # ITM head for BLIP-style training (optional)
        if fusion_type in ['blip2', 'coca']:
            self.itm_head = ImageTextMatchingHead(self.text_hidden_size)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                sensor_priors=None, return_features=False, return_contrastive=False):

        # Text encoding
        text_features = None
        text_sequence = None
        if input_ids is not None:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            text_sequence = text_outputs.last_hidden_state  # [B, seq_len, dim]
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_features = text_outputs.pooler_output
            else:
                text_features = text_sequence[:, 0]

        # Vision encoding
        vision_features = None
        vision_sequence = None
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True
            )
            vision_sequence = vision_outputs.last_hidden_state  # [B, num_patches, dim]
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                vision_features = vision_outputs.pooler_output
            else:
                vision_features = vision_sequence[:, 0]

        # ============================================================
        # FUSION LOGIC
        # ============================================================
        contrastive_outputs = None
        attn_weights = None

        if text_features is not None and vision_features is not None:

            if self.fusion_type == 'concat':
                fused = torch.cat([text_features, vision_features], dim=-1)

            elif self.fusion_type == 'attention':
                vision_proj = self.vision_proj(vision_features).unsqueeze(1)
                text_seq = text_features.unsqueeze(1)
                attn_out, attn_weights = self.cross_attention(text_seq, vision_proj, vision_proj)
                fused = (text_features + attn_out.squeeze(1)) / 2

            elif self.fusion_type == 'gated':
                vision_proj = self.vision_proj(vision_features)
                gate = self.gate(torch.cat([text_features, vision_features], dim=-1))
                fused = text_features + gate * vision_proj

            elif self.fusion_type == 'clip':
                # CLIP-style: project to shared space + concat
                text_embeds, vision_embeds, logit_scale = self.contrastive_head(
                    text_features, vision_features
                )
                fused = torch.cat([text_embeds, vision_embeds], dim=-1)
                contrastive_outputs = {
                    'text_embeds': text_embeds,
                    'vision_embeds': vision_embeds,
                    'logit_scale': logit_scale
                }

            elif self.fusion_type == 'flamingo':
                # Flamingo-style: perceiver resampler + gated cross-attention
                vision_proj = self.vision_proj(vision_sequence)  # [B, num_patches, text_dim]
                visual_tokens = self.perceiver(vision_proj)  # [B, num_latents, text_dim]

                # Gated cross-attention: text attends to compressed visual tokens
                text_seq = text_features.unsqueeze(1)  # [B, 1, dim]
                fused_seq, attn_weights = self.gated_cross_attn(text_seq, visual_tokens)
                fused = fused_seq.squeeze(1)

            elif self.fusion_type == 'blip2':
                # BLIP-2 style: Q-Former bridges frozen encoders
                vision_proj = self.vision_proj(vision_sequence)  # [B, num_patches, text_dim]
                query_outputs = self.qformer(vision_proj, text_sequence)  # [B, num_queries, dim]

                # Pool query outputs and combine with text
                pooled_queries = query_outputs.mean(dim=1)  # [B, dim]
                fused = self.query_proj(pooled_queries) + text_features

            elif self.fusion_type == 'coca':
                # CoCa-style: contrastive + cross-attention for classification
                text_embeds, vision_embeds, logit_scale = self.contrastive_head(
                    text_features, vision_features
                )
                contrastive_outputs = {
                    'text_embeds': text_embeds,
                    'vision_embeds': vision_embeds,
                    'logit_scale': logit_scale
                }

                # Cross-attention for richer fusion
                vision_proj = self.vision_proj(vision_sequence)
                text_seq = text_features.unsqueeze(1)
                caption_out, attn_weights = self.caption_cross_attn(text_seq, vision_proj, vision_proj)
                caption_features = caption_out.squeeze(1)

                # Concatenate all representations
                fused = torch.cat([text_embeds, vision_embeds, caption_features], dim=-1)

            elif self.fusion_type == 'unified_io':
                # Unified-IO style: modality tokens + shared transformer
                batch_size = text_features.size(0)

                # Add modality embeddings
                text_token = self.modality_embeddings(torch.zeros(batch_size, dtype=torch.long, device=text_features.device))
                vision_token = self.modality_embeddings(torch.ones(batch_size, dtype=torch.long, device=text_features.device))
                fused_token = self.modality_embeddings(torch.full((batch_size,), 2, dtype=torch.long, device=text_features.device))

                # Project vision to text space
                vision_proj = self.vision_proj(vision_features)

                # Stack as sequence: [fused_query, text+modality, vision+modality]
                sequence = torch.stack([
                    fused_token,
                    text_features + text_token,
                    vision_proj + vision_token
                ], dim=1)  # [B, 3, dim]

                # Process through unified transformer
                unified_out = self.unified_transformer(sequence)
                fused = unified_out[:, 0]  # Take fused query output

            else:
                fused = torch.cat([text_features, vision_features], dim=-1)

        elif text_features is not None:
            # Text only - handle based on fusion type
            if self.fusion_type == 'concat':
                fused = torch.cat([text_features, torch.zeros(text_features.size(0), self.vision_hidden_size, device=text_features.device)], dim=-1)
            elif self.fusion_type in ['clip', 'coca']:
                # For contrastive models, project text only
                text_embeds = F.normalize(self.contrastive_head.text_proj(text_features), dim=-1)
                if self.fusion_type == 'clip':
                    fused = torch.cat([text_embeds, torch.zeros_like(text_embeds)], dim=-1)
                else:
                    fused = torch.cat([text_embeds, torch.zeros_like(text_embeds), text_features], dim=-1)
            else:
                fused = text_features

        elif vision_features is not None:
            # Vision only
            if self.fusion_type == 'concat':
                fused = torch.cat([torch.zeros(vision_features.size(0), self.text_hidden_size, device=vision_features.device), vision_features], dim=-1)
            elif self.fusion_type in ['clip', 'coca']:
                vision_embeds = F.normalize(self.contrastive_head.vision_proj(vision_features), dim=-1)
                if self.fusion_type == 'clip':
                    fused = torch.cat([torch.zeros_like(vision_embeds), vision_embeds], dim=-1)
                else:
                    fused = torch.cat([torch.zeros_like(vision_embeds), vision_embeds,
                                      torch.zeros(vision_features.size(0), self.text_hidden_size, device=vision_features.device)], dim=-1)
            else:
                fused = self.vision_proj(vision_features) if hasattr(self, 'vision_proj') else vision_features
        else:
            raise ValueError("At least one of text or image must be provided")

        # Classification
        logits = self.classifier(fused)

        # Apply sensor priors if available
        if sensor_priors is not None:
            prior_adjustment = self.sensor_proj(sensor_priors)
            logits = logits + 0.3 * prior_adjustment

        # Build output
        if return_features or return_contrastive:
            output = {'logits': logits, 'fused_features': fused}
            if contrastive_outputs is not None:
                output['contrastive'] = contrastive_outputs
            if attn_weights is not None:
                output['attention_weights'] = attn_weights
            return output
        return logits


def compute_contrastive_loss(text_embeds, vision_embeds, logit_scale, labels=None):
    """
    Compute CLIP-style contrastive loss (InfoNCE).

    Args:
        text_embeds: [B, dim] normalized text embeddings
        vision_embeds: [B, dim] normalized vision embeddings
        logit_scale: scalar temperature
        labels: optional ground truth for supervised contrastive

    Returns:
        Contrastive loss value
    """
    # Compute similarity matrix
    logits_per_text = logit_scale * text_embeds @ vision_embeds.T  # [B, B]
    logits_per_image = logits_per_text.T

    # Labels are just the diagonal (matching pairs)
    batch_size = text_embeds.size(0)
    target = torch.arange(batch_size, device=text_embeds.device)

    # Symmetric cross-entropy loss
    loss_t2i = F.cross_entropy(logits_per_text, target)
    loss_i2t = F.cross_entropy(logits_per_image, target)

    return (loss_t2i + loss_i2t) / 2


class VLMFusionLoss(nn.Module):
    """
    Combined loss for VLM-style fusion models.
    Supports classification + contrastive + ITM objectives.
    """
    def __init__(self, classification_weight: float = 1.0,
                 contrastive_weight: float = 0.5,
                 itm_weight: float = 0.3,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.itm_weight = itm_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, outputs, targets, compute_contrastive=True, compute_itm=False):
        """
        Args:
            outputs: dict with 'logits' and optionally 'contrastive', 'itm_logits'
            targets: classification targets [B, num_labels]
            compute_contrastive: whether to compute contrastive loss
            compute_itm: whether to compute ITM loss
        """
        losses = {}

        # Classification loss (always)
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        losses['classification'] = self.focal_loss(logits, targets)

        # Contrastive loss (for CLIP/CoCa-style)
        if compute_contrastive and isinstance(outputs, dict) and 'contrastive' in outputs:
            cont = outputs['contrastive']
            losses['contrastive'] = compute_contrastive_loss(
                cont['text_embeds'], cont['vision_embeds'], cont['logit_scale']
            )

        # ITM loss (for BLIP-style)
        if compute_itm and isinstance(outputs, dict) and 'itm_logits' in outputs:
            # Would need ITM labels (match/no-match)
            pass

        # Weighted combination
        total_loss = self.classification_weight * losses['classification']
        if 'contrastive' in losses:
            total_loss = total_loss + self.contrastive_weight * losses['contrastive']
        if 'itm' in losses:
            total_loss = total_loss + self.itm_weight * losses['itm']

        losses['total'] = total_loss
        return losses

# ============================================================================
# 5. ADVANCED TRAINING UTILITIES
# ============================================================================
class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def client_dropout(client_models: List, dropout_prob: float = 0.1) -> List:
    """Simulate client dropout during federated learning."""
    if dropout_prob <= 0:
        return client_models

    kept = []
    for model in client_models:
        if np.random.random() > dropout_prob:
            kept.append(model)

    # Ensure at least one client remains
    if len(kept) == 0:
        kept = [client_models[np.random.randint(len(client_models))]]

    return kept

# ============================================================================
# 6. ENHANCED DATASET CLASS
# ============================================================================
class EnhancedMultiModalDataset(Dataset):
    """Enhanced dataset with sensor fusion and weak labeling."""

    def __init__(self, texts, images, labels, sources=None,
                 tokenizer=None, image_transform=None,
                 max_length=128, use_sensor_fusion=True,
                 use_weak_labels=True):

        self.texts = texts
        self.images = images
        self.labels = labels
        self.sources = sources
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.use_sensor_fusion = use_sensor_fusion
        self.use_weak_labels = use_weak_labels

        # Pre-generate sensors for consistency
        self.sensors = [simulate_sensor_summary() for _ in range(len(texts))]

    def __len__(self):
        return len(self.texts) if self.texts is not None else len(self.images)

    def __getitem__(self, idx):
        item = {}
        sensors = self.sensors[idx]

        # Process text
        if self.texts is not None and self.tokenizer is not None:
            text = str(self.texts[idx])

            # Add sensor fusion
            if self.use_sensor_fusion:
                text = fuse_sensor_with_text(text, sensors)

            encoded = self.tokenizer(
                text, max_length=self.max_length,
                padding='max_length', truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = encoded['input_ids'].squeeze(0)
            item['attention_mask'] = encoded['attention_mask'].squeeze(0)
            item['raw_text'] = text

        # Process image
        if self.images is not None:
            img = self.images[idx]
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert('RGB')
                except:
                    img = Image.new('RGB', (224, 224), color='gray')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            if self.image_transform is not None:
                item['pixel_values'] = self.image_transform(img)
            else:
                item['pixel_values'] = T.ToTensor()(img)

        # Process labels
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, list):
                label_tensor = torch.zeros(NUM_LABELS, dtype=torch.float32)
                for l in label:
                    if isinstance(l, int) and 0 <= l < NUM_LABELS:
                        label_tensor[l] = 1.0
            else:
                label_tensor = torch.tensor(label, dtype=torch.float32)

            # Enhance with weak labels if enabled
            if self.use_weak_labels and self.texts is not None:
                weak = weak_label_text(str(self.texts[idx]))
                for l in weak:
                    label_tensor[l] = max(label_tensor[l], 0.8)  # Soft label

            item['labels'] = label_tensor

        # Add sensor priors
        item['sensor_priors'] = torch.tensor(sensor_to_priors(sensors), dtype=torch.float32)

        return item

# ============================================================================
# 7. FUSION ARCHITECTURE COMPARISON - Train and compare all VLM-style models
# ============================================================================

class RealMultiModalDataset(Dataset):
    """Dataset using real HuggingFace text and image data for fusion comparison."""

    def __init__(self, texts, labels, image_dataset, tokenizer, image_processor,
                 max_seq_len=128):
        self.texts = texts
        self.labels = labels
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.num_images = len(image_dataset) if image_dataset else 0
        print(f"[Dataset] {len(texts)} texts, {self.num_images} images")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Text processing
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text, max_length=self.max_seq_len, padding='max_length',
            truncation=True, return_tensors='pt'
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
            except Exception:
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)
        else:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            pixel_values = self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': label_tensor
        }


class LightweightFusionModel(nn.Module):
    """Lightweight fusion model for faster architecture comparison."""

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

        # Vision encoder (lightweight CNN)
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
                nn.Linear(text_dim, text_dim * 4), nn.GELU(), nn.Linear(text_dim * 4, text_dim)
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
                nn.Linear(text_dim, text_dim * 4), nn.GELU(), nn.Linear(text_dim * 4, text_dim)
            )
            self.query_proj = nn.Linear(text_dim, text_dim)
        elif fusion_type == 'coca':
            fusion_dim = projection_dim * 2 + text_dim
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, projection_dim), nn.LayerNorm(projection_dim)
            )
            self.vision_proj_contrastive = nn.Sequential(
                nn.Linear(vision_dim, projection_dim), nn.LayerNorm(projection_dim)
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
                    d_model=text_dim, nhead=4, dim_feedforward=text_dim * 4,
                    dropout=0.1, batch_first=True
                ), num_layers=2
            )
        else:
            fusion_dim = text_dim + vision_dim

        self.fusion_dim = fusion_dim
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
                'text_embeds': text_embeds, 'vision_embeds': vision_embeds,
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
                'text_embeds': text_embeds, 'vision_embeds': vision_embeds,
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


def _train_fusion_epoch(model, dataloader, optimizer, fusion_type, device):
    """Train one epoch for fusion comparison."""
    from tqdm import tqdm
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

        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
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


def _evaluate_fusion(model, dataloader, fusion_type, device):
    """Evaluate fusion model."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, pixel_values)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return {
        'loss': total_loss / num_batches,
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'exact_match': np.mean(np.all(all_preds == all_labels, axis=1))
    }


def _measure_inference_time(model, dataloader, device, num_batches=10):
    """Measure average inference time."""
    import time
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


def run_fusion_comparison(
    text_source: str = "localmini",
    max_text_samples: int = 500,
    max_image_samples: int = 500,
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    fusion_types: List[str] = None,
    save_results: bool = True,
    show_plots: bool = True
):
    """
    Train and compare all VLM-style fusion architectures using real HuggingFace datasets.

    Args:
        text_source: Text data source ("localmini", "gardian", "argilla", "agnews", or comma-separated)
        max_text_samples: Maximum text samples to use
        max_image_samples: Maximum images to use
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        fusion_types: List of fusion types to compare (default: all 8)
        save_results: Whether to save results to JSON and PNG
        show_plots: Whether to display plots

    Returns:
        results: Dict with metrics for each fusion type
        histories: Dict with training histories
    """
    import time
    import json
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, random_split
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import pandas as pd

    # Import data loaders
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
        from datasets_loader import build_text_corpus_mix, load_stress_image_datasets_hf
        HAS_LOADER = True
    except ImportError:
        print("[WARNING] Could not import datasets_loader")
        HAS_LOADER = False

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    if fusion_types is None:
        fusion_types = ['concat', 'attention', 'gated', 'clip', 'flamingo', 'blip2', 'coca', 'unified_io']

    print("=" * 70)
    print("FUSION ARCHITECTURE COMPARISON (REAL DATASETS)")
    print("=" * 70)

    # Load datasets
    print("\n[1/5] Loading REAL datasets from HuggingFace...")

    if HAS_LOADER:
        print("\n--- Loading Text Data ---")
        df_text = build_text_corpus_mix(
            mix_sources=text_source,
            max_per_source=max_text_samples,
            max_samples=max_text_samples
        )
        texts = df_text['text'].tolist()
        labels = df_text['labels'].tolist()
        print(f"Loaded {len(texts)} text samples")

        print("\n--- Loading Image Data ---")
        image_dataset = load_stress_image_datasets_hf(
            max_total_images=max_image_samples,
            max_per_dataset=max_image_samples // 2
        )
        print(f"Loaded {len(image_dataset) if image_dataset else 0} images")
    else:
        print("Using fallback synthetic data...")
        texts = ["Plant showing signs of water stress and wilting."] * 100
        labels = [[0]] * 100
        image_dataset = None

    # Load tokenizer and processor
    print("\n[2/5] Loading tokenizer and image processor...")
    from transformers import AutoTokenizer, AutoImageProcessor
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Create dataset
    print("\n[3/5] Creating multimodal dataset...")
    dataset = RealMultiModalDataset(
        texts=texts, labels=labels, image_dataset=image_dataset,
        tokenizer=tokenizer, image_processor=image_processor, max_seq_len=128
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Train each fusion type
    results = {}
    training_histories = {}

    print("\n[4/5] Training each fusion architecture...")
    print("-" * 70)

    for fusion_type in fusion_types:
        print(f"\n>>> Training: {fusion_type.upper()}")

        model = LightweightFusionModel(
            fusion_type=fusion_type, text_dim=128, vision_dim=768,
            num_labels=NUM_LABELS, projection_dim=256,
            num_perceiver_latents=32, num_qformer_queries=16
        ).to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {num_params:,}")

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        best_f1 = 0

        for epoch in range(epochs):
            train_loss = _train_fusion_epoch(model, train_loader, optimizer, fusion_type, DEVICE)
            val_metrics = _evaluate_fusion(model, val_loader, fusion_type, DEVICE)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_micro'])

            if val_metrics['f1_micro'] > best_f1:
                best_f1 = val_metrics['f1_micro']

            print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1_micro']:.4f}")

        final_metrics = _evaluate_fusion(model, val_loader, fusion_type, DEVICE)
        inference_time = _measure_inference_time(model, val_loader, DEVICE)

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

    # Report
    print("\n[5/5] Generating comparison report...")
    print("=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    df = pd.DataFrame(results).T.round(4)
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
    if show_plots or save_results:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Fusion Architecture Comparison (Real Data)', fontsize=14, fontweight='bold')
        colors = plt.cm.tab10(np.linspace(0, 1, len(fusion_types)))

        # Training loss
        for i, ft in enumerate(fusion_types):
            axes[0, 0].plot(training_histories[ft]['train_loss'], label=ft, color=colors[i])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # Val F1
        for i, ft in enumerate(fusion_types):
            axes[0, 1].plot(training_histories[ft]['val_f1'], label=ft, color=colors[i])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Validation F1')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Bar charts
        f1_scores = [results[ft]['f1_micro'] for ft in fusion_types]
        axes[0, 2].bar(fusion_types, f1_scores, color=colors)
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Final F1')
        axes[0, 2].set_xticklabels(fusion_types, rotation=45, ha='right')

        inf_times = [results[ft]['inference_time_ms'] for ft in fusion_types]
        axes[1, 0].bar(fusion_types, inf_times, color=colors)
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Inference Time')
        axes[1, 0].set_xticklabels(fusion_types, rotation=45, ha='right')

        params = [results[ft]['num_params'] / 1e6 for ft in fusion_types]
        axes[1, 1].bar(fusion_types, params, color=colors)
        axes[1, 1].set_ylabel('Parameters (M)')
        axes[1, 1].set_title('Parameter Count')
        axes[1, 1].set_xticklabels(fusion_types, rotation=45, ha='right')

        # Efficiency scatter
        for i, ft in enumerate(fusion_types):
            axes[1, 2].scatter(results[ft]['inference_time_ms'], results[ft]['f1_micro'],
                             s=100, c=[colors[i]], label=ft, alpha=0.7)
        axes[1, 2].set_xlabel('Inference Time (ms)')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_title('Efficiency: F1 vs Speed')
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_results:
            plt.savefig('fusion_comparison_results.png', dpi=150, bbox_inches='tight')
            print("\nVisualization saved to: fusion_comparison_results.png")
            with open('fusion_comparison_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("Results saved to: fusion_comparison_results.json")

        if show_plots:
            plt.show()

    # Recommendations
    best_f1_type = max(results.items(), key=lambda x: x[1]['f1_micro'])[0]
    fastest_type = min(results.items(), key=lambda x: x[1]['inference_time_ms'])[0]

    print(f"""
========================================
RECOMMENDATIONS
========================================
Best F1 Score:      {best_f1_type.upper()} ({results[best_f1_type]['f1_micro']:.4f})
Fastest Inference:  {fastest_type.upper()} ({results[fastest_type]['inference_time_ms']:.2f}ms)
""")

    return results, training_histories


print("[OK] Advanced features loaded:")
print("  - Sensor fusion (IoT data simulation)")
print("  - Weak labeling (keyword-based)")
print("  - FocalLoss (class imbalance handling)")
print("  - MultiModalFusionModel with VLM-style fusion:")
print("      * concat: Simple concatenation (baseline)")
print("      * attention: Cross-attention fusion")
print("      * gated: Gated fusion with learned weights")
print("      * clip: CLIP-style contrastive + concat")
print("      * flamingo: Perceiver resampler + gated cross-attention")
print("      * blip2: Q-Former with learnable queries")
print("      * coca: Dual contrastive + captioning objectives")
print("      * unified_io: Modality tokens + shared transformer")
print("  - VLMFusionLoss (classification + contrastive + ITM)")
print("  - EMA (exponential moving average)")
print("  - Client dropout simulation")
print("  - Enhanced dataset class")
print("  - run_fusion_comparison() - Train & compare all fusion architectures")
