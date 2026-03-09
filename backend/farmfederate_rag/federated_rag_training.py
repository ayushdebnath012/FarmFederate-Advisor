#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
federated_rag_training.py — Federated training extensions for FarmFederate-RAG.

Components:
  RetrieverContrastiveLoss — InfoNCE loss for retriever alignment
  AdvisoryGenerator        — On-device relevance scorer (BCE)
  FedRAGConfig             — Dataclass of all training hyperparams
  FedRAGClient             — Per-farm training: 3-phase joint loss
  FedRAGServer             — FedAvg aggregation with EMA for retriever
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetrieverContrastiveLoss  (InfoNCE)
# ---------------------------------------------------------------------------

class RetrieverContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for the dual-encoder retriever.

    Given a batch of (query_emb, pos_doc_emb) pairs, treats all other docs in
    the batch as negatives.  Temperature τ is learnable (started at 0.07).

    L = -mean( log( exp(q·d+ / τ) / Σ_j exp(q·d_j / τ) ) )
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def forward(
        self,
        query_emb: torch.Tensor,   # [B, D] L2-normalised
        pos_doc_emb: torch.Tensor, # [B, D] L2-normalised
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE: each query-doc pair is a positive for both sides.
        Returns scalar loss.
        """
        B = query_emb.size(0)
        # similarity matrix [B, B]
        sim = torch.matmul(query_emb, pos_doc_emb.T) / self.temperature
        labels = torch.arange(B, device=query_emb.device)
        # both directions
        loss_q = F.cross_entropy(sim, labels)
        loss_d = F.cross_entropy(sim.T, labels)
        return (loss_q + loss_d) / 2.0


# ---------------------------------------------------------------------------
# AdvisoryGenerator
# ---------------------------------------------------------------------------

class AdvisoryGenerator(nn.Module):
    """
    On-device advisory relevance scorer.

    Given a context embedding (query_emb [B, 128]) and a candidate document
    embedding (doc_emb [B, 128]), predicts a relevance score in [0, 1].

    Trained with BCE against binary relevance targets.
    At inference, top-scoring retrieved documents are surfaced to the farmer.
    """

    def __init__(self, emb_dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        # Interaction features: element-wise product + difference
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        query_emb: torch.Tensor,  # [B, D]
        doc_emb: torch.Tensor,    # [B, D]
    ) -> torch.Tensor:
        """Returns relevance logits [B, 1]."""
        prod = query_emb * doc_emb
        diff = (query_emb - doc_emb).abs()
        x = torch.cat([query_emb, doc_emb, prod, diff], dim=-1)  # [B, 4D]
        return self.scorer(x)  # [B, 1]

    def compute_loss(
        self,
        query_emb: torch.Tensor,  # [B, D]
        doc_emb: torch.Tensor,    # [B, D]
        targets: torch.Tensor,    # [B, 1] float in {0, 1}
    ) -> torch.Tensor:
        logits = self.forward(query_emb, doc_emb)
        return F.binary_cross_entropy_with_logits(logits, targets.float())

    @torch.no_grad()
    def score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Returns relevance probability [B, 1] (sigmoid of logits)."""
        return torch.sigmoid(self.forward(query_emb, doc_emb))


# ---------------------------------------------------------------------------
# FedRAGConfig
# ---------------------------------------------------------------------------

@dataclass
class FedRAGConfig:
    """All training hyperparameters for federated RAG training."""

    # Federated rounds
    num_rounds: int = 8
    local_epochs: int = 3
    clients_per_round: int = -1    # -1 = all clients

    # Retriever warmup: classification-only for first N rounds
    retriever_warmup_rounds: int = 2

    # Joint loss weights
    lambda_cls: float = 0.5
    lambda_ret: float = 0.3
    lambda_gen: float = 0.2

    # Retriever EMA (server-side slow update)
    retriever_ema_mu: float = 0.9

    # Optimiser
    lr: float = 1e-4
    lr_retriever: float = 1e-4
    lr_query_builder: float = 1e-4
    lr_generator: float = 1e-4
    weight_decay: float = 0.01

    # RAG retrieval
    top_k_docs: int = 5

    # Differential privacy (optional)
    dp_noise_scale: float = 0.0   # 0 = disabled
    dp_clip_norm: float = 1.0

    # Misc
    device: str = "cpu"
    seed: int = 42

    # Extra per-field metadata (ignored by dataclass machinery)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FedRAGClient
# ---------------------------------------------------------------------------

class FedRAGClient:
    """
    Per-farm federated RAG client.

    Wraps:
      - classification_model  : MultiModalModel (existing FarmFederate model)
      - retriever             : RetrieverEncoder
      - query_builder         : RAGQueryBuilder
      - generator             : AdvisoryGenerator
      - vector_store          : FarmVectorStore   (local KB, never transmitted)

    Training produces a joint loss:
        L = λ_cls·L_cls + λ_ret·L_ret + λ_gen·L_gen

    get_state_dicts() returns the four transmittable component states.
    load_state_dicts() restores from aggregated server states.
    """

    def __init__(
        self,
        farm_id: int,
        classification_model: nn.Module,
        retriever: nn.Module,
        query_builder: nn.Module,
        generator: nn.Module,
        vector_store,          # FarmVectorStore — type hint avoided to prevent circular import
        config: FedRAGConfig,
        contrastive_loss: Optional[nn.Module] = None,
    ):
        self.farm_id = farm_id
        self.cls_model = classification_model
        self.retriever = retriever
        self.query_builder = query_builder
        self.generator = generator
        self.vector_store = vector_store
        self.config = config
        self.contrastive_loss = contrastive_loss or RetrieverContrastiveLoss()
        self._round = 0  # current FL round (set by server)

    # ------------------------------------------------------------------ util

    def _make_optimizer(self) -> torch.optim.Optimizer:
        param_groups = [
            {"params": self.cls_model.parameters(),     "lr": self.config.lr},
            {"params": self.retriever.parameters(),     "lr": self.config.lr_retriever},
            {"params": self.query_builder.parameters(), "lr": self.config.lr_query_builder},
            {"params": self.generator.parameters(),     "lr": self.config.lr_generator},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    # ------------------------------------------------------------------ train

    def train_round(
        self,
        dataloader: DataLoader,
        focal_loss_fn: nn.Module,
        round_idx: int,
        doc_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Run `local_epochs` of joint training.

        Args:
            dataloader:   yields {input_ids, attention_mask, pixel_values, labels, raw_text}
            focal_loss_fn: FocalLoss instance for classification
            round_idx:    current FL round (controls retriever warmup)
            doc_batch:    optional {doc_ids, doc_mask, relevance_targets [B,1]}
                          used for contrastive + generator loss

        Returns:
            metrics dict with average losses over this round
        """
        self._round = round_idx
        retriever_active = round_idx >= self.config.retriever_warmup_rounds
        device = self.config.device

        self.cls_model.to(device).train()
        self.retriever.to(device).train()
        self.query_builder.to(device).train()
        self.generator.to(device).train()

        opt = self._make_optimizer()

        total_cls_loss = 0.0
        total_ret_loss = 0.0
        total_gen_loss = 0.0
        steps = 0

        for _epoch in range(self.config.local_epochs):
            for batch in dataloader:
                input_ids     = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values  = batch["pixel_values"].to(device)
                labels        = batch["labels"].to(device)

                opt.zero_grad(set_to_none=True)

                # --- Phase 1: classification ---
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    out = self.cls_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        return_features=True,
                    )
                    logits    = out.logits         # [B, 5]
                    h_f       = out.fused_features  # [B, 512]
                    l_cls     = focal_loss_fn(logits, labels)

                    # --- Phase 2: retriever (after warmup) ---
                    l_ret = torch.zeros(1, device=device)
                    l_gen = torch.zeros(1, device=device)

                    if retriever_active and doc_batch is not None:
                        doc_ids  = doc_batch["doc_ids"].to(device)
                        doc_mask = doc_batch["doc_mask"].to(device)

                        # query embedding from multimodal features
                        class_probs = torch.softmax(logits.detach(), dim=-1)
                        iot_vals    = doc_batch.get("iot_vals")
                        if iot_vals is not None:
                            iot_vals = iot_vals.to(device)
                        query_emb = self.query_builder(h_f.detach(), class_probs, iot_vals)

                        # doc embedding from retriever
                        ret_out  = self.retriever(
                            query_ids=input_ids,
                            query_mask=attention_mask,
                            doc_ids=doc_ids,
                            doc_mask=doc_mask,
                        )
                        doc_emb = ret_out["doc_emb"]   # [B, 128]

                        l_ret = self.contrastive_loss(query_emb, doc_emb)

                        # --- Phase 3: generator ---
                        rel_targets = doc_batch.get("relevance_targets")
                        if rel_targets is not None:
                            rel_targets = rel_targets.to(device)
                            l_gen = self.generator.compute_loss(query_emb, doc_emb, rel_targets)

                    # joint loss
                    loss = (
                        self.config.lambda_cls * l_cls
                        + self.config.lambda_ret * l_ret
                        + self.config.lambda_gen * l_gen
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.cls_model.parameters())
                    + list(self.retriever.parameters())
                    + list(self.query_builder.parameters())
                    + list(self.generator.parameters()),
                    1.0,
                )
                opt.step()

                total_cls_loss += l_cls.item()
                total_ret_loss += (l_ret.item() if retriever_active else 0.0)
                total_gen_loss += (l_gen.item() if retriever_active else 0.0)
                steps += 1

        n = max(1, steps)
        return {
            "cls_loss": total_cls_loss / n,
            "ret_loss": total_ret_loss / n,
            "gen_loss": total_gen_loss / n,
            "total_loss": (
                self.config.lambda_cls * total_cls_loss / n
                + self.config.lambda_ret * total_ret_loss / n
                + self.config.lambda_gen * total_gen_loss / n
            ),
        }

    # ------------------------------------------------------------------ state

    def get_state_dicts(self) -> Dict[str, Dict]:
        """Returns all transmittable component state dicts (CPU)."""
        return {
            "classification": {
                k: v.detach().cpu() for k, v in self.cls_model.state_dict().items()
            },
            "retriever": {
                k: v.detach().cpu() for k, v in self.retriever.state_dict().items()
            },
            "query_builder": {
                k: v.detach().cpu() for k, v in self.query_builder.state_dict().items()
            },
            "generator": {
                k: v.detach().cpu() for k, v in self.generator.state_dict().items()
            },
        }

    def load_state_dicts(self, states: Dict[str, Dict]) -> None:
        """Loads aggregated global state dicts from server."""
        if "classification" in states:
            self.cls_model.load_state_dict(states["classification"], strict=False)
        if "retriever" in states:
            self.retriever.load_state_dict(states["retriever"], strict=False)
        if "query_builder" in states:
            self.query_builder.load_state_dict(states["query_builder"], strict=False)
        if "generator" in states:
            self.generator.load_state_dict(states["generator"], strict=False)

    @property
    def num_samples(self) -> int:
        """Override in subclass or set externally for weighted FedAvg."""
        return getattr(self, "_num_samples", 1)

    @num_samples.setter
    def num_samples(self, v: int) -> None:
        self._num_samples = v


# ---------------------------------------------------------------------------
# FedRAGServer
# ---------------------------------------------------------------------------

class FedRAGServer:
    """
    Federated aggregation server with EMA update for the retriever.

    Aggregation strategy:
      - classification, query_builder, generator : standard weighted FedAvg
      - retriever                                : EMA  θ_ret = μ·θ_old + (1-μ)·θ_fedavg
        (prevents embedding drift that would invalidate per-farm FAISS indices)

    Optional differential privacy noise can be injected before aggregation.
    """

    def __init__(
        self,
        global_cls_model: nn.Module,
        global_retriever: nn.Module,
        global_query_builder: nn.Module,
        global_generator: nn.Module,
        config: FedRAGConfig,
    ):
        self.cls_model     = global_cls_model
        self.retriever     = global_retriever
        self.query_builder = global_query_builder
        self.generator     = global_generator
        self.config        = config
        self._round        = 0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ aggregation

    @staticmethod
    def _fedavg(
        client_states: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Weighted FedAvg over a single component's state dicts."""
        total = float(sum(sample_counts))
        weights = [n / total for n in sample_counts]
        aggregated: Dict[str, torch.Tensor] = {}
        for key in client_states[0]:
            aggregated[key] = sum(
                st[key].float() * w for st, w in zip(client_states, weights)
            )
        return aggregated

    def _ema_update(
        self,
        global_state: Dict[str, torch.Tensor],
        fedavg_state: Dict[str, torch.Tensor],
        mu: float,
    ) -> Dict[str, torch.Tensor]:
        """EMA: θ_global = μ·θ_old + (1-μ)·θ_fedavg"""
        return {
            k: mu * global_state[k].float() + (1.0 - mu) * fedavg_state[k].float()
            for k in global_state
        }

    def _add_dp_noise(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.dp_noise_scale <= 0:
            return state
        noisy = {}
        for k, v in state.items():
            if v.is_floating_point():
                # clip
                norm = v.norm()
                v = v / max(1.0, (norm / self.config.dp_clip_norm).item())
                # noise
                noise = torch.randn_like(v) * self.config.dp_noise_scale
                noisy[k] = v + noise
            else:
                noisy[k] = v
        return noisy

    def aggregate(
        self,
        client_updates: List[Dict[str, Dict]],
        sample_counts: List[int],
    ) -> None:
        """
        Aggregates client updates into global model.

        Args:
            client_updates: list of dicts with keys
                            {classification, retriever, query_builder, generator}
            sample_counts:  number of training samples per client (for weighted avg)
        """
        if not client_updates:
            logger.warning("No client updates received; skipping aggregation.")
            return

        n = len(client_updates)
        if len(sample_counts) != n:
            sample_counts = [1] * n

        # collect per-component state dicts
        def _collect(key: str) -> List[Dict]:
            return [upd[key] for upd in client_updates if key in upd]

        # --- standard FedAvg for cls / query_builder / generator ---
        for component, model in [
            ("classification", self.cls_model),
            ("query_builder",  self.query_builder),
            ("generator",      self.generator),
        ]:
            states = _collect(component)
            if not states:
                continue
            counts = sample_counts[:len(states)]
            agg = self._fedavg(states, counts)
            if self.config.dp_noise_scale > 0:
                agg = self._add_dp_noise(agg)
            model.load_state_dict(agg, strict=False)

        # --- EMA for retriever ---
        ret_states = _collect("retriever")
        if ret_states:
            counts = sample_counts[:len(ret_states)]
            fedavg_ret = self._fedavg(ret_states, counts)
            if self.config.dp_noise_scale > 0:
                fedavg_ret = self._add_dp_noise(fedavg_ret)
            global_ret_state = {k: v.cpu() for k, v in self.retriever.state_dict().items()}
            ema_ret = self._ema_update(global_ret_state, fedavg_ret, self.config.retriever_ema_mu)
            self.retriever.load_state_dict(ema_ret, strict=False)

        self._round += 1
        logger.info("Server aggregation round %d complete.", self._round)

    def get_global_state_dicts(self) -> Dict[str, Dict]:
        """Returns all global component states (for broadcasting to clients)."""
        return {
            "classification": {k: v.cpu() for k, v in self.cls_model.state_dict().items()},
            "retriever":      {k: v.cpu() for k, v in self.retriever.state_dict().items()},
            "query_builder":  {k: v.cpu() for k, v in self.query_builder.state_dict().items()},
            "generator":      {k: v.cpu() for k, v in self.generator.state_dict().items()},
        }

    # ------------------------------------------------------------------ full training loop

    def run_federated_training(
        self,
        clients: List[FedRAGClient],
        dataloaders: List[DataLoader],
        focal_loss_fn: nn.Module,
        doc_batches: Optional[List[Optional[Dict]]] = None,
    ) -> Dict[str, Any]:
        """
        Full federated training loop.

        Args:
            clients:       list of FedRAGClient
            dataloaders:   one DataLoader per client
            focal_loss_fn: shared FocalLoss for classification
            doc_batches:   optional per-client document batches for retrieval training

        Returns:
            results dict with per-round metrics
        """
        if doc_batches is None:
            doc_batches = [None] * len(clients)

        per_round_metrics: List[Dict] = []

        for r in range(self.config.num_rounds):
            logger.info("=== Federated Round %d / %d ===", r + 1, self.config.num_rounds)

            # client selection
            if self.config.clients_per_round > 0:
                import random
                selected = random.sample(
                    list(range(len(clients))),
                    min(self.config.clients_per_round, len(clients)),
                )
            else:
                selected = list(range(len(clients)))

            # broadcast global weights
            global_states = self.get_global_state_dicts()
            for i in selected:
                clients[i].load_state_dicts(global_states)

            # local training
            round_metrics: List[Dict] = []
            client_updates: List[Dict] = []
            sample_counts: List[int] = []

            for i in selected:
                client = clients[i]
                metrics = client.train_round(
                    dataloader=dataloaders[i],
                    focal_loss_fn=focal_loss_fn,
                    round_idx=r,
                    doc_batch=doc_batches[i],
                )
                round_metrics.append(metrics)
                client_updates.append(client.get_state_dicts())
                sample_counts.append(client.num_samples)
                logger.debug(
                    "  Client %d: cls=%.4f ret=%.4f gen=%.4f",
                    client.farm_id, metrics["cls_loss"],
                    metrics["ret_loss"], metrics["gen_loss"],
                )

            # aggregate
            self.aggregate(client_updates, sample_counts)

            # round summary
            avg_metrics = {
                k: float(np.mean([m[k] for m in round_metrics]))
                for k in round_metrics[0]
            }
            avg_metrics["round"] = r + 1
            per_round_metrics.append(avg_metrics)
            logger.info(
                "  Round %d avg: cls=%.4f ret=%.4f gen=%.4f total=%.4f",
                r + 1, avg_metrics["cls_loss"], avg_metrics["ret_loss"],
                avg_metrics["gen_loss"], avg_metrics["total_loss"],
            )

        return {
            "num_rounds": self.config.num_rounds,
            "per_round_metrics": per_round_metrics,
            "final_global_states": self.get_global_state_dicts(),
        }
