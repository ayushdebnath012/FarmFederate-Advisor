"""Federated adaptation sweep for the audited FarmFederate tea VLM.

This experiment starts from the validation-selected v6 multimodal checkpoint,
keeps its ResNet-50 image encoder frozen, and performs label-Dirichlet FedAvg
updates on the remaining multimodal parameters.  It is intentionally reported
as *federated adaptation*, not as federated training from scratch.

The sweep never uses the locked test partition for model selection or plots.
All round/client sensitivity values are validation macro-F1 on the same
source-image-disjoint split used by ``tea_train.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tea_train as tea  # noqa: E402


@dataclass(frozen=True)
class SweepConfig:
    checkpoint: str
    output_dir: str
    clients: List[int]
    rounds: int
    seeds: List[int]
    alpha: float
    local_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    split_seed: int


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def primary_labels(frame: pd.DataFrame) -> List[int]:
    labels: List[int] = []
    for value in frame["labels"].tolist():
        labels.append(int(value[0] if isinstance(value, list) else value))
    return labels


def prepare_split_and_cache(
    output_dir: Path,
    checkpoint: Path,
    split_seed: int,
    batch_size: int,
    device: torch.device,
):
    """Build the audited split and cache frozen spatial ResNet features."""

    cache_path = output_dir / "v6_split42_frozen_vision_cache.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_ds = tea.CachedMultiModalDataset(
            payload["train"]["input_ids"],
            payload["train"]["attention_mask"],
            payload["train"]["labels_tensor"],
            payload["train"]["vision_feature_maps"],
            payload["train"]["primary_labels"],
            payload["train"]["pairing_coverage"],
        )
        val_ds = tea.CachedMultiModalDataset(
            payload["validation"]["input_ids"],
            payload["validation"]["attention_mask"],
            payload["validation"]["labels_tensor"],
            payload["validation"]["vision_feature_maps"],
            payload["validation"]["primary_labels"],
            payload["validation"]["pairing_coverage"],
        )
        return train_ds, val_ds, payload["audit"]

    data_dir = ROOT / "Real Dataset"
    annotation_path = ROOT / "tea_results" / "annotation" / "annotations.csv"
    full_ds = tea.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        transform=tea.get_transforms(train=False),
        crop_padding=0.10,
    )
    train_idx, val_idx, test_idx = tea.grouped_train_val_test_split(
        full_ds.labels,
        full_ds.groups,
        val_split=0.20,
        test_split=0.20,
        seed=split_seed,
    )

    text_df = tea.load_annotations_csv(str(annotation_path))
    row_keys = list(zip(text_df["image_file"], text_df["box_idx"]))
    split_keys = {
        "train": {full_ds.sample_ids[i] for i in train_idx},
        "validation": {full_ds.sample_ids[i] for i in val_idx},
        "test": {full_ds.sample_ids[i] for i in test_idx},
    }
    frames = {}
    for split_name, keys in split_keys.items():
        mask = pd.Series([key in keys for key in row_keys])
        frames[split_name] = text_df.loc[mask].reset_index(drop=True)
        if frames[split_name].empty:
            raise RuntimeError(f"No exact annotations found for {split_name}")

    blocked_tokens = tea.fit_label_leakage_vocabulary(frames["train"])
    for split_name in frames:
        frames[split_name] = tea.sanitize_annotation_text(
            frames[split_name], blocked_tokens
        )

    train_obb = tea.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        transform=tea.get_transforms(train=True),
        crop_padding=0.10,
        indices=train_idx,
    )
    val_obb = tea.TeaOBBDataset(
        str(data_dir / "images"),
        str(data_dir / "labels"),
        transform=tea.get_transforms(train=False),
        crop_padding=0.10,
        indices=val_idx,
    )
    train_mm = tea.MultiModalDataset(
        train_obb, frames["train"], max_length=128, seed=split_seed
    )
    val_mm = tea.MultiModalDataset(
        val_obb, frames["validation"], max_length=128, seed=split_seed
    )

    class_weights = tea.compute_class_weights(train_obb.labels, device)
    model = build_model(class_weights, checkpoint, device)
    seed_everything(split_seed)
    train_cached = tea.cache_frozen_vision_features(
        train_mm,
        model,
        device,
        batch_size=batch_size,
        num_views=1,
        seed=split_seed,
    )
    val_cached = tea.cache_frozen_vision_features(
        val_mm,
        model,
        device,
        batch_size=batch_size,
        num_views=1,
        seed=split_seed + 10_000,
    )

    train_groups = {full_ds.groups[i] for i in train_idx}
    val_groups = {full_ds.groups[i] for i in val_idx}
    test_groups = {full_ds.groups[i] for i in test_idx}
    audit = {
        "split_seed": split_seed,
        "train_crops": len(train_idx),
        "validation_crops": len(val_idx),
        "locked_test_crops_not_used": len(test_idx),
        "train_source_images": len(train_groups),
        "validation_source_images": len(val_groups),
        "locked_test_source_images_not_used": len(test_groups),
        "pairwise_source_overlap": int(
            len(train_groups & val_groups)
            + len(train_groups & test_groups)
            + len(val_groups & test_groups)
        ),
        "train_pairing_coverage": train_mm.pairing_coverage,
        "validation_pairing_coverage": val_mm.pairing_coverage,
        "masked_target_shortcut_tokens": len(blocked_tokens),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    payload = {
        "train": cache_payload(train_cached),
        "validation": cache_payload(val_cached),
        "audit": audit,
    }
    torch.save(payload, cache_path)
    return train_cached, val_cached, audit


def cache_payload(dataset: tea.CachedMultiModalDataset) -> Dict:
    return {
        "input_ids": dataset.input_ids,
        "attention_mask": dataset.attention_mask,
        "labels_tensor": dataset.labels_tensor,
        "vision_feature_maps": dataset.vision_feature_maps,
        "primary_labels": dataset.labels,
        "pairing_coverage": dataset.pairing_coverage,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(
    class_weights: torch.Tensor,
    checkpoint: Path,
    device: torch.device,
) -> tea.MultiModalClassifier:
    model = tea.MultiModalClassifier(
        num_labels=tea.NUM_CLASSES,
        class_weights=class_weights,
        max_seq_len=128,
        modality_dropout=0.20,
        image_only_probability=0.50,
        text_only_probability=0.10,
        vision_backbone="resnet50",
        pretrained_vision=False,
        freeze_vision_backbone=True,
        finetune_vision_last_stage=False,
        text_auxiliary_weight=0.20,
        vision_auxiliary_weight=1.25,
        alignment_weight=0.05,
        text_confidence_guard=None,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state"], strict=True)
    return model


def label_dirichlet_split(
    labels: Sequence[int], num_clients: int, alpha: float, seed: int
) -> List[List[int]]:
    """Per-class Dirichlet partition with exact sample conservation."""

    rng = np.random.RandomState(seed)
    labels_array = np.asarray(labels, dtype=np.int64)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for class_id in range(tea.NUM_CLASSES):
        indices = np.where(labels_array == class_id)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(indices, cuts)):
            client_indices[client_id].extend(part.tolist())
    for indices in client_indices:
        rng.shuffle(indices)
    flattened = sorted(index for part in client_indices for index in part)
    if flattened != list(range(len(labels))):
        raise RuntimeError("Dirichlet partition lost or duplicated samples")
    return client_indices


def client_partition_audit(
    labels: Sequence[int], client_indices: Sequence[Sequence[int]]
) -> Dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    global_dist = np.bincount(
        labels_array, minlength=tea.NUM_CLASSES
    ) / len(labels_array)
    total_variation = []
    rows = []
    for client_id, indices in enumerate(client_indices):
        counts = np.bincount(
            labels_array[np.asarray(indices, dtype=np.int64)],
            minlength=tea.NUM_CLASSES,
        ) if indices else np.zeros(tea.NUM_CLASSES, dtype=int)
        if len(indices):
            distribution = counts / len(indices)
            total_variation.append(
                float(0.5 * np.abs(distribution - global_dist).sum())
            )
        rows.append(
            {
                "client": client_id,
                "samples": len(indices),
                "class_counts": counts.astype(int).tolist(),
            }
        )
    sizes = np.asarray([len(indices) for indices in client_indices], dtype=float)
    return {
        "clients": rows,
        "mean_label_total_variation": (
            float(np.mean(total_variation)) if total_variation else 0.0
        ),
        "size_coefficient_of_variation": (
            float(np.std(sizes) / np.mean(sizes)) if np.mean(sizes) else 0.0
        ),
        "empty_clients": int(sum(len(indices) == 0 for indices in client_indices)),
    }


def compact_metrics(metrics: Dict) -> Dict[str, float]:
    return {
        key: float(metrics[key])
        for key in ("f1_macro", "accuracy", "nll", "ece", "mean_confidence")
        if key in metrics
    }


def evaluate_model(
    model: tea.MultiModalClassifier,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return compact_metrics(tea.evaluate(model, loader, device, "multimodal"))


def federated_adapt(
    base_checkpoint: Path,
    train_dataset: tea.CachedMultiModalDataset,
    validation_dataset: tea.CachedMultiModalDataset,
    class_weights: torch.Tensor,
    num_clients: int,
    rounds: int,
    alpha: float,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Dict:
    seed_everything(seed)
    client_indices = label_dirichlet_split(
        train_dataset.labels, num_clients, alpha, seed
    )
    nonempty = [indices for indices in client_indices if indices]
    if not nonempty:
        raise RuntimeError("Every client shard is empty")

    global_model = build_model(class_weights, base_checkpoint, device)
    local_model = build_model(class_weights, base_checkpoint, device)
    base_metrics = evaluate_model(
        global_model, validation_dataset, batch_size, device
    )
    history = []

    # The frozen ResNet is identical on every client, so it is neither trained
    # nor redundantly accumulated. All multimodal projection, attention,
    # reliability, expert, and classifier states are aggregated.
    aggregate_keys = [
        key for key in global_model.state_dict().keys()
        if not key.startswith("v_enc.")
    ]

    for round_index in range(rounds):
        round_start = time.perf_counter()
        global_state = global_model.state_dict()
        total_samples = sum(len(indices) for indices in nonempty)
        largest_client = max(nonempty, key=len)
        aggregate: Dict[str, torch.Tensor] = {}

        for client_position, indices in enumerate(nonempty):
            local_model.load_state_dict(global_state, strict=True)
            local_model.train()
            generator = torch.Generator()
            generator.manual_seed(seed * 100_000 + round_index * 1_000 + client_position)
            loader = DataLoader(
                IndexedSubset(train_dataset, indices),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
            optimizer = torch.optim.AdamW(
                [parameter for parameter in local_model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            for local_epoch in range(local_epochs):
                seed_everything(
                    seed * 1_000_000
                    + round_index * 10_000
                    + client_position * 100
                    + local_epoch
                )
                for batch in loader:
                    batch = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    output = local_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        vision_feature_map=batch["vision_feature_map"],
                        labels=batch["labels"],
                    )
                    output["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in local_model.parameters()
                            if parameter.requires_grad
                        ],
                        1.0,
                    )
                    optimizer.step()

            local_state = local_model.state_dict()
            weight = len(indices) / total_samples
            for key in aggregate_keys:
                value = local_state[key].detach().cpu()
                if torch.is_floating_point(value):
                    if key not in aggregate:
                        aggregate[key] = torch.zeros_like(
                            value, dtype=torch.float32
                        )
                    aggregate[key].add_(value.float(), alpha=weight)
                elif indices is largest_client:
                    aggregate[key] = value.clone()

        updated_state = copy.copy(global_state)
        for key, value in aggregate.items():
            updated_state[key] = value.to(
                device=global_state[key].device,
                dtype=global_state[key].dtype,
            )
        global_model.load_state_dict(updated_state, strict=True)
        metrics = evaluate_model(
            global_model, validation_dataset, batch_size, device
        )
        metrics.update(
            {
                "round": round_index + 1,
                "elapsed_seconds": time.perf_counter() - round_start,
            }
        )
        history.append(metrics)
        print(
            f"K={num_clients:>2} seed={seed} round={round_index + 1:>2}/{rounds} "
            f"val macro-F1={metrics['f1_macro']:.4f} "
            f"accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )

    best_round = max(history, key=lambda row: row["f1_macro"])
    return {
        "num_clients": num_clients,
        "seed": seed,
        "alpha": alpha,
        "local_epochs": local_epochs,
        "baseline_validation": base_metrics,
        "history": history,
        "final_validation": history[-1],
        "best_validation": best_round,
        "partition": client_partition_audit(
            train_dataset.labels, client_indices
        ),
    }


def summarize_runs(runs: Sequence[Dict], clients: Sequence[int]) -> Dict:
    summary = {}
    for num_clients in clients:
        selected = [run for run in runs if run["num_clients"] == num_clients]
        if not selected:
            continue
        round_count = len(selected[0]["history"])
        round_means = []
        round_stds = []
        for round_index in range(round_count):
            values = [
                run["history"][round_index]["f1_macro"] for run in selected
            ]
            round_means.append(float(np.mean(values)))
            round_stds.append(float(np.std(values, ddof=0)))
        final_values = [
            run["final_validation"]["f1_macro"] for run in selected
        ]
        best_values = [
            run["best_validation"]["f1_macro"] for run in selected
        ]
        summary[str(num_clients)] = {
            "round_mean_macro_f1": round_means,
            "round_std_macro_f1": round_stds,
            "final_macro_f1_mean": float(np.mean(final_values)),
            "final_macro_f1_std": float(np.std(final_values, ddof=0)),
            "best_macro_f1_mean": float(np.mean(best_values)),
            "best_macro_f1_std": float(np.std(best_values, ddof=0)),
            "mean_label_total_variation": float(
                np.mean(
                    [
                        run["partition"]["mean_label_total_variation"]
                        for run in selected
                    ]
                )
            ),
        }
    return summary


def write_csv(output_dir: Path, runs: Sequence[Dict]) -> None:
    path = output_dir / "round_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "num_clients",
                "seed",
                "round",
                "macro_f1",
                "accuracy",
                "nll",
                "ece",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for run in runs:
            for row in run["history"]:
                writer.writerow(
                    {
                        "num_clients": run["num_clients"],
                        "seed": run["seed"],
                        "round": row["round"],
                        "macro_f1": row["f1_macro"],
                        "accuracy": row["accuracy"],
                        "nll": row.get("nll"),
                        "ece": row.get("ece"),
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                )


def write_report(output_dir: Path, payload: Dict) -> None:
    lines = [
        "# Tea VLM Federated-Adaptation Sweep",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        "- Evidence type: validation-only sensitivity analysis",
        "- Initialization: validation-selected centralized v6 checkpoint",
        "- Aggregation: full-participation, sample-weighted FedAvg",
        "- Client split: per-class Dirichlet label skew",
        "- Frozen component: ResNet-50 visual backbone",
        "- Locked test: not used for selection or plotted sweep values",
        "",
        "## Source-group audit",
        "",
    ]
    for key, value in payload["data_audit"].items():
        lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    lines.extend(
        [
            "",
            "## Validation macro-F1",
            "",
            "| Clients | Final mean | Best mean | Mean label TV |",
            "|---:|---:|---:|---:|",
        ]
    )
    for clients, row in payload["summary"].items():
        lines.append(
            f"| {clients} | {row['final_macro_f1_mean']:.4f} | "
            f"{row['best_macro_f1_mean']:.4f} | "
            f"{row['mean_label_total_variation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "These values measure federated adaptation of one already selected "
            "checkpoint on the validation partition. They do not constitute "
            "federated training from scratch, a multi-estate privacy trial, or "
            "new locked-test evidence.",
            "",
        ]
    )
    (output_dir / "FEDERATED_ADAPTATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_int_list(value: str) -> List[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected a comma-separated positive integer list")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "tea_results"
            / "multimodal_v6_vision_full_20260727"
            / "models"
            / "best_vlm.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "tea_results" / "federated_adaptation_v6"),
    )
    parser.add_argument("--clients", type=parse_int_list, default=parse_int_list("2,3,5,8"))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if args.rounds <= 0 or args.local_epochs <= 0:
        raise ValueError("rounds and local-epochs must be positive")

    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_dataset, validation_dataset, audit = prepare_split_and_cache(
        output_dir,
        checkpoint,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        device=device,
    )
    class_weights = tea.compute_class_weights(train_dataset.labels, device)

    config = SweepConfig(
        checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        clients=args.clients,
        rounds=args.rounds,
        seeds=args.seeds,
        alpha=args.alpha,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        split_seed=args.split_seed,
    )
    runs = []
    start = time.perf_counter()
    for clients in args.clients:
        for seed in args.seeds:
            runs.append(
                federated_adapt(
                    checkpoint,
                    train_dataset,
                    validation_dataset,
                    class_weights,
                    num_clients=clients,
                    rounds=args.rounds,
                    alpha=args.alpha,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=seed,
                    device=device,
                )
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "validation_only_federated_adaptation",
        "config": asdict(config),
        "device": str(device),
        "data_audit": audit,
        "runs": runs,
        "summary": summarize_runs(runs, args.clients),
        "wall_time_seconds": time.perf_counter() - start,
    }
    with (output_dir / "federated_adaptation_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    write_csv(output_dir, runs)
    write_report(output_dir, payload)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
