"""
FarmFederate reviewer ablation suite (E1-E8).

This is the FarmFederate counterpart of the OmniMed-FL suite. It answers the
same reviewer asks -- quantify each component, report cost/scalability, and
compare against other federated methods under matched settings -- but binds to
FarmFederate's own functions rather than reimplementing the method. Nothing
here invents a new training procedure: models, losses, local training and
FedAvg all come from backend/FarmFederate_Colab_Complete.py.

WHAT THE REPO ALREADY HAS (not duplicated here)
  * run_ablation_study()  -- 2x2 diversity x balanced-sampler, but text-only,
    centralized, and at a single partition. E3 below is the federated,
    multimodal, two-alpha version of that question.
  * run_multi_seed_test() -- 3 seeds, text-only. E5 covers fusion x seeds.
  * run_federated_sweep() -- rounds/clients/alpha grid. E1/E2 keep the sweep
    but add the measurement that makes it interpretable (see below).
  * experiments/advisory_retrieval_eval.py -- retrieval evaluation. That is the
    E7 slot and it is NOT re-implemented here; run that script instead.

THE ONE THING TO UNDERSTAND BEFORE READING RESULTS
  split_data_non_iid() has two behaviours. Called WITHOUT `labels` it performs
  the legacy split: it shuffles and cuts contiguous Dirichlet-sized chunks, so
  alpha perturbs client shard SIZE and barely touches label mix. Called WITH
  `labels` it allocates each class independently -- a real label-skew split.

  federated_train() at its default call site still uses the legacy path
  (it calls the splitter without labels). So a sweep run through the default
  path varies alpha while holding the label distribution nearly IID, and the
  resulting "robustness to non-IID" curve is close to flat by construction.

  Every partition in this suite therefore passes `labels=` and `groups=`
  explicitly, and E1 additionally runs the legacy splitter as a control and
  reports the realised label skew (total-variation distance from the global
  class distribution) for both. That number, not the alpha label, is what makes
  the non-IID claim checkable.

USAGE (Colab)
    import farm_ablation as fa
    fa.main(base_py="/content/FarmFederate_Colab_Complete.py",
            tier="smoke", out="/content/farm_results.json")

Resumable: every result is keyed and flushed to the JSON as it completes, so
re-running the same command skips finished work. `only=["E1","E2"]` restricts
the run to one chunk, which is how a multi-hour job survives Colab timeouts.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
except ImportError:  # pragma: no cover - the suite cannot run without torch
    raise SystemExit("PyTorch is required. On Colab: Runtime -> Change runtime type -> GPU.")


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------
# smoke  -- validates every code path cheaply. Results are NOT publishable.
# standard -- the numbers you put in the paper.
# full   -- adds a third seed; expect multiple sessions.
# ---------------------------------------------------------------------------
# architectures compared under federation
# ---------------------------------------------------------------------------
# The federated sweeps used to run one fixed architecture (concat VLM), so a
# round-count or alpha result said nothing about whether the finding held for
# any other model. Every federated section now runs each architecture below on
# the identical client partition, so differences are attributable to the
# architecture rather than to a different split.
#
# `kwargs` are resolved lazily against the loaded base module because the model
# classes only exist after load_base().
def build_archs(mf, num_classes: int, names=None):
    """Return {arch_name: (model_class, kwargs, model_type)}.

    Unimodal entries ignore the modality they do not consume; they read the
    same paired rows, so the held-out support is identical across every arch.
    """
    catalogue = {
        "image_only": (
            mf.LightweightVisionClassifier,
            dict(num_labels=num_classes),
            "vision",
        ),
        "text_only": (
            mf.LightweightTextClassifier,
            dict(num_labels=num_classes),
            "text",
        ),
        "concat_vlm": (
            mf.MultiModalClassifier,
            dict(num_labels=num_classes, fusion_type="concat"),
            "multimodal",
        ),
        "attention_vlm": (
            mf.MultiModalClassifier,
            dict(num_labels=num_classes, fusion_type="attention"),
            "multimodal",
        ),
        # The paper's proposed family. modality_dropout is disabled here so the
        # comparison isolates the fusion mechanism instead of also handing this
        # one architecture a regulariser the others do not get.
        "cross_attention_vlm": (
            getattr(mf, "FlexibleCrossAttentionVLM", None),
            dict(num_labels=num_classes, modality_dropout=0.0),
            "multimodal",
        ),
    }
    if names is None:
        names = list(catalogue)
    out = {}
    for name in names:
        entry = catalogue.get(name)
        if entry is None or entry[0] is None:
            print(f"  [arch] {name} unavailable in base module; skipped")
            continue
        out[name] = entry
    return out


TIERS = {
    "smoke": dict(
        text_per_class=40, image_fill_target=30, batch_size=8,
        fed_rounds=2, local_epochs=1, central_epochs=2,
        seeds=[0], fusion_seeds=[0],
        alphas=[0.1, 1.0], client_counts=[2, 3],
        # two architectures is enough to prove the per-arch loop works
        fed_archs=["concat_vlm", "text_only"],
    ),
    "standard": dict(
        text_per_class=600, image_fill_target=200, batch_size=16,
        fed_rounds=50, local_epochs=3, central_epochs=8,
        seeds=[0, 1], fusion_seeds=[0, 1],
        alphas=[0.1, 0.5, 1.0, 10.0],
        # 260 training pairs, so K=50 leaves ~5 per client: a genuine
        # breakdown probe rather than another comfortable point.
        client_counts=[2, 3, 5, 10, 20, 50],
        fed_archs=["image_only", "text_only", "concat_vlm",
                   "attention_vlm", "cross_attention_vlm"],
    ),
    "full": dict(
        text_per_class=600, image_fill_target=200, batch_size=16,
        fed_rounds=50, local_epochs=3, central_epochs=10,
        seeds=[0, 1, 2], fusion_seeds=[0, 1, 2],
        alphas=[0.1, 0.5, 1.0, 10.0],
        client_counts=[2, 3, 5, 10, 20, 50],
        fed_archs=["image_only", "text_only", "concat_vlm",
                   "attention_vlm", "cross_attention_vlm"],
    ),
}

FUSION_TYPES = ["concat", "attention", "gated"]


# ---------------------------------------------------------------------------
# base module + result store
# ---------------------------------------------------------------------------
def load_base(base_py: str):
    """Import FarmFederate_Colab_Complete.py without firing its __main__ block."""
    base_py = str(base_py)
    if not os.path.exists(base_py):
        raise FileNotFoundError(
            f"Base module not found at {base_py}. Pass --base with the path to "
            f"backend/FarmFederate_Colab_Complete.py."
        )
    spec = importlib.util.spec_from_file_location("farmfed_base", base_py)
    mod = importlib.util.module_from_spec(spec)
    # Name is not __main__, so the auto-run guard at the bottom stays shut.
    sys.modules["farmfed_base"] = mod
    spec.loader.exec_module(mod)

    # The base module binds several module-level globals (tqdm, plt, sns,
    # AutoTokenizer, ...) inside check_imports() rather than at import time, and
    # its own training loops reference them. Skipping this call leaves
    # train_model() raising "NameError: name 'tqdm' is not defined" partway
    # through a run -- after hours of federated work has already completed.
    check = getattr(mod, "check_imports", None)
    if check is not None:
        # check_imports() drops every sys.path entry containing "FarmFederate"
        # to stop a local datasets/ folder shadowing HuggingFace's package. If
        # the virtualenv itself lives under a directory of that name -- e.g.
        # ~/FarmFederate/.gpu-venv -- that also removes its site-packages, and
        # any module not already imported becomes invisible. Importing them here
        # first puts them in sys.modules, so check_imports' own imports resolve
        # from cache regardless of what it does to sys.path.
        # Importing the top-level package is not enough for transformers, which
        # resolves AutoTokenizer/AutoModel lazily on first attribute access --
        # that access would happen inside check_imports, with sys.path already
        # stripped. Touch the attributes here to force resolution while the path
        # is still intact.
        for name, attrs in (("transformers", ("AutoTokenizer", "AutoModel",
                                              "AutoImageProcessor")),
                            ("seaborn", ()), ("tqdm", ("tqdm",)),
                            ("matplotlib.pyplot", ()), ("PIL.Image", ()),
                            ("torchvision.transforms", ()),
                            ("datasets", ("load_dataset",))):
            try:
                module = importlib.import_module(name)
                for attr in attrs:
                    getattr(module, attr)
            except Exception:
                pass
        try:
            check()
            print("  [base] check_imports() bound the module globals")
        except Exception as exc:
            print(f"  [base] check_imports() failed: {exc}\n"
                  f"         The repo's own train_model()/train_epoch() reference "
                  f"globals bound there (tqdm, plt, ...) and will raise NameError. "
                  f"Install the missing package before running E3/E4.")
    return mod


class Store:
    """Append-only, atomically-flushed result file. Makes the suite resumable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def has(self, exp: str, key: str) -> bool:
        return key in self.data.get(exp, {})

    def get(self, exp: str, key: str):
        return self.data.get(exp, {}).get(key)

    def put(self, exp: str, key: str, value):
        self.data.setdefault(exp, {})[key] = value
        self.flush()

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=float))
        tmp.replace(self.path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def resolve_data_root(explicit: Optional[str] = None) -> Path:
    """Find (and if needed extract) the portable data bundle.

    Looks for an already-extracted `data_final/` first, then falls back to
    extracting `data_final.zip`. Both layouts put images under
    `real_dataset_sorted/` and text under `text_data/annotations.csv`.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path("/content/data_final"), Path("data_final"),
                   Path("/content/FarmFederate/data_final")]
    for cand in candidates:
        if (cand / "real_dataset_sorted").is_dir() and (cand / "text_data").is_dir():
            return cand

    for zip_cand in [Path("/content/data_final.zip"), Path("data_final.zip"),
                     Path("/content/drive/MyDrive/data_final.zip")]:
        if zip_cand.is_file():
            dest = Path("/content") if Path("/content").is_dir() else Path(".")
            print(f"  [data] extracting {zip_cand} -> {dest}")
            with zipfile.ZipFile(zip_cand) as zf:
                zf.extractall(dest)
            root = dest / "data_final"
            if (root / "real_dataset_sorted").is_dir():
                return root

    raise FileNotFoundError(
        "Could not locate the data bundle. Upload data_final.zip to /content, "
        "or pass --data-root pointing at a folder containing real_dataset_sorted/ "
        "and text_data/."
    )


def _as_int_label(mf, value) -> int:
    """Normalize one label to an int class id.

    load_local_images returns per-image labels as lists (``[0]``) because a
    source photograph can in principle carry several lesion classes. The repo's
    own ``_single_label_index`` defines how that collapses to one class, so use
    it rather than inventing a second convention.
    """
    helper = getattr(mf, "_single_label_index", None)
    if helper is not None:
        try:
            return int(helper(value))
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return int(value[0])
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def build_data(mf, cfg, tier: dict, data_root: Path, cache_path: Optional[Path] = None) -> dict:
    """Build the paired multimodal dataset the whole suite trains on.

    Returns a dict carrying the dataset, the split indices, and -- importantly --
    `pairing_mode`, because how image and text rows are paired changes what a
    fusion result means.
    """
    import pandas as pd

    img_root = data_root / "real_dataset_sorted"
    txt_csv = data_root / "text_data" / "annotations.csv"

    # -- genuine sample-linked pairs (preferred) ---------------------------
    # annotations.csv + crops/<disease>/<stem>_box<NN>.jpg gives observations
    # where the note was written about the exact crop it is paired with. That
    # is the only pairing on which a fusion number means anything. Label-matched
    # stapling below is a fallback and is recorded as such in the results.
    linked = None
    linked_dirs = [
        data_root / "annotation",
        Path("tea_results/annotation"),
        Path(__file__).resolve().parent.parent / "tea_results" / "annotation",
    ]
    if hasattr(mf, "load_linked_crop_observations"):
        for candidate in linked_dirs:
            try:
                if not (Path(candidate) / "annotations.csv").is_file():
                    continue
                linked = mf.load_linked_crop_observations(
                    candidate, img_size=cfg.image_size
                )
            except Exception as exc:
                print(f"  [data] linked-pair load failed at {candidate}: {exc}")
                linked = None
            if linked is not None:
                print(f"  [data] genuine sample-linked pairs from {candidate}")
                break

    if linked is not None:
        paired_texts = list(linked["texts"])
        paired_images = list(linked["images"])
        paired_labels = [_as_int_label(mf, x) for x in linked["labels"]]
        paired_groups = [str(g) for g in linked["groups"]]
        img_groups = list(paired_groups)
        n_real_images = len(paired_images)
        pairing_mode = "genuine_box_linked_observation"
        print(f"  [data] paired samples: {len(paired_labels)} (mode={pairing_mode})")
        print(f"  [data] per-class {np.bincount(paired_labels, minlength=cfg.num_labels).tolist()}"
              f"  independent source groups: {len(set(paired_groups))}")
    else:
        print("  [data] WARNING: no genuine linked pairs found; "
              "falling back to label-matched stapling")
        if not txt_csv.is_file():
            raise FileNotFoundError(f"missing {txt_csv}")

        # -- images ------------------------------------------------------------
        # return_source_groups keeps every crop of one photograph together, so a
        # source photo can never straddle the train/val boundary.
        loaded = mf.load_local_images(
            img_root,
            max_per_class=800,
            img_size=cfg.image_size,
            seed=cfg.seed,
            return_source_groups=True,
        )
        if isinstance(loaded, tuple) and len(loaded) == 3:
            images, img_labels, img_groups = loaded
        elif isinstance(loaded, tuple) and len(loaded) == 2:
            images, img_labels = loaded
            img_groups = [f"img_{i}" for i in range(len(images))]
        else:
            raise RuntimeError(f"unexpected load_local_images return: {type(loaded)}")

        images = list(images)
        img_labels = [_as_int_label(mf, x) for x in img_labels]
        img_groups = [str(g) for g in img_groups]
        print(f"  [data] real images: {len(images)}  "
              f"per-class {np.bincount(img_labels, minlength=cfg.num_labels).tolist()}")

        n_real_images = len(images)

        # -- no image fill -----------------------------------------------------
        # Neither procedurally generated images nor materialized augmented copies
        # are added here. Both inflate the apparent dataset without adding
        # evidence, and a frozen augmented copy is replayed identically every
        # epoch. The 200 real photographs are used as-is; class balance is handled
        # by the balanced sampler, which resamples minority rows online, and each
        # draw receives a fresh random augmentation from the training Dataset
        # (see augment_images=True below). Validation and test stay deterministic.
        fill_target = int(tier.get("image_fill_target", 0))
        if fill_target > 0:
            print(f"  [data] image_fill_target={fill_target} ignored: real images only "
                  f"({n_real_images} photographs); balancing is online")

        # -- text --------------------------------------------------------------
        text_df = pd.read_csv(txt_csv)
        if not {"text", "class_id"}.issubset(text_df.columns):
            raise RuntimeError(f"{txt_csv} needs columns text,class_id")
        per_class = int(tier["text_per_class"])
        parts = []
        for class_id in range(cfg.num_labels):
            rows = text_df[text_df["class_id"] == class_id]
            if len(rows) > per_class:
                rows = rows.sample(n=per_class, random_state=cfg.seed)
            parts.append(rows)
        text_df = pd.concat(parts).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        texts = text_df["text"].astype(str).tolist()
        txt_labels = [int(x) for x in text_df["class_id"].tolist()]
        print(f"  [data] text rows: {len(texts)}  "
              f"per-class {np.bincount(txt_labels, minlength=cfg.num_labels).tolist()}")

        # -- pair them ---------------------------------------------------------
        # This bundle has no crops/ directory, so there are no genuine image-text
        # pairs: a text row and an image row are joined only by sharing a label.
        # That is a real limitation of what fusion numbers here can claim, and it is
        # recorded in the results rather than left implicit.
        pairing_mode = "label_matched_resample"
        n = min(len(texts), len(images))
        by_class_imgs: Dict[int, List[int]] = {c: [] for c in range(cfg.num_labels)}
        for idx, lab in enumerate(img_labels):
            by_class_imgs[lab].append(idx)

        rng = np.random.default_rng(cfg.seed)
        paired_texts, paired_images, paired_labels, paired_groups = [], [], [], []
        for t_idx in range(n):
            lab = txt_labels[t_idx]
            pool = by_class_imgs.get(lab) or []
            if not pool:
                continue
            i_idx = int(rng.choice(pool))
            paired_texts.append(texts[t_idx])
            paired_images.append(images[i_idx])
            paired_labels.append(lab)
            # group on the image source: that is the leakage risk that matters,
            # since one photograph can produce many crops.
            paired_groups.append(img_groups[i_idx])

        print(f"  [data] paired samples: {len(paired_labels)} (mode={pairing_mode})")

    # Tokenizer is shared by both pairing paths, so it is built after the
    # branch rather than inside the label-matched fallback.
    tokenizer = mf.SimpleTokenizer() if hasattr(mf, "SimpleTokenizer") else None
    if getattr(cfg, "use_pretrained_backbones", False):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_text_tokenizer)
        except Exception as exc:
            print(f"  [data] falling back to SimpleTokenizer ({exc})")

    # Two views over the identical arrays. The evaluation view is fully
    # deterministic; the training view applies a fresh random augmentation on
    # every access, so a replayed minority image is never the same tensor twice.
    dataset = mf.MultiModalDataset(
        paired_texts, paired_labels, paired_images,
        tokenizer=tokenizer, max_length=cfg.max_seq_length,
        augment_images=False,
    )
    train_dataset = mf.MultiModalDataset(
        paired_texts, paired_labels, paired_images,
        tokenizer=tokenizer, max_length=cfg.max_seq_length,
        augment_images=True,
    )

    # -- leakage-safe split -------------------------------------------------
    try:
        train_idx, val_idx, test_idx = mf.grouped_multimodal_split(
            paired_labels, paired_groups, train_ratio=0.7, val_ratio=0.15, seed=cfg.seed
        )
    except Exception as exc:
        print(f"  [data] grouped split unavailable ({exc}); using stratified split")
        train_idx, val_idx, test_idx = mf.stratified_split(
            list(range(len(paired_labels))), paired_labels,
            train_ratio=0.7, val_ratio=0.15, seed=cfg.seed
        )

    train_idx, val_idx = list(train_idx), list(val_idx)
    print(f"  [data] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    return dict(
        dataset=dataset,
        train_dataset=train_dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=list(test_idx),
        labels=paired_labels,
        groups=paired_groups,
        tokenizer=tokenizer,
        pairing_mode=pairing_mode,
        # Every image is now a real photograph or an augmented view of one.
        # Independent evidence is the number of distinct source photographs,
        # not the number of tensors.
        num_real_images=int(n_real_images),
        num_image_rows=int(len(img_groups)),
        num_independent_source_photos=int(len(set(img_groups))),
        synthetic_images_used=False,
    )


def make_loaders(mf, cfg, data: dict, batch_size: int):
    """Balanced train loader + plain val loader, matching repo conventions."""
    train_ds = Subset(data["train_dataset"], data["train_idx"])
    val_ds = Subset(data["dataset"], data["val_idx"])
    train_labels = [data["labels"][i] for i in data["train_idx"]]

    try:
        train_loader = mf.create_balanced_dataloader(
            train_ds, train_labels, batch_size=batch_size,
            num_classes=cfg.num_labels, shuffle=True,
        )
    except Exception as exc:
        print(f"  [loader] balanced sampler unavailable ({exc}); plain shuffle")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    plain_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, plain_loader, val_loader, train_ds, train_labels


# ---------------------------------------------------------------------------
# partition measurement -- the number that makes "non-IID" checkable
# ---------------------------------------------------------------------------
def label_skew_tv(client_indices: Sequence[Sequence[int]],
                  labels: Sequence[int], num_classes: int) -> Dict[str, float]:
    """Realised label skew of a partition.

    Reports the mean total-variation distance between each client's class
    distribution and the global one. TV ~ 0 means the clients are IID in labels
    no matter what alpha was requested; that is exactly the failure the legacy
    splitter produces. Also reports the coefficient of variation of shard sizes,
    which is what the legacy splitter actually varies.
    """
    labels = np.asarray(labels)
    global_hist = np.bincount(labels, minlength=num_classes).astype(float)
    global_hist /= max(global_hist.sum(), 1.0)

    tvs, sizes = [], []
    for indices in client_indices:
        indices = list(indices)
        sizes.append(len(indices))
        if not indices:
            continue
        hist = np.bincount(labels[indices], minlength=num_classes).astype(float)
        hist /= max(hist.sum(), 1.0)
        tvs.append(0.5 * float(np.abs(hist - global_hist).sum()))

    sizes_arr = np.asarray(sizes, dtype=float)
    return {
        "label_tv_mean": float(np.mean(tvs)) if tvs else 0.0,
        "label_tv_max": float(np.max(tvs)) if tvs else 0.0,
        "size_cv": float(sizes_arr.std() / sizes_arr.mean()) if sizes_arr.mean() > 0 else 0.0,
        "client_sizes": [int(s) for s in sizes],
    }


def partition(mf, data: dict, num_clients: int, alpha: float, seed: int,
              corrected: bool = True, use_groups: bool = True):
    """Partition the TRAIN split across clients.

    corrected=True  -> class-wise Dirichlet (pass labels): a real label skew.
    corrected=False -> legacy size-only split (omit labels), kept as the control
                       for E1 so the difference is measured rather than asserted.
    """
    train_ds = Subset(data["train_dataset"], data["train_idx"])
    train_labels = [data["labels"][i] for i in data["train_idx"]]
    train_groups = [data["groups"][i] for i in data["train_idx"]]

    if corrected:
        return mf.split_data_non_iid(
            train_ds, num_clients, alpha,
            labels=train_labels,
            seed=seed,
            groups=train_groups if use_groups else None,
        ), train_labels
    return mf.split_data_non_iid(train_ds, num_clients, alpha), train_labels


# ---------------------------------------------------------------------------
# cost accounting
# ---------------------------------------------------------------------------
def trainable_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    """Bytes a client uploads per round under FedAvg (float params only)."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad and p.is_floating_point()) * dtype_bytes


def param_count(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


def _norm_key(key: str) -> bool:
    """True for normalization-layer parameters (the ones FedBN keeps local)."""
    kl = key.lower()
    return any(tok in kl for tok in
               ("layernorm", "layer_norm", "batchnorm", ".bn", "norm.weight",
                "norm.bias", "running_mean", "running_var"))


# ---------------------------------------------------------------------------
# algorithm-aware federated training (E8)
# ---------------------------------------------------------------------------
def _forward(model, batch, device, model_type="multimodal"):
    """Mirror of the repo's train_epoch dispatch, so losses stay comparable."""
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    if model_type == "text":
        return model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"])
    if model_type == "vision":
        return model(pixel_values=batch["pixel_values"], labels=batch["labels"])
    return model(input_ids=batch["input_ids"],
                 attention_mask=batch["attention_mask"],
                 pixel_values=batch["pixel_values"],
                 labels=batch["labels"])


def federated_train_algo(mf, model_class, model_kwargs, data, client_indices,
                         val_loader, cfg, device, algorithm="fedavg",
                         model_type="multimodal", rounds=None, local_epochs=None,
                         lr=None, batch_size=None, mu=0.01, seed=0,
                         initial_state=None, balanced_sampler=False,
                         diversity_weight=0.0):
    """FedAvg / FedProx / SCAFFOLD / FedBN under identical everything else.

    The point of this function is that the aggregation rule is the ONLY thing
    that changes between arms: same backbone, same partition, same local budget,
    same optimizer, same seed. Table IV-style comparisons against numbers other
    authors measured on other corpora cannot support that claim; this can.

    FedProx  -- adds (mu/2)||w - w_global||^2 to the local objective.
    SCAFFOLD -- server/client control variates; local gradient is corrected by
                (c - c_i), and c_i is refreshed from the local drift.
    FedBN    -- normalization parameters are never aggregated.

    ``balanced_sampler`` and ``diversity_weight`` are orthogonal training
    controls used by E3. Keeping them here (rather than calling the centralized
    ``train_model`` helper) ensures that the alpha-specific client partition is
    actually exercised by every anti-collapse ablation arm.
    """
    set_seed(seed)
    rounds = int(rounds or cfg.fed_rounds)
    local_epochs = int(local_epochs or cfg.local_epochs)
    batch_size = int(batch_size or cfg.batch_size)
    lr = float(lr if lr is not None else cfg.learning_rate * 2)

    train_ds = Subset(data["train_dataset"], data["train_idx"])

    global_model = model_class(**model_kwargs).to(device)
    if initial_state is not None:
        global_model.load_state_dict(initial_state, strict=True)
    global_state = {k: v.detach().cpu().clone()
                    for k, v in global_model.state_dict().items()}

    upload_bytes = trainable_bytes(global_model)
    n_params = param_count(global_model)

    # SCAFFOLD control variates, one server-side c and one per client.
    server_c: Dict[str, torch.Tensor] = {}
    client_c: List[Dict[str, torch.Tensor]] = []
    if algorithm == "scaffold":
        server_c = {k: torch.zeros_like(v) for k, v in global_state.items()
                    if v.is_floating_point()}
        client_c = [{k: torch.zeros_like(v) for k, v in server_c.items()}
                    for _ in client_indices]

    # FedBN keeps normalization parameters on the client across rounds.
    local_norm_state: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(client_indices)

    history = {"round_f1_macro": [], "round_f1_micro": [],
               "round_seconds": [], "round_peak_mib": [],
               "balanced_fallback_clients": []}
    best_f1, best_metrics, best_state = -1.0, None, None

    for rnd in range(rounds):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        client_states, client_sizes = [], []
        new_client_c: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(client_indices)

        for cid, indices in enumerate(client_indices):
            indices = list(indices)
            if not indices:
                continue

            local_model = model_class(**model_kwargs).to(device)
            local_model.load_state_dict(global_state, strict=True)
            if algorithm == "fedbn" and local_norm_state[cid] is not None:
                # restore this client's own normalization statistics
                merged = local_model.state_dict()
                merged.update(local_norm_state[cid])
                local_model.load_state_dict(merged, strict=True)

            local_ds = Subset(train_ds, indices)
            fallback_balancing = False
            if balanced_sampler:
                # Client indices address ``train_ds``; map them back to the
                # full paired dataset before looking up labels.
                local_labels = [
                    _as_int_label(mf, data["labels"][data["train_idx"][i]])
                    for i in indices
                ]
                present = set(local_labels)
                if len(present) == int(cfg.num_labels):
                    loader = mf.create_balanced_dataloader(
                        local_ds, local_labels, batch_size=batch_size,
                        num_classes=cfg.num_labels, shuffle=True,
                    )
                else:
                    # A severely non-IID client cannot draw classes it does not
                    # possess. Balance the classes it *does* possess instead of
                    # failing or silently reverting to ordinary shuffling.
                    counts = {label: local_labels.count(label) for label in present}
                    sample_weights = torch.as_tensor(
                        [1.0 / counts[label] for label in local_labels],
                        dtype=torch.double,
                    )
                    generator = torch.Generator()
                    generator.manual_seed(seed * 100000 + rnd * 1000 + cid)
                    sampler = WeightedRandomSampler(
                        sample_weights, num_samples=len(local_labels),
                        replacement=True, generator=generator,
                    )
                    loader = DataLoader(
                        local_ds, batch_size=batch_size, sampler=sampler,
                    )
                    fallback_balancing = True
            else:
                loader = DataLoader(local_ds, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr)
            diversity_loss_fn = None
            if float(diversity_weight) > 0:
                diversity_loss_fn = mf.DiversityLoss(
                    num_classes=cfg.num_labels,
                    diversity_weight=float(diversity_weight),
                ).to(device)

            global_ref = None
            if algorithm == "fedprox":
                global_ref = {k: v.to(device) for k, v in global_state.items()
                              if v.is_floating_point()}

            steps = 0
            local_model.train()
            for _ in range(local_epochs):
                for batch in loader:
                    optimizer.zero_grad()
                    out = _forward(local_model, batch, device, model_type)
                    loss = out["loss"]
                    if diversity_loss_fn is not None:
                        loss = loss + diversity_loss_fn(out["logits"])

                    if algorithm == "fedprox":
                        prox = torch.zeros((), device=device)
                        for name, param in local_model.named_parameters():
                            if name in global_ref:
                                prox = prox + ((param - global_ref[name]) ** 2).sum()
                        loss = loss + (mu / 2.0) * prox

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)

                    if algorithm == "scaffold":
                        # g <- g - c_i + c, applied before the optimizer step
                        for name, param in local_model.named_parameters():
                            if param.grad is None or name not in server_c:
                                continue
                            param.grad.add_(
                                (server_c[name] - client_c[cid][name]).to(device)
                            )

                    optimizer.step()
                    steps += 1

            local_state = {k: v.detach().cpu().clone()
                           for k, v in local_model.state_dict().items()}

            if algorithm == "scaffold" and steps > 0:
                # c_i^+ = c_i - c + (x - y_i) / (K * eta)
                updated = {}
                for name in server_c:
                    drift = (global_state[name].float() - local_state[name].float())
                    updated[name] = (client_c[cid][name] - server_c[name]
                                     + drift / (steps * lr))
                new_client_c[cid] = updated

            if algorithm == "fedbn":
                local_norm_state[cid] = {k: v.clone()
                                         for k, v in local_state.items() if _norm_key(k)}

            client_states.append(local_state)
            client_sizes.append(len(indices))
            if fallback_balancing:
                history["balanced_fallback_clients"].append(
                    {"round": rnd + 1, "client": cid,
                     "present_classes": sorted(present)}
                )

            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not client_states:
            raise RuntimeError("no client had data; check the partition")

        # ---- aggregation -------------------------------------------------
        total = float(sum(client_sizes))
        new_global = {}
        for key in global_state:
            if algorithm == "fedbn" and _norm_key(key):
                new_global[key] = global_state[key]      # never aggregated
                continue
            ref = global_state[key]
            if not ref.is_floating_point():
                new_global[key] = client_states[0][key]  # counters etc.
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for state, size in zip(client_states, client_sizes):
                acc += state[key].float() * (size / total)
            new_global[key] = acc.to(ref.dtype)
        global_state = new_global

        if algorithm == "scaffold":
            # c <- c + (1/N) sum_i (c_i^+ - c_i)
            n_clients = max(1, len(client_indices))
            for cid, updated in enumerate(new_client_c):
                if updated is None:
                    continue
                for name in server_c:
                    server_c[name] += (updated[name] - client_c[cid][name]) / n_clients
                client_c[cid] = updated

        # ---- evaluate ----------------------------------------------------
        global_model.load_state_dict(global_state, strict=True)
        metrics = mf.evaluate(global_model, val_loader, device, model_type)
        f1_macro = float(metrics["f1_macro"])
        history["round_f1_macro"].append(f1_macro)
        history["round_f1_micro"].append(float(metrics["f1_micro"]))
        history["round_seconds"].append(time.time() - t0)
        history["round_peak_mib"].append(
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        )
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_metrics = {k: v for k, v in metrics.items()
                            if isinstance(v, (int, float, list))}
            best_state = {k: v.clone() for k, v in global_state.items()}
        print(f"    [{algorithm} r{rnd+1}/{rounds}] macro-F1 {f1_macro:.4f}")

    # Report prediction diversity from the same validation-selected checkpoint
    # as Macro F1, not merely from the last communication round.
    selected_predictions = []
    if best_state is not None:
        global_model.load_state_dict(best_state, strict=True)
        selected_metrics = mf.evaluate(global_model, val_loader, device, model_type)
        raw_predictions = selected_metrics.get("predictions")
        if raw_predictions is not None:
            selected_predictions = list(np.asarray(raw_predictions).ravel())
    unique_predictions = len({int(p) for p in selected_predictions})

    return dict(
        algorithm=algorithm,
        f1_macro=best_f1,
        f1_micro=float(best_metrics.get("f1_micro", 0.0)) if best_metrics else 0.0,
        metrics=best_metrics,
        history=history,
        client_sizes=[len(list(c)) for c in client_indices],
        num_params=n_params,
        upload_bytes_per_client_per_round=upload_bytes,
        total_comm_bytes=2 * len(client_indices) * rounds * upload_bytes,
        wall_seconds=float(sum(history["round_seconds"])),
        peak_mib=float(max(history["round_peak_mib"])) if history["round_peak_mib"] else 0.0,
        balanced_sampler=bool(balanced_sampler),
        diversity_weight=float(diversity_weight),
        unique_predicted_classes=(unique_predictions if selected_predictions else None),
        diversity_ratio=(unique_predictions / int(cfg.num_labels)
                         if selected_predictions else None),
    ), best_state


def local_only_baseline(mf, model_class, model_kwargs, data, client_indices,
                        val_loader, cfg, device, model_type="multimodal",
                        local_epochs=None, batch_size=None, lr=None, seed=0):
    """Each client trains alone on its own shard -- no federation at all.

    Without this row, "federated matches centralized" is only half an argument:
    it shows federation costs little, not that it buys anything. The gap between
    local-only and any federated arm is what federating actually earns.
    """
    set_seed(seed)
    batch_size = int(batch_size or cfg.batch_size)
    lr = float(lr if lr is not None else cfg.learning_rate * 2)
    # Match the federated arms' total local compute: rounds x local_epochs.
    epochs = int(local_epochs or (cfg.fed_rounds * cfg.local_epochs))

    train_ds = Subset(data["train_dataset"], data["train_idx"])
    per_client = []

    for cid, indices in enumerate(client_indices):
        indices = list(indices)
        if not indices:
            continue
        model = model_class(**model_kwargs).to(device)
        loader = DataLoader(Subset(train_ds, indices), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                loss = _forward(model, batch, device, model_type)["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        metrics = mf.evaluate(model, val_loader, device, model_type)
        per_client.append(float(metrics["f1_macro"]))
        print(f"    [local-only client {cid}] macro-F1 {per_client[-1]:.4f} "
              f"(n={len(indices)})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arr = np.asarray(per_client, dtype=float)
    return dict(
        per_client_f1=per_client,
        mean_f1=float(arr.mean()) if arr.size else 0.0,
        std_f1=float(arr.std()) if arr.size else 0.0,
        best_f1=float(arr.max()) if arr.size else 0.0,
        worst_f1=float(arr.min()) if arr.size else 0.0,
        num_clients=len(per_client),
        local_epochs=epochs,
    )


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------
def run_all(mf, store: Store, tier: dict, cfg, data, device, only=None):
    want = (lambda tag: True) if not only else (lambda tag: tag in set(only))

    num_classes = cfg.num_labels
    batch_size = int(tier["batch_size"])
    train_loader, plain_loader, val_loader, train_ds, train_labels = make_loaders(
        mf, cfg, data, batch_size)

    mm_kwargs = dict(num_labels=num_classes, fusion_type="concat")
    model_class = mf.MultiModalClassifier

    # Architectures compared under federation. Each one trains on the SAME
    # client partition inside a given (alpha, K, seed) cell, so a difference
    # between them is the architecture and not a different split.
    fed_archs = build_archs(mf, num_classes, tier.get("fed_archs"))
    print("federated architectures: " + ", ".join(fed_archs) if fed_archs
          else "federated architectures: none resolved")

    store.put("_meta", "data", {
        "pairing_mode": data["pairing_mode"],
        "num_paired": len(data["labels"]),
        "num_real_images": data["num_real_images"],
        "train": len(data["train_idx"]),
        "val": len(data["val_idx"]),
        "class_counts": np.bincount(data["labels"], minlength=num_classes).tolist(),
        # The caveat must describe the pairing that was actually used, not a
        # fixed string. A genuine box-linked run and a label-matched fallback
        # support very different fusion claims.
        "note": (
            "image-text pairs are genuine box-linked observations: each note "
            "was written about the exact crop it is paired with; no synthetic "
            "images are used"
            if data["pairing_mode"] == "genuine_box_linked_observation" else
            "image-text pairs are label-matched resamples, not genuine "
            "co-observations; fusion results must be read with that caveat"
        ),
        "synthetic_images_used": bool(data.get("synthetic_images_used", False)),
        "num_independent_source_groups": data.get("num_independent_source_photos"),
    })

    # ---- E1: alpha sweep, corrected vs legacy splitter ---------------------
    if not want("E1"):
        print("skipping E1")
    else:
        print("\n=== E1  Dirichlet alpha sweep (corrected vs legacy splitter) ===")
        for alpha in tier["alphas"]:
            for corrected in (True, False):
                tag = "corrected" if corrected else "legacy"
                for seed in tier["seeds"]:
                    cell = f"alpha={alpha}|split={tag}|seed={seed}"
                    pending = [a for a in fed_archs
                               if not store.has("E1_alpha_sweep", f"{cell}|arch={a}")]
                    if not pending:
                        print(f"  skip {cell} (all archs done)")
                        continue
                    # One partition per cell, reused by every architecture.
                    idx, tl = partition(mf, data, cfg.num_clients, alpha, seed,
                                        corrected=corrected)
                    skew = label_skew_tv(idx, tl, num_classes)
                    print(f"  {cell}  realised label TV={skew['label_tv_mean']:.3f} "
                          f"size CV={skew['size_cv']:.3f}")
                    for arch in pending:
                        acls, akw, atype = fed_archs[arch]
                        print(f"    arch={arch} ({atype})")
                        res, _ = federated_train_algo(
                            mf, acls, akw, data, idx, val_loader, cfg,
                            device, algorithm="fedavg", model_type=atype,
                            rounds=tier["fed_rounds"],
                            local_epochs=tier["local_epochs"],
                            batch_size=batch_size, seed=seed)
                        res.update(partition=skew, alpha=alpha, splitter=tag,
                                   arch=arch, model_type=atype,
                                   shared_partition=True)
                        store.put("E1_alpha_sweep", f"{cell}|arch={arch}", res)

    # ---- E2: client-count sweep -------------------------------------------
    if not want("E2"):
        print("skipping E2")
    else:
        print("\n=== E2  client-count sweep ===")
        for k_clients in tier["client_counts"]:
            for seed in tier["seeds"][:1]:
                cell = f"K={k_clients}|seed={seed}"
                pending = [a for a in fed_archs
                           if not store.has("E2_client_sweep", f"{cell}|arch={a}")]
                if not pending:
                    print(f"  skip {cell} (all archs done)")
                    continue
                idx, tl = partition(mf, data, k_clients, 1.0, seed, corrected=True)
                skew = label_skew_tv(idx, tl, num_classes)
                print(f"  {cell}")
                for arch in pending:
                    acls, akw, atype = fed_archs[arch]
                    print(f"    arch={arch} ({atype})")
                    res, _ = federated_train_algo(
                        mf, acls, akw, data, idx, val_loader, cfg, device,
                        algorithm="fedavg", model_type=atype,
                        rounds=tier["fed_rounds"],
                        local_epochs=tier["local_epochs"],
                        batch_size=batch_size, seed=seed)
                    res.update(partition=skew, num_clients=k_clients,
                               arch=arch, model_type=atype, shared_partition=True)
                    store.put("E2_client_sweep", f"{cell}|arch={arch}", res)

    # ---- E3: anti-collapse ablation, federated + multimodal ----------------
    # The repo's run_ablation_study asks this centrally, text-only, at one
    # partition. Collapse is a federated, imbalanced-data failure, so it is
    # tested here under federation at both a mild and a severe partition.
    if not want("E3"):
        print("skipping E3")
    else:
        print("\n=== E3  anti-collapse components (federated, multimodal) ===")
        arms = [("both", True, 1.0), ("no_diversity", True, 0.0),
                ("no_balanced", False, 1.0), ("neither", False, 0.0)]
        for alpha in [a for a in (1.0, 0.1) if a in tier["alphas"]] or [1.0]:
            for name, balanced, div_w in arms:
                key = f"{name}|alpha={alpha}"
                if store.has("E3_anticollapse", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}  (balanced={balanced}, diversity={div_w})")
                seed = tier["seeds"][0]
                idx, tl = partition(mf, data, cfg.num_clients, alpha, seed, corrected=True)

                result, _ = federated_train_algo(
                    mf, model_class, mm_kwargs, data, idx, val_loader, cfg,
                    device, algorithm="fedavg", rounds=tier["fed_rounds"],
                    local_epochs=tier["local_epochs"], batch_size=batch_size,
                    seed=seed, balanced_sampler=balanced,
                    diversity_weight=div_w,
                )
                result.update({
                    "arm": name, "alpha": alpha,
                    "balanced_sampler": balanced, "diversity_weight": div_w,
                    "partition": label_skew_tv(idx, tl, num_classes),
                })
                store.put("E3_anticollapse", key, result)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- E4: warm start vs cold start --------------------------------------
    if not want("E4"):
        print("skipping E4")
    else:
        print("\n=== E4  warm start vs cold start ===")
        warm_path = Path(store.path).parent / "farm_warmstart_concat.pt"
        # Gate on the weights file, not a store record: across a chunked run the
        # record travels in the JSON while the .pt may not have been re-uploaded.
        # Trusting the record alone would leave the "warm" arm silently cold.
        if not warm_path.exists():
            print("  [warm] training the shared centralized initialization")
            warm_cfg = copy.copy(cfg)
            warm_cfg.epochs = int(tier["central_epochs"])
            warm_cfg.batch_size = batch_size
            set_seed(0)
            wmodel = model_class(**mm_kwargs).to(device)
            mf.train_model(wmodel, train_loader, val_loader, warm_cfg, device,
                           "multimodal", diversity_weight=1.0)
            torch.save({k: v.detach().cpu() for k, v in wmodel.state_dict().items()},
                       warm_path)
            del wmodel
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        warm_state = torch.load(warm_path, map_location="cpu")

        for seed in tier["seeds"]:
            for mode in ("warm", "cold"):
                key = f"{mode}|seed={seed}"
                if store.has("E4_warmstart", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                idx, _ = partition(mf, data, cfg.num_clients, 1.0, seed, corrected=True)
                res, _ = federated_train_algo(
                    mf, model_class, mm_kwargs, data, idx, val_loader, cfg, device,
                    algorithm="fedavg", rounds=tier["fed_rounds"],
                    local_epochs=tier["local_epochs"], batch_size=batch_size,
                    seed=seed,
                    initial_state=warm_state if mode == "warm" else None)
                res["mode"] = mode
                store.put("E4_warmstart", key, res)

    # ---- E5: fusion strategy x seeds (variance) ----------------------------
    if not want("E5"):
        print("skipping E5")
    else:
        print("\n=== E5  fusion strategies x seeds ===")
        for fusion in FUSION_TYPES:
            for seed in tier["fusion_seeds"]:
                key = f"{fusion}|seed={seed}"
                if store.has("E5_fusion_seeds", key):
                    print(f"  skip {key}")
                    continue
                print(f"  {key}")
                kwargs = dict(num_labels=num_classes, fusion_type=fusion)
                try:
                    idx, _ = partition(mf, data, cfg.num_clients, 1.0, seed, corrected=True)
                    res, _ = federated_train_algo(
                        mf, model_class, kwargs, data, idx, val_loader, cfg, device,
                        algorithm="fedavg", rounds=tier["fed_rounds"],
                        local_epochs=tier["local_epochs"], batch_size=batch_size,
                        seed=seed)
                    res["fusion_type"] = fusion
                    store.put("E5_fusion_seeds", key, res)
                except Exception as exc:
                    print(f"    fusion '{fusion}' failed: {exc}")
                    store.put("E5_fusion_seeds", key, {"error": str(exc),
                                                       "fusion_type": fusion})

    # ---- E6: measured cost -------------------------------------------------
    if not want("E6"):
        print("skipping E6")
    else:
        print("\n=== E6  measured cost per branch ===")
        for fusion in FUSION_TYPES:
            key = f"concat_cost|{fusion}"
            if store.has("E6_cost", key):
                continue
            try:
                model = model_class(num_labels=num_classes, fusion_type=fusion)
            except Exception as exc:
                store.put("E6_cost", key, {"error": str(exc)})
                continue
            up = trainable_bytes(model)
            store.put("E6_cost", key, {
                "fusion_type": fusion,
                "num_params": param_count(model),
                "trainable_params": param_count(model, trainable_only=True),
                "upload_bytes_per_client_per_round": up,
                "upload_mib_per_client_per_round": up / 2**20,
                "round_trip_bytes_per_client": 2 * up,
                "note": "analytic size; wall-clock and peak memory come from "
                        "the per-round history recorded in E1/E2/E8",
            })
            del model

        # Same cost question for every architecture compared under federation.
        # Communication is what federation actually charges the estate, and it
        # scales with parameter count, so it differs across these by ~3.5x.
        for arch, (acls, akw, atype) in fed_archs.items():
            key = f"arch_cost|{arch}"
            if store.has("E6_cost", key):
                continue
            try:
                model = acls(**akw)
            except Exception as exc:
                store.put("E6_cost", key, {"arch": arch, "error": str(exc)})
                continue
            up = trainable_bytes(model)
            store.put("E6_cost", key, {
                "arch": arch,
                "model_class": acls.__name__,
                "model_type": atype,
                "num_params": param_count(model),
                "trainable_params": param_count(model, trainable_only=True),
                "upload_bytes_per_client_per_round": up,
                "upload_mib_per_client_per_round": up / 2**20,
                "round_trip_bytes_per_client": 2 * up,
                "note": "analytic float32 size; accuracy for this same arch is "
                        "in E1/E2/E8 under matching arch= keys",
            })
            del model

    # ---- E7: retrieval -- already covered elsewhere -------------------------
    if want("E7"):
        print("\n=== E7  retrieval ===")
        print("  Not duplicated here: experiments/advisory_retrieval_eval.py "
              "already evaluates advisory retrieval. Run that script instead.")

    # ---- E8: matched-setting federated baselines ---------------------------
    if not want("E8"):
        print("skipping E8")
    else:
        print("\n=== E8  matched-setting aggregation baselines ===")
        algos = ["fedavg", "fedprox", "scaffold", "fedbn"]
        for alpha in [a for a in (1.0, 0.1) if a in tier["alphas"]] or [1.0]:
            for algo in algos:
                for seed in tier["seeds"]:
                    cell = f"{algo}|alpha={alpha}|seed={seed}"
                    pending = [a for a in fed_archs
                               if not store.has("E8_baselines", f"{cell}|arch={a}")]
                    if not pending:
                        print(f"  skip {cell} (all archs done)")
                        continue
                    idx, tl = partition(mf, data, cfg.num_clients, alpha, seed,
                                        corrected=True)
                    skew = label_skew_tv(idx, tl, num_classes)
                    print(f"  {cell}")
                    for arch in pending:
                        acls, akw, atype = fed_archs[arch]
                        print(f"    arch={arch} ({atype})")
                        res, _ = federated_train_algo(
                            mf, acls, akw, data, idx, val_loader, cfg,
                            device, algorithm=algo, model_type=atype,
                            rounds=tier["fed_rounds"],
                            local_epochs=tier["local_epochs"],
                            batch_size=batch_size, seed=seed)
                        res.update(alpha=alpha, partition=skew, arch=arch,
                                   model_type=atype, shared_partition=True)
                        store.put("E8_baselines", f"{cell}|arch={arch}", res)

            # local-only control, also per architecture
            seed = tier["seeds"][0]
            lo_cell = f"local_only|alpha={alpha}|seed={seed}"
            lo_pending = [a for a in fed_archs
                          if not store.has("E8_baselines", f"{lo_cell}|arch={a}")]
            if lo_pending:
                idx, _ = partition(mf, data, cfg.num_clients, alpha, seed, corrected=True)
                print(f"  {lo_cell}")
                for arch in lo_pending:
                    acls, akw, atype = fed_archs[arch]
                    print(f"    arch={arch} ({atype})")
                    res = local_only_baseline(
                        mf, acls, akw, data, idx, val_loader, cfg, device,
                        model_type=atype,
                        local_epochs=tier["fed_rounds"] * tier["local_epochs"],
                        batch_size=batch_size, seed=seed)
                    res.update(alpha=alpha, arch=arch, model_type=atype,
                               shared_partition=True)
                    store.put("E8_baselines", f"{lo_cell}|arch={arch}", res)

    print("\nDone.")


# ---------------------------------------------------------------------------
def main(base_py: str, tier: str = "standard", out: str = "farm_results.json",
         data_root: Optional[str] = None, only=None, seeds=None, alphas=None):
    t = dict(TIERS[tier])
    if seeds:
        t["seeds"] = list(seeds)
        t["fusion_seeds"] = list(seeds)
    if alphas:
        t["alphas"] = list(alphas)

    print("=" * 68)
    print(f"FarmFederate ablation suite | tier={tier}")
    print("=" * 68)

    mf = load_base(base_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("\n!! No GPU detected. Runtime -> Change runtime type -> GPU.")
        print("   Continuing on CPU; expect this to be very slow.\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = mf.Config(
        batch_size=t["batch_size"],
        epochs=t["central_epochs"],
        fed_rounds=t["fed_rounds"],
        local_epochs=t["local_epochs"],
        max_samples_per_class=t["text_per_class"],
        image_fill_target=t["image_fill_target"],
    )

    root = resolve_data_root(data_root)
    print(f"data root: {root}")
    data = build_data(mf, cfg, t, root)

    store = Store(Path(out))
    run_all(mf, store, t, cfg, data, device, only=only)
    print(f"\nresults -> {store.path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FarmFederate reviewer ablation suite")
    ap.add_argument("--base", default="backend/FarmFederate_Colab_Complete.py",
                    help="path to FarmFederate_Colab_Complete.py")
    ap.add_argument("--tier", default="standard", choices=list(TIERS))
    ap.add_argument("--out", default="farm_results.json")
    ap.add_argument("--data-root", default=None,
                    help="folder holding real_dataset_sorted/ and text_data/")
    ap.add_argument("--only", default=None,
                    help="comma-separated experiment tags, e.g. E1,E8")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds override")
    ap.add_argument("--alphas", default=None, help="comma-separated alphas override")
    a = ap.parse_args()
    main(base_py=a.base, tier=a.tier, out=a.out, data_root=a.data_root,
         only=[x.strip() for x in a.only.split(",")] if a.only else None,
         seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else None,
         alphas=[float(x) for x in a.alphas.split(",")] if a.alphas else None)
