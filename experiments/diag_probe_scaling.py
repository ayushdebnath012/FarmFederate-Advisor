#!/usr/bin/env python3
"""Is the federated text probe failing because of skew, or because of scaling?

The robustness sweep gives the text branch ~0.18 macro-F1 at every Dirichlet
alpha, including alpha=10 where the partition is nearly IID. A failure that does
not respond to the variable being swept is not a property of that variable. The
suspect is feature scale: DistilBERT mean-pooled features are 768-d and
unnormalized, DeiT-tiny CLS features are 192-d, and both are fed to the same
SGD head at the same learning rate.

This fits centralized single-client probes on the identical frozen features with
and without train-fitted standardization. If standardization moves text and
leaves image roughly alone, the sweep is measuring the optimizer and has to be
rerun with the scaling fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import tea_train as tea  # noqa: E402
from tea_federated_all_systems import (  # noqa: E402
    build_split, frozen_image_features, frozen_text_features, macro_f1,
)


def probe(Xtr, ytr, Xva, yva, device, lr, steps=150, seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], tea.NUM_CLASSES).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    lf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device, dtype=torch.long)
    Xv = torch.tensor(Xva, device=device)
    for _ in range(steps):
        opt.zero_grad()
        lf(head(Xt), yt).backward()
        opt.step()
    with torch.no_grad():
        return macro_f1(yva, head(Xv).argmax(1).cpu().numpy())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, obb, audit = build_split()
    idx = lambda v: int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    ytr = np.array([idx(l) for l in parts["train"]["labels"]])
    yva = np.array([idx(l) for l in parts["validation"]["labels"]])

    F = {}
    for n in ("train", "validation"):
        F[f"text_{n}"] = frozen_text_features(parts[n]["text"].tolist(), device)
        F[f"image_{n}"] = frozen_image_features(obb[n], device)

    for name in ("text", "image"):
        A, B_ = F[f"{name}_train"], F[f"{name}_validation"]
        mu, sd = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        print(f"\n{name}: dim={A.shape[1]}  |x| mean={np.abs(A).mean():.3f}  "
              f"row-norm mean={np.linalg.norm(A, axis=1).mean():.2f}")
        for lr in (1e-3, 1e-2, 1e-1):
            raw = probe(A, ytr, B_, yva, device, lr)
            std = probe((A - mu) / sd, ytr, (B_ - mu) / sd, yva, device, lr)
            print(f"  lr={lr:<6} raw={raw:.4f}   standardized={std:.4f}")


if __name__ == "__main__":
    main()
