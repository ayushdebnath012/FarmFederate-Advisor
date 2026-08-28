#!/usr/bin/env python3
"""Federated robustness: severe skew, dropout, stragglers and poisoned updates.

The paper federates under mild and moderate label skew with every client
reporting a fresh, honest update in every round. That is the easy case, and the
Threats to Validity section admits it. This script measures the four failure
modes it does not test, on the same audited source-grouped split and the same
frozen encoders, so the robustness numbers sit on the identical support as the
accuracy numbers.

  heterogeneity  Dirichlet alpha down to 0.01, plus a pathological partition
                 that gives each client a single class. alpha=0.01 is not just
                 a smaller 0.1: most clients end up holding one class, which is
                 where the averaging assumption in FedAvg actually breaks.

  dropout        a fraction of clients fail to report in a round, resampled
                 independently each round. The server averages whoever arrived,
                 reweighted -- what FedAvg does in deployment.

  stragglers     a fraction of clients report an update computed from an
                 earlier global model. Distinct from dropout: the update
                 arrives, but it is stale, so it pulls the average backwards
                 instead of merely shrinking the cohort.

  poisoning      a fraction of clients are malicious:
                   label_flip  honest training on deranged labels
                   sign_flip   the update negated and scaled
                   gaussian    the update replaced by noise
                 Label flip is the subtle one -- a well-formed gradient step
                 toward the wrong target -- so it tests the aggregator rather
                 than an outlier filter. Each attack is run under plain FedAvg
                 mean and under coordinate-wise median, so the table reports
                 whether a robust aggregator recovers the loss.

Within a cell every arm shares the client partition with its own control at the
same seed, so a difference is the failure mode and not the split.

Features are standardized with train-fitted per-dimension mean and SD before the
head is trained. Without it the comparison is not between modalities at all:
DistilBERT mean-pooled features are 768-d with a mean absolute value of 0.20 and
DeiT-tiny CLS features are 192-d at 1.31, so at one shared learning rate the text
head barely moves (diag_probe_scaling.py: 0.352 centralized raw against 0.646
standardized, while image changes by 0.02). A simulation-only convenience is
noted rather than hidden: a real server would not hold global training
statistics, but the same transform is applied to every arm, so no condition is
advantaged relative to its control.
"""

from __future__ import annotations

import argparse
import json
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
    build_split, dirichlet_partition, frozen_image_features,
    frozen_text_features, macro_f1,
)

ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
DROPOUT = [0.0, 0.1, 0.3, 0.5, 0.7]
STRAGGLER = [0.0, 0.25, 0.5, 0.75]
ATTACKS = ["label_flip", "sign_flip", "gaussian"]
POISON_FRACS = [0.1, 0.2, 0.3]
BASE_ALPHA = 0.5


def pathological_partition(labels, num_clients, seed):
    """One class per client: the worst case for parameter averaging."""
    rng = np.random.RandomState(seed)
    lab = np.asarray(labels)
    classes = sorted({int(v) for v in lab.tolist()})
    clients = [[] for _ in range(num_clients)]
    for i, c in enumerate(classes):
        idx = np.where(lab == c)[0]
        rng.shuffle(idx)
        owners = [k for k in range(num_clients) if k % len(classes) == i]
        if not owners:
            owners = [i % num_clients]
        for j, part in enumerate(np.array_split(idx, len(owners))):
            clients[owners[j]].extend(part.tolist())
    return [sorted(c) for c in clients]


def derangement(n, rng):
    """A label permutation with no fixed point, so the flip always flips."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def aggregate(states, weights, how):
    if how == "mean":
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        return {k: sum(float(wi) * st[k] for wi, st in zip(w, states))
                for k in states[0]}
    if how == "median":                       # coordinate-wise, unweighted
        return {k: torch.stack([st[k] for st in states]).median(0).values
                for k in states[0]}
    raise ValueError(how)


def fedavg(Xtr, ytr, Xva, yva, client_idx, rounds, local_epochs, lr, device,
           seed, dropout=0.0, straggler=0.0, attack=None, attack_frac=0.0,
           agg="mean"):
    """FedAvg with optional dropout, staleness, malicious clients, robust agg."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 9973)
    d = Xtr.shape[1]
    g = nn.Linear(d, tea.NUM_CLASSES).to(device)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device, dtype=torch.long)
    Xv = torch.tensor(Xva, device=device)
    loss_fn = nn.CrossEntropyLoss()
    active = [i for i, c in enumerate(client_idx) if c]
    n_bad = int(round(attack_frac * len(active))) if attack else 0
    bad = set(rng.choice(active, size=n_bad, replace=False).tolist()) if n_bad else set()
    perm = torch.tensor(derangement(tea.NUM_CLASSES, rng), device=device)
    history, stale_cache, dropped = [], {}, 0

    for _ in range(rounds):
        states, weights = [], []
        for i in active:
            if dropout and rng.random() < dropout:
                dropped += 1
                continue                          # client never reports
            if straggler and rng.random() < straggler and i in stale_cache:
                states.append(stale_cache[i])     # stale update still counted
                weights.append(len(client_idx[i]))
                continue
            local = nn.Linear(d, tea.NUM_CLASSES).to(device)
            local.load_state_dict(g.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9)
            idx = torch.tensor(client_idx[i], dtype=torch.long, device=device)
            target = perm[yt[idx]] if (i in bad and attack == "label_flip") else yt[idx]
            for _ in range(local_epochs):
                opt.zero_grad()
                loss_fn(local(Xt[idx]), target).backward()
                opt.step()
            sd = {k: v.detach().clone() for k, v in local.state_dict().items()}
            if i in bad and attack == "sign_flip":
                gsd = g.state_dict()
                sd = {k: gsd[k] - 5.0 * (sd[k] - gsd[k]) for k in sd}
            elif i in bad and attack == "gaussian":
                sd = {k: torch.randn_like(v) * v.std().clamp_min(1e-3)
                      for k, v in sd.items()}
            states.append(sd)
            weights.append(len(client_idx[i]))
            stale_cache[i] = sd
        if not states:                            # every client dropped
            history.append(history[-1] if history else 0.0)
            continue
        g.load_state_dict(aggregate(states, weights, agg))
        with torch.no_grad():
            history.append(macro_f1(yva, g(Xv).argmax(1).cpu().numpy()))
    return {"final_macro_f1": history[-1] if history else 0.0,
            "best_macro_f1": max(history) if history else 0.0,
            "round_macro_f1": history,
            "num_malicious": len(bad), "active_clients": len(active),
            "client_rounds_dropped": dropped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tea_results/federated_robustness")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts, obb, audit = build_split()
    print(f"device: {device}")
    print(f"split: {audit}")

    def label_index(v):
        return int(v[0]) if isinstance(v, (list, tuple)) else int(v)

    ytr = np.array([label_index(l) for l in parts["train"]["labels"]])
    yva = np.array([label_index(l) for l in parts["validation"]["labels"]])

    F = {}
    for name in ("train", "validation"):
        F[f"text_{name}"] = frozen_text_features(parts[name]["text"].tolist(), device)
        F[f"image_{name}"] = frozen_image_features(obb[name], device)
        print(f"  {name}: text {F['text_' + name].shape}  "
              f"image {F['image_' + name].shape}")

    for mod in ("text", "image"):                  # train-fitted standardization
        tr = F[f"{mod}_train"]
        mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6
        for name in ("train", "validation"):
            F[f"{mod}_{name}"] = ((F[f"{mod}_{name}"] - mu) / sd).astype(np.float32)
        print(f"  standardized {mod}: train |x| mean "
              f"{np.abs(F[mod + '_train']).mean():.3f}")

    SYSTEMS = {
        "text": (F["text_train"], F["text_validation"]),
        "image": (F["image_train"], F["image_validation"]),
        "fusion": (np.hstack([F["image_train"], F["text_train"]]),
                   np.hstack([F["image_validation"], F["text_validation"]])),
    }
    res = {
        "_meta": {
            "protocol": "frozen encoders, SGD linear head, FedAvg; "
                        "arm and control share the partition at each seed",
            "split_audit": audit, "rounds": a.rounds, "clients": a.clients,
            "local_epochs": a.local_epochs, "lr": a.lr, "seeds": seeds,
            "base_alpha": BASE_ALPHA, "systems": list(SYSTEMS),
            "features": "frozen, standardized with train-fitted mean/SD",
        },
        "heterogeneity": {}, "dropout": {}, "stragglers": {}, "poisoning": {},
    }

    def run(store, key, part_fn, **kw):
        for sname, (Xtr, Xva) in SYSTEMS.items():
            vals, hist = [], []
            for sd in seeds:
                r = fedavg(Xtr, ytr, Xva, yva, part_fn(sd), a.rounds,
                           a.local_epochs, a.lr, device, sd, **kw)
                vals.append(r["final_macro_f1"])
                hist.append(r["round_macro_f1"])
            store[f"{key}|system={sname}"] = {
                "system": sname, "mean_macro_f1": float(np.mean(vals)),
                "sd": float(np.std(vals)), "per_seed": vals,
                "round_macro_f1_seed0": hist[0], **kw,
            }
            print(f"    {key:<26}{sname:<8}{np.mean(vals):.4f} "
                  f"+-{np.std(vals):.4f}", flush=True)

    def dirichlet(alpha):
        return lambda sd: dirichlet_partition(ytr, a.clients, alpha, sd)

    print("\n=== severe heterogeneity ===")
    for al in ALPHAS:
        run(res["heterogeneity"], f"alpha={al}", dirichlet(al))
    run(res["heterogeneity"], "alpha=pathological",
        lambda sd: pathological_partition(ytr, a.clients, sd))

    print(f"\n=== client dropout (alpha={BASE_ALPHA}) ===")
    for p in DROPOUT:
        run(res["dropout"], f"dropout={p}", dirichlet(BASE_ALPHA), dropout=p)

    print(f"\n=== stragglers (alpha={BASE_ALPHA}) ===")
    for p in STRAGGLER:
        run(res["stragglers"], f"straggler={p}", dirichlet(BASE_ALPHA),
            straggler=p)

    print(f"\n=== poisoned updates (alpha={BASE_ALPHA}) ===")
    for how in ("mean", "median"):
        print(f"  aggregator: {how}")
        run(res["poisoning"], f"clean|agg={how}", dirichlet(BASE_ALPHA), agg=how)
        for atk in ATTACKS:
            for frac in POISON_FRACS:
                run(res["poisoning"], f"{atk}@{frac}|agg={how}",
                    dirichlet(BASE_ALPHA), attack=atk, attack_frac=frac, agg=how)

    path = out_dir / "federated_robustness.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
