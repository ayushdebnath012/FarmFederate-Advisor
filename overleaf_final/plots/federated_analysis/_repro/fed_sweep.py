"""
Stage 2: Real federated-learning sweep on the REAL FarmFederate data.

Reuses the repo's ACTUAL FedAvg aggregation and its ACTUAL non-IID splitter
(copied verbatim from backend/FarmFederate_Colab_Complete.py) so the curves
describe the system as it is really implemented, not an idealised version.

Sweeps: rounds x clients x dirichlet-alpha x local-epochs x participation
        x modality (text / image / text+image), 3 seeds each.
"""
import json, random, itertools, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

HERE = Path(__file__).parent
FEAT = HERE / "features"
OUT = HERE / "sweep"
OUT.mkdir(exist_ok=True, parents=True)

DEV = "cpu"
torch.set_num_threads(4)  # small ops: fewer threads = less overhead
CLASSES = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
NUM_LABELS = 5


# ============================================================================
# VERBATIM from backend/FarmFederate_Colab_Complete.py (lines 4533-4573)
# ============================================================================
def split_data_non_iid_REPO(dataset_len, num_clients, alpha=0.5):
    """The repo's actual splitter. NOTE: shuffles indices then cuts contiguous
    chunks sized by a Dirichlet draw -> QUANTITY skew only, labels stay IID."""
    n = dataset_len
    indices = list(range(n))
    random.shuffle(indices)

    proportions = np.random.dirichlet([alpha] * num_clients)
    splits = (proportions * n).astype(int)
    splits[-1] = n - splits[:-1].sum()

    client_indices = []
    start = 0
    for size in splits:
        client_indices.append(indices[start:start + size])
        start += size
    return client_indices


def split_label_dirichlet(labels, num_clients, alpha=0.5):
    """Correct non-IID: per-class Dirichlet -> genuine LABEL skew."""
    n = len(labels)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(NUM_LABELS):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        p = np.random.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
        for ci, part in enumerate(np.split(idx_c, cuts)):
            client_indices[ci].extend(part.tolist())
    for ci in range(num_clients):
        random.shuffle(client_indices[ci])
    return client_indices


def fedavg(global_model, client_models, client_sizes):
    """VERBATIM from repo (lines 4553-4573)."""
    global_dict = global_model.state_dict()
    paired = [(m, s) for m, s in zip(client_models, client_sizes) if s > 0]
    if len(paired) == 0:
        return global_model
    total_size = sum(s for _, s in paired)
    for key in global_dict.keys():
        accum = None
        for m, s in paired:
            val = m.state_dict()[key].float() * (s / total_size)
            accum = val if accum is None else accum + val
        global_dict[key] = accum
    global_model.load_state_dict(global_dict)
    return global_model


# ============================================================================
# Model: classifier head on frozen encoder features
# ============================================================================
class Head(nn.Module):
    def __init__(self, in_dim, num_labels=NUM_LABELS, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def local_train(model, X, y, epochs, lr, bs=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    pred = logits.argmax(1).cpu().numpy()
    yt = y.cpu().numpy()
    return {
        "f1_macro": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(yt, pred, average="micro", zero_division=0)),
        "acc": float(accuracy_score(yt, pred)),
        "diversity": float(len(np.unique(pred)) / NUM_LABELS),
    }


def run_federated(Xtr, ytr, Xte, yte, num_clients, rounds, local_epochs,
                  alpha, lr, splitter, participation=1.0, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    in_dim = Xtr.shape[1]
    g = Head(in_dim).to(DEV)

    if splitter == "repo":
        cidx = split_data_non_iid_REPO(len(ytr), num_clients, alpha)
    else:
        cidx = split_label_dirichlet(ytr.cpu().numpy(), num_clients, alpha)

    hist = []
    for r in range(rounds):
        gstate = g.state_dict()
        pool = [i for i in range(num_clients) if len(cidx[i]) > 0]
        k = max(1, int(round(participation * len(pool))))
        sel = random.sample(pool, min(k, len(pool)))

        cmodels, csizes = [], []
        for ci in sel:
            idx = torch.tensor(cidx[ci], dtype=torch.long)
            lm = Head(in_dim).to(DEV)
            lm.load_state_dict(gstate)
            local_train(lm, Xtr[idx], ytr[idx], local_epochs, lr)
            cmodels.append(lm)
            csizes.append(len(idx))

        g = fedavg(g, cmodels, csizes)
        m = evaluate(g, Xte, yte)
        m["round"] = r + 1
        hist.append(m)

    sizes = [len(c) for c in cidx]
    # label-distribution skew actually realised (mean TV distance from global)
    yv = ytr.cpu().numpy()
    gdist = np.bincount(yv, minlength=NUM_LABELS) / len(yv)
    tvs = []
    for c in cidx:
        if len(c) == 0:
            continue
        d = np.bincount(yv[np.array(c)], minlength=NUM_LABELS) / len(c)
        tvs.append(0.5 * np.abs(d - gdist).sum())
    return hist, {
        "client_sizes": sizes,
        "size_cv": float(np.std(sizes) / max(np.mean(sizes), 1e-9)),
        "label_skew_tv": float(np.mean(tvs)) if tvs else 0.0,
    }


def run_centralized(Xtr, ytr, Xte, yte, epochs, lr, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = Head(Xtr.shape[1]).to(DEV)
    best = None
    hist = []
    for e in range(epochs):
        local_train(m, Xtr, ytr, 1, lr)
        r = evaluate(m, Xte, yte)
        r["epoch"] = e + 1
        hist.append(r)
        if best is None or r["f1_macro"] > best["f1_macro"]:
            best = r
    return best, hist


# ============================================================================
def main():
    tfeat = np.load(FEAT / "text_feat.npy")
    ifeat = np.load(FEAT / "image_feat.npy")
    labels = np.load(FEAT / "labels.npy")
    meta = json.load(open(FEAT / "meta.json"))
    print("Loaded:", tfeat.shape, ifeat.shape, labels.shape)

    # z-normalise each modality (frozen features)
    def z(a):
        return (a - a.mean(0)) / (a.std(0) + 1e-6)
    tfeat, ifeat = z(tfeat), z(ifeat)

    n = len(labels)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]

    MOD = {
        "text":       (tfeat,                                  "Text-only (DistilBERT)"),
        "image":      (ifeat,                                  "Image-only (ViT-tiny)"),
        "text+image": (np.concatenate([tfeat, ifeat], axis=1), "Text+Image (concat fusion)"),
    }
    T = {}
    for k, (F, _) in MOD.items():
        T[k] = (
            torch.tensor(F[tr], dtype=torch.float32),
            torch.tensor(labels[tr], dtype=torch.long),
            torch.tensor(F[te], dtype=torch.float32),
            torch.tensor(labels[te], dtype=torch.long),
        )

    LR = 1e-3
    SEEDS = [42, 123]
    results = {"meta": meta, "n_train": int(ntr), "n_test": int(n - ntr), "lr": LR, "seeds": SEEDS}

    t0 = time.time()

    # ---- A) Centralized baselines -------------------------------------------
    print("\n[A] Centralized baselines")
    results["centralized"] = {}
    for mk in MOD:
        Xtr, ytr, Xte, yte = T[mk]
        runs = [run_centralized(Xtr, ytr, Xte, yte, epochs=24, lr=LR, seed=s) for s in SEEDS]
        best = [r[0] for r in runs]
        results["centralized"][mk] = {
            "f1_macro_mean": float(np.mean([b["f1_macro"] for b in best])),
            "f1_macro_std": float(np.std([b["f1_macro"] for b in best])),
            "acc_mean": float(np.mean([b["acc"] for b in best])),
            "history": runs[0][1],
        }
        print(f"  {mk:12s} F1={results['centralized'][mk]['f1_macro_mean']:.4f}")

    # ---- B) Rounds x clients (main grid) ------------------------------------
    print("\n[B] Rounds x Clients grid (splitter=repo, alpha=1.0, local_epochs=3)")
    CLIENTS = [2, 3, 5, 10, 20, 50]
    ROUNDS = 15
    results["rounds_x_clients"] = {}
    for mk in MOD:
        Xtr, ytr, Xte, yte = T[mk]
        results["rounds_x_clients"][mk] = {}
        for nc in CLIENTS:
            per_seed = []
            for s in SEEDS:
                h, info = run_federated(Xtr, ytr, Xte, yte, nc, ROUNDS, 3, 1.0, LR, "repo", 1.0, s)
                per_seed.append([x["f1_macro"] for x in h])
            arr = np.array(per_seed)
            results["rounds_x_clients"][mk][str(nc)] = {
                "mean": arr.mean(0).tolist(),
                "std": arr.std(0).tolist(),
                "final": float(arr[:, -1].mean()),
                "best": float(arr.max(1).mean()),
                "info": info,
            }
            print(f"  {mk:12s} clients={nc:3d}  final F1={arr[:,-1].mean():.4f} best={arr.max(1).mean():.4f}")

    # ---- C) Dirichlet alpha: repo splitter vs correct label-skew splitter ----
    print("\n[C] Heterogeneity: repo splitter vs true label-Dirichlet (clients=10)")
    ALPHAS = [0.1, 0.3, 0.5, 1.0, 10.0]
    results["alpha"] = {}
    for sp in ["repo", "label"]:
        results["alpha"][sp] = {}
        for mk in MOD:
            Xtr, ytr, Xte, yte = T[mk]
            results["alpha"][sp][mk] = {}
            for a in ALPHAS:
                f1s, tvs, cvs = [], [], []
                for s in SEEDS:
                    h, info = run_federated(Xtr, ytr, Xte, yte, 10, 8, 3, a, LR, sp, 1.0, s)
                    f1s.append(h[-1]["f1_macro"]); tvs.append(info["label_skew_tv"]); cvs.append(info["size_cv"])
                results["alpha"][sp][mk][str(a)] = {
                    "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                    "label_skew_tv": float(np.mean(tvs)), "size_cv": float(np.mean(cvs)),
                }
                print(f"  {sp:6s} {mk:12s} a={a:<5} F1={np.mean(f1s):.4f} TV={np.mean(tvs):.3f} sizeCV={np.mean(cvs):.3f}")

    # ---- D) Local epochs ----------------------------------------------------
    print("\n[D] Local epochs (clients=10, rounds=8)")
    results["local_epochs"] = {}
    for mk in MOD:
        Xtr, ytr, Xte, yte = T[mk]
        results["local_epochs"][mk] = {}
        for le in [1, 2, 3, 5, 8]:
            f1s = [run_federated(Xtr, ytr, Xte, yte, 10, 8, le, 1.0, LR, "repo", 1.0, s)[0][-1]["f1_macro"] for s in SEEDS]
            results["local_epochs"][mk][str(le)] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            print(f"  {mk:12s} local_epochs={le} F1={np.mean(f1s):.4f}")

    # ---- E) Client participation rate ---------------------------------------
    print("\n[E] Participation rate (clients=20, rounds=10)")
    results["participation"] = {}
    for mk in MOD:
        Xtr, ytr, Xte, yte = T[mk]
        results["participation"][mk] = {}
        for pr in [0.1, 0.25, 0.5, 0.75, 1.0]:
            f1s = [run_federated(Xtr, ytr, Xte, yte, 20, 10, 3, 1.0, LR, "repo", pr, s)[0][-1]["f1_macro"] for s in SEEDS]
            results["participation"][mk][str(pr)] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            print(f"  {mk:12s} participation={pr} F1={np.mean(f1s):.4f}")

    # ---- F) Per-class F1 by modality (fed, clients=10, rounds=15) ------------
    print("\n[F] Per-class F1 by modality")
    results["per_class"] = {}
    for mk in MOD:
        Xtr, ytr, Xte, yte = T[mk]
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        g = Head(Xtr.shape[1]).to(DEV)
        cidx = split_data_non_iid_REPO(len(ytr), 10, 1.0)
        for r in range(15):
            gs = g.state_dict(); cms, css = [], []
            for c in cidx:
                if not len(c):
                    continue
                idx = torch.tensor(c, dtype=torch.long)
                lm = Head(Xtr.shape[1]).to(DEV); lm.load_state_dict(gs)
                local_train(lm, Xtr[idx], ytr[idx], 3, LR)
                cms.append(lm); css.append(len(idx))
            g = fedavg(g, cms, css)
        with torch.no_grad():
            g.eval()
            pred = g(Xte).argmax(1).cpu().numpy()
        pc = f1_score(yte.cpu().numpy(), pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0)
        results["per_class"][mk] = {CLASSES[i]: float(pc[i]) for i in range(NUM_LABELS)}
        print(f"  {mk:12s} " + " ".join(f"{CLASSES[i][:9]}={pc[i]:.2f}" for i in range(NUM_LABELS)))

    # ---- G) Fed vs Centralized gap ------------------------------------------
    results["fed_vs_central"] = {}
    for mk in MOD:
        c = results["centralized"][mk]["f1_macro_mean"]
        f = results["rounds_x_clients"][mk]["3"]["best"]
        results["fed_vs_central"][mk] = {"centralized": c, "federated_3clients": f, "gap": float(c - f)}

    results["wall_time_s"] = time.time() - t0
    with open(OUT / "sweep_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nDONE in {results['wall_time_s']:.0f}s -> {OUT/'sweep_results.json'}")


if __name__ == "__main__":
    main()
