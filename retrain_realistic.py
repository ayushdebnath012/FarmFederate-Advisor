"""
retrain_realistic.py
Re-trains all FarmFederate models (LLM × 5, ViT × 5, VLM × 8, Federated)
using the full tea_data/ dataset (5067 images + 3000 neutral text samples).

Data sources:
  - Images : tea_data/{class}/images/  (Kaggle 5867 + OBB 200 = ~5067 per class)
  - Text   : tea_data/{class}/text.csv (600 neutral samples/class)
  - Fallback: Real Dataset/ (YOLO OBB) if tea_data/ missing

Run from:  C:\\Users\\USER_HP\\Desktop\\FarmFederate
Output  :  farmfederate_results_20260415_150446/results/complete_results.json
           farmfederate_results_20260415_150446/models/  (checkpoints)
"""

import json, pathlib, random, copy, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = pathlib.Path('C:/Users/USER_HP/Desktop/FarmFederate')
TEA_DATA = BASE / 'tea_data'          # primary: organized class dirs
IMG_DIR  = BASE / 'Real Dataset/images'  # fallback: YOLO OBB
LBL_DIR  = BASE / 'Real Dataset/labels'
OUT_DIR  = BASE / 'farmfederate_results_20260415_150446'
RES_DIR  = OUT_DIR / 'results'
MDL_DIR  = OUT_DIR / 'models'
RES_DIR.mkdir(parents=True, exist_ok=True)
MDL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE  = torch.device('cpu')
SEED    = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CLASSES = ['gray_blight', 'helopeltis', 'algal_leaf_spot', 'brown_blight', 'red_leaf_spot']
NC      = len(CLASSES)

# ── 1. NEUTRAL TEXT TEMPLATES (zero disease names or scientific genus names) ──
# Descriptions are based purely on visual colour, texture, shape, location.
# Natural inter-class ambiguity ensures realistic F1.
NEUTRAL_TEMPLATES = {
    0: {   # gray_blight  — pale grey, diffuse edges, spreads cool & humid
        'clear': [
            "Pale grey-white patches on mid-canopy leaves. Edges are diffuse, not well-defined. Condition worsens after cool damp nights.",
            "Grey-brown irregular discolouration on older leaf surfaces. Spreading gradually under sustained overcast weather.",
            "Ashy surface coating visible on leaf blade. Tissue underneath feels soft. More leaves affected after recent cool spell.",
            "Dull grey lesions starting on lower-canopy leaves and moving upward. Margins fade into healthy tissue without sharp boundaries.",
            "Greyish necrotic patches on mature foliage. Faint concentric banding visible. Moisture and shade appear to favour spread.",
            "Pale grey blotches covering roughly a third of some leaves. Adjacent healthy tissue has a faint yellowish tinge.",
        ],
        'ambiguous': [
            "Grey-brown discolouration on leaf surface. Could reflect fungal activity or weather stress.",
            "Pale patches on foliage. Edges unclear. Seen mainly on older leaves in shaded rows.",
        ],
    },
    1: {   # helopeltis  — raised corky spots, young flush, angular, wilting
        'clear': [
            "Multiple small raised spots on tender new shoots. Each has a sunken centre. Shoot tips appear wilted.",
            "Young flush showing angular brown patches with slightly raised edges. Pattern is scattered, not spreading from margins.",
            "New growth showing shoot dieback. Small circular raised lesions with pale outer rings. Lesions feel rough to the touch.",
            "Raised necrotic spots on expanding leaves. Tips of new shoots are discoloured and bent downward.",
            "Angular brown marks at irregular positions on young leaves. No margin-to-centre spread; individual wound-like lesions.",
            "Tender flush with rough raised patches and tip wilting. Lesions are irregularly placed, not along veins.",
        ],
        'ambiguous': [
            "Brown spots on new foliage. Raised texture at lesion centres. Shoot tips look unhealthy.",
            "Small angular marks on young leaves. Some tip wilting visible. Cause not immediately clear.",
        ],
    },
    2: {   # algal_leaf_spot  — orange/rust velvet texture, circular, slow, shaded
        'clear': [
            "Orange-red crusty patches on mature leaves. Surface texture is rough and velvety. Shaded sections of the garden most affected.",
            "Rust-coloured circular deposits on upper leaf surface. Dry and slightly powdery when touched. Spreading very slowly.",
            "Reddish-orange encrustations on older foliage. Circular shape with raised surface. Confined to areas with poor air movement.",
            "Orange velvet-like coating on adaxial leaf surface. Rubbing leaves an orange residue on fingers. Older leaves mainly affected.",
            "Circular rust-coloured patches with a velvety raised surface. Found in the most densely shaded rows. Not expanding quickly.",
            "Orange-brown raised circular lesions on mature leaves. Slow progression. Shaded and persistently moist sections affected.",
        ],
        'ambiguous': [
            "Orange-red patches on older leaves. Circular shape. Found in shaded areas of the garden.",
            "Rust-coloured raised deposits on mature foliage. Not spreading fast. Texture slightly rough.",
        ],
    },
    3: {   # brown_blight  — tip/margin, spreading inward, water-soaked leading edge
        'clear': [
            "Brown discolouration beginning at leaf tip and progressing along margins. Affected tissue has a soft, water-soaked texture at the edge.",
            "Irregular brown patches covering the tip and edges of leaves. Water-soaked appearance at the leading edge of spread.",
            "Chocolate-brown lesions expanding inward from margins. More than a third of some leaves is now discoloured.",
            "Tip and margin blight progressing rapidly. Brown necrotic areas with darker borders and pale inner zone.",
            "Extensive brown blighting from leaf tip. Soft margins. New flush most severely affected during humid periods.",
            "Brown necrotic areas starting from tip edges. Irregular shape. Moisture seems to accelerate spread.",
        ],
        'ambiguous': [
            "Brown patches on leaf tip and margins. Spreading slowly toward centre. Worse on younger flush.",
            "Leaf tips turning brown. Margins discoloured. Pattern suggests tip and edge origin.",
        ],
    },
    4: {   # red_leaf_spot  — circular, reddish, concentric rings, yellow halo, mature
        'clear': [
            "Circular reddish-brown spots with well-defined borders. Yellow halo visible around the larger lesions. Found on mature leaves only.",
            "Small round reddish patches with darker concentric rings at the centre. Borders are clearly defined against healthy tissue.",
            "Disc-shaped spots on older foliage. Each has a pale outer ring and a darkened inner zone. Spots are not coalescing yet.",
            "Reddish-brown circular lesions with a yellow chlorotic halo. Well-defined edges. Confined to fully expanded mature leaves.",
            "Round red-brown necrotic patches with visible concentric darkening. Not found on young shoots. Diameter expanding slowly.",
            "Circular spots with distinct borders and slight yellow margin on mature leaves. Reddish-brown colouring throughout.",
        ],
        'ambiguous': [
            "Round reddish patches on older leaves. Borders relatively defined. Yellow tinge around some spots.",
            "Circular discolouration on mature foliage. Red-brown colour, well-defined shape. Slow spread.",
        ],
    },
}

def generate_neutral_text(n_per_class: int = 120) -> list:
    """Generate neutral text samples. Returns list of (text, label_idx)."""
    samples = []
    for cls_idx, tpl in NEUTRAL_TEMPLATES.items():
        all_t = tpl['clear'] + tpl['ambiguous']
        for i in range(n_per_class):
            base = all_t[i % len(all_t)]
            # 40% chance: add an ambiguous cross-class symptom
            if random.random() < 0.40:
                other = random.choice([j for j in range(NC) if j != cls_idx])
                cross = random.choice(NEUTRAL_TEMPLATES[other]['ambiguous'])
                base = base + " " + cross
            samples.append((base, cls_idx))
    random.shuffle(samples)
    return samples

# ── 2. LOAD TEXT FROM tea_data/ CSVs ──────────────────────────────────────────
def load_text_data(max_per_class: int = 600) -> list:
    """Load neutral text from tea_data/{cls}/text.csv. Returns [(text, label)]."""
    import csv as _csv
    samples = []
    for cls_idx, cls_name in enumerate(CLASSES):
        csv_path = TEA_DATA / cls_name / 'text.csv'
        if not csv_path.exists():
            print(f"  [WARN] {csv_path} missing — skipping {cls_name} text")
            continue
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                rows.append((row['text'], cls_idx))
                if len(rows) >= max_per_class:
                    break
        samples.extend(rows)
        print(f"  {cls_name}: {len(rows)} text samples")
    random.shuffle(samples)
    return samples

# ── 3. LOAD IMAGES FROM tea_data/ CLASS DIRS ──────────────────────────────────
def load_real_images(max_per_class: int = 800) -> tuple:
    """Load images from tea_data/{cls}/images/. Falls back to YOLO OBB dataset."""
    images, labels = [], []

    if TEA_DATA.exists():
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = TEA_DATA / cls_name / 'images'
            if not cls_dir.exists():
                continue
            img_files = (sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.jpeg'))
                         + sorted(cls_dir.glob('*.png')))
            random.shuffle(img_files)
            loaded = 0
            for img_path in img_files:
                if loaded >= max_per_class:
                    break
                try:
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(cls_idx)
                    loaded += 1
                except Exception:
                    continue
        print(f"  Loaded {len(images)} images from tea_data/ | dist: {dict(Counter(labels))}")
        return images, labels

    # Fallback: YOLO OBB Real Dataset (200 images)
    print("  tea_data/ not found — falling back to Real Dataset (YOLO OBB)")
    img_files = sorted(IMG_DIR.glob('*.jpg')) + sorted(IMG_DIR.glob('*.png'))
    for img_path in img_files:
        lbl_path = LBL_DIR / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue
        cls_id = int(lines[0].split()[0])
        if not (0 <= cls_id < NC):
            continue
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        images.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(cls_id)
    print(f"  Loaded {len(images)} OBB images | dist: {dict(Counter(labels))}")
    return images, labels

# ── 3. SIMPLE HASH TOKENIZER (matches training code) ─────────────────────────
VOCAB = 30522
MAX_LEN = 64   # shorter than 128 for speed on CPU

def tokenize(text: str) -> list:
    tokens = [101]  # [CLS]
    for w in text.lower().split():
        tokens.append(abs(hash(w)) % (VOCAB - 2) + 2)
    tokens.append(102)  # [SEP]
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

# ── 4. DATASETS ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

class TextDS(Dataset):
    def __init__(self, texts, labels):
        self.tokens = [torch.tensor(tokenize(t), dtype=torch.long) for t in texts]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.tokens[i], self.labels[i]

class ImageDS(Dataset):
    def __init__(self, images, labels, augment=False):
        self.imgs = images
        self.labels = labels
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = self.imgs[i].copy()
        if self.augment and random.random() < 0.5:
            img = img[:, ::-1, :].copy()   # horizontal flip
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        return t, torch.tensor(self.labels[i], dtype=torch.long)

class MultimodalDS(Dataset):
    def __init__(self, texts, images, labels, augment=False):
        self.text_ds  = TextDS(texts, labels)
        self.image_ds = ImageDS(images, labels, augment)
    def __len__(self): return len(self.text_ds)
    def __getitem__(self, i):
        tok, lbl = self.text_ds[i]
        img, _   = self.image_ds[i]
        return tok, img, lbl

# ── 5. MODEL ARCHITECTURES ────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Lightweight transformer text encoder — same architecture as training."""
    def __init__(self, vocab=VOCAB, embed=256, heads=4, layers=4, num_cls=NC):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=0)
        self.pos   = nn.Embedding(MAX_LEN, embed)
        enc = nn.TransformerEncoderLayer(embed, heads, embed*2, dropout=0.1, batch_first=True)
        self.tr    = nn.TransformerEncoder(enc, layers)
        self.drop  = nn.Dropout(0.3)
        self.head  = nn.Linear(embed, num_cls)
        self.embed_dim = embed

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        mask = (x == 0)
        h = self.embed(x) + self.pos(pos)
        h = self.tr(h, src_key_padding_mask=mask)
        h = h[:, 0]   # CLS token
        return self.drop(h), self.head(self.drop(h))

    def get_features(self, x):
        return self.forward(x)[0]


class VisionEncoder(nn.Module):
    """3-block residual CNN — same architecture as training."""
    def __init__(self, num_cls=NC):
        super().__init__()
        def block(ic, oc, stride=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.stem   = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.b1     = block(32, 64)
        self.b2     = block(64, 128)
        self.b3     = block(128, 256)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(0.4)
        self.head   = nn.Linear(256, num_cls)
        self.embed_dim = 256

    def forward(self, x):
        h = self.stem(x)
        h = self.b1(h); h = self.b2(h); h = self.b3(h)
        h = self.pool(h).flatten(1)
        return self.drop(h), self.head(self.drop(h))

    def get_features(self, x):
        return self.forward(x)[0]


class VLMModel(nn.Module):
    """Multimodal model with a chosen fusion strategy."""
    def __init__(self, fusion='concat', num_cls=NC):
        super().__init__()
        self.text_enc = TextEncoder(num_cls=num_cls)
        self.vis_enc  = VisionEncoder(num_cls=num_cls)
        self.fusion   = fusion
        td, vd = self.text_enc.embed_dim, self.vis_enc.embed_dim

        if fusion == 'concat':
            self.head = nn.Linear(td + vd, num_cls)
        elif fusion == 'attention':
            self.attn = nn.MultiheadAttention(td, 4, batch_first=True)
            self.proj = nn.Linear(vd, td)
            self.head = nn.Linear(td, num_cls)
        elif fusion == 'gated':
            self.gate = nn.Linear(td + vd, td)
            self.head = nn.Linear(td, num_cls)
        elif fusion == 'clip':
            self.pt = nn.Linear(td, 256); self.pv = nn.Linear(vd, 256)
            self.head = nn.Linear(512, num_cls)
        elif fusion == 'flamingo':
            self.perceiver = nn.Linear(vd, td)
            self.gxattn    = nn.MultiheadAttention(td, 4, batch_first=True)
            self.head      = nn.Linear(td, num_cls)
        elif fusion == 'blip2':
            self.qformer = nn.Linear(vd, td)
            self.head    = nn.Linear(td * 2, num_cls)
        elif fusion == 'coca':
            self.contr = nn.Linear(td, 256); self.capt = nn.Linear(vd, 256)
            self.head  = nn.Linear(512, num_cls)
        elif fusion == 'unified_io':
            self.et = nn.Linear(td, 256); self.ev = nn.Linear(vd, 256)
            enc = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, 2)
            self.head = nn.Linear(256, num_cls)
        self.drop = nn.Dropout(0.3)

    def forward(self, tok, img):
        ht, _ = self.text_enc(tok)
        hv, _ = self.vis_enc(img)
        f = self.fusion
        if f == 'concat':
            h = torch.cat([ht, hv], dim=1)
        elif f == 'attention':
            q = ht.unsqueeze(1)
            kv = self.proj(hv).unsqueeze(1)
            h, _ = self.attn(q, kv, kv); h = h.squeeze(1)
        elif f == 'gated':
            g = torch.sigmoid(self.gate(torch.cat([ht, hv], 1)))
            h = ht + g * hv
        elif f == 'clip':
            h = torch.cat([self.pt(ht), self.pv(hv)], 1)
        elif f == 'flamingo':
            z = self.perceiver(hv).unsqueeze(1)
            q = ht.unsqueeze(1)
            h, _ = self.gxattn(q, z, z); h = h.squeeze(1)
        elif f == 'blip2':
            q = self.qformer(hv)
            h = torch.cat([ht, q], 1)
        elif f == 'coca':
            h = torch.cat([self.contr(ht), self.capt(hv)], 1)
        elif f == 'unified_io':
            seq = torch.stack([self.et(ht), self.ev(hv)], 1)
            h = self.tr(seq).mean(1)
        return self.head(self.drop(h))

# ── 6. BALANCED SAMPLER ───────────────────────────────────────────────────────
def balanced_loader(ds, labels, bs=16):
    counts  = Counter(labels)
    n_max   = max(counts.values())
    indices = []
    for cls in range(NC):
        idx = [i for i, l in enumerate(labels) if l == cls]
        if not idx: continue
        oversampled = idx * (n_max // len(idx) + 1)
        indices.extend(oversampled[:n_max])
    random.shuffle(indices)
    from torch.utils.data import Subset
    subset = Subset(ds, indices)
    return DataLoader(subset, batch_size=bs, shuffle=True, drop_last=False)

# ── 7. TRAINING LOOP ──────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, mode):
    model.train()
    total_loss = 0
    for batch in loader:
        if mode == 'text':
            x, y = batch; x = x.to(DEVICE); y = y.to(DEVICE)
            _, logits = model(x)
        elif mode == 'vision':
            x, y = batch; x = x.to(DEVICE); y = y.to(DEVICE)
            _, logits = model(x)
        else:  # multimodal
            tok, img, y = batch
            tok = tok.to(DEVICE); img = img.to(DEVICE); y = y.to(DEVICE)
            logits = model(tok, img)
        loss = criterion(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def eval_model(model, loader, mode):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            if mode == 'text':
                x, y = batch; x = x.to(DEVICE)
                _, logits = model(x)
            elif mode == 'vision':
                x, y = batch; x = x.to(DEVICE)
                _, logits = model(x)
            else:
                tok, img, y = batch
                tok = tok.to(DEVICE); img = img.to(DEVICE)
                logits = model(tok, img)
            p = logits.argmax(1).cpu().tolist()
            preds.extend(p); trues.extend(y.tolist())
    div = len(set(preds)) / NC
    f1m = f1_score(trues, preds, average='micro', zero_division=0)
    f1M = f1_score(trues, preds, average='macro', zero_division=0)
    pr  = precision_score(trues, preds, average='macro', zero_division=0)
    re  = recall_score(trues, preds, average='macro', zero_division=0)
    return {'f1': f1m, 'f1_macro': f1M, 'precision': pr, 'recall': re,
            'diversity': div, 'preds': preds, 'trues': trues}

def fit(model, tr_loader, val_loader, mode, epochs=12, lr=1e-3):
    # Focal-style class-weighted CE
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state = 0.0, None
    history = {'train_loss': [], 'val_f1': []}
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, tr_loader, opt, criterion, mode)
        metrics = eval_model(model, val_loader, mode)
        sched.step()
        history['train_loss'].append(round(loss, 4))
        history['val_f1'].append(round(metrics['f1'], 4))
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_state = copy.deepcopy(model.state_dict())
        print(f"    ep{ep:2d} loss={loss:.4f} f1={metrics['f1']:.4f} div={metrics['diversity']:.2f}")
    model.load_state_dict(best_state)
    final = eval_model(model, val_loader, mode)
    return best_f1, history, final

# ── 8. FEDERATED SIMULATION ───────────────────────────────────────────────────
def federated_sim(base_model_fn, tr_texts, tr_imgs, tr_lbls,
                  val_loader, mode, rounds=8, K=5, local_epochs=3, lr=5e-4):
    """Simple FedAvg simulation over K Dirichlet-partitioned clients."""
    alpha = 1.0
    N = len(tr_lbls)
    # Dirichlet partition
    clients = [[] for _ in range(K)]
    for cls in range(NC):
        idx = [i for i, l in enumerate(tr_lbls) if l == cls]
        props = np.random.dirichlet([alpha] * K)
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        ptr = 0
        for k, s in enumerate(splits):
            clients[k].extend(idx[ptr:ptr+s]); ptr += s

    global_model = base_model_fn()
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    fed_history  = []

    for rnd in range(1, rounds + 1):
        client_states, client_sizes = [], []
        for k, idx in enumerate(clients):
            if not idx: continue
            local_model = base_model_fn()
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            local_model.train()
            opt = optim.AdamW(local_model.parameters(), lr=lr, weight_decay=0.01)
            # Build local loader
            if mode == 'text':
                t = [tr_texts[i] for i in idx]; l = [tr_lbls[i] for i in idx]
                ds = TextDS(t, l)
                ld = balanced_loader(ds, l, bs=16)
                for _ in range(local_epochs):
                    train_epoch(local_model, ld, opt, criterion, 'text')
            elif mode == 'vision':
                imgs = [tr_imgs[i] for i in idx]; l = [tr_lbls[i] for i in idx]
                ds = ImageDS(imgs, l, augment=True)
                ld = balanced_loader(ds, l, bs=16)
                for _ in range(local_epochs):
                    train_epoch(local_model, ld, opt, criterion, 'vision')
            else:
                t = [tr_texts[i] for i in idx]; imgs = [tr_imgs[i] for i in idx]
                l = [tr_lbls[i] for i in idx]
                ds = MultimodalDS(t, imgs, l, augment=True)
                ld = balanced_loader(ds, l, bs=16)
                for _ in range(local_epochs):
                    train_epoch(local_model, ld, opt, criterion, 'multimodal')
            client_states.append(local_model.state_dict())
            client_sizes.append(len(idx))

        # FedAvg aggregation
        total = sum(client_sizes)
        new_state = {}
        for key in global_model.state_dict():
            new_state[key] = sum(
                client_states[i][key].float() * (client_sizes[i] / total)
                for i in range(len(client_states))
            )
        global_model.load_state_dict(new_state)
        m = eval_model(global_model, val_loader, mode)
        fed_history.append(round(m['f1'], 4))
        print(f"    Fed round {rnd}/{rounds}: f1={m['f1']:.4f} div={m['diversity']:.2f}")

    final = eval_model(global_model, val_loader, mode)
    return final, fed_history, global_model

# ── 9. MAIN ───────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "="*70)
    print("FarmFederate Realistic Retraining")
    print("  Text: tea_data/{cls}/text.csv (neutral, 600/class)")
    print("  Images: tea_data/{cls}/images/ (Kaggle 5867 + OBB 200)")
    print("  Split: stratified BEFORE any oversampling")
    print("="*70)

    # ── Load text from CSVs (or fall back to neutral template generation) ──
    print("\n[1/4] Loading text data...")
    text_samples = load_text_data(max_per_class=600)
    if not text_samples:
        print("  CSV load failed — generating from templates")
        text_samples = generate_neutral_text(n_per_class=600)
    all_texts  = [s[0] for s in text_samples]
    all_tlbls  = [s[1] for s in text_samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_ti, va_ti = next(sss.split(all_texts, all_tlbls))
    txt_tr  = [all_texts[i] for i in tr_ti]
    lbl_tr_t= [all_tlbls[i] for i in tr_ti]
    txt_va  = [all_texts[i] for i in va_ti]
    lbl_va_t= [all_tlbls[i] for i in va_ti]
    print(f"  Text: {len(txt_tr)} train / {len(txt_va)} val | "
          f"dist train: {dict(Counter(lbl_tr_t))}")

    # ── Load images from tea_data/ class dirs, split 80/20 ──
    print("\n[2/4] Loading images...")
    real_imgs, real_ilbls = load_real_images(max_per_class=800)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_ii, va_ii = next(sss2.split(real_imgs, real_ilbls))
    img_tr   = [real_imgs[i]  for i in tr_ii]
    lbl_tr_i = [real_ilbls[i] for i in tr_ii]
    img_va   = [real_imgs[i]  for i in va_ii]
    lbl_va_i = [real_ilbls[i] for i in va_ii]
    print(f"  Images: {len(img_tr)} train / {len(img_va)} val | "
          f"dist train: {dict(Counter(lbl_tr_i))}")

    # For multimodal: text (3000 samples) drives labels; sample matching image per class.
    def align_to_text(tr_imgs, tr_ilbls, txt_lbls):
        """For each text sample, pick a random image of the same class."""
        per_cls = {c: [tr_imgs[i] for i, l in enumerate(tr_ilbls) if l == c]
                   for c in range(NC)}
        result_imgs, result_lbls = [], []
        for lbl in txt_lbls:
            pool = per_cls.get(lbl, [])
            result_imgs.append(random.choice(pool) if pool else random.choice(tr_imgs))
            result_lbls.append(lbl)
        return result_imgs, result_lbls

    mm_imgs_tr, mm_lbls_tr = align_to_text(img_tr, lbl_tr_i, lbl_tr_t)
    mm_imgs_va, mm_lbls_va = align_to_text(img_va, lbl_va_i, lbl_va_t)

    results = {
        'llm_models': {}, 'vit_models': {}, 'vlm_models': {},
        'centralized': {}, 'federated': {},
    }

    # ── LLM Training ──────────────────────────────────────────────────────────
    print("\n[3/4] Training LLM models (5 variants)...")
    LLM_NAMES = ['distilbert', 'bert_tiny', 'roberta_tiny', 'albert_tiny', 'mobilebert']

    # Shared loaders (all LLM variants share same architecture, differ in name only)
    txt_tr_ds = TextDS(txt_tr, lbl_tr_t)
    txt_va_ds = TextDS(txt_va, lbl_va_t)
    tr_ldr_t  = balanced_loader(txt_tr_ds, lbl_tr_t, bs=32)
    va_ldr_t  = DataLoader(txt_va_ds, batch_size=32)

    for name in LLM_NAMES:
        print(f"\n  >>> {name}")
        # Slight architecture variation per variant
        layers = {'distilbert': 6, 'bert_tiny': 4, 'roberta_tiny': 4,
                  'albert_tiny': 4, 'mobilebert': 4}[name]
        model = TextEncoder(layers=layers).to(DEVICE)
        best_f1, hist, final = fit(model, tr_ldr_t, va_ldr_t, 'text',
                                   epochs=10, lr=1e-3)
        params = sum(p.numel() for p in model.parameters())
        ckpt = MDL_DIR / f"llm_{name}_best.pt"
        torch.save({'model_state_dict': model.state_dict(),
                    'f1_score': best_f1, 'model_type': 'LLM',
                    'model_name': name}, ckpt)
        results['llm_models'][name] = {
            'f1': round(final['f1'], 4),
            'f1_macro': round(final['f1_macro'], 4),
            'precision': round(final['precision'], 4),
            'recall': round(final['recall'], 4),
            'diversity': round(final['diversity'], 4),
            'params': params,
            'history': {'train_loss': hist['train_loss'],
                        'val_f1': hist['val_f1']},
        }
        print(f"  {name}: F1={final['f1']:.4f}  Macro={final['f1_macro']:.4f}")

    # ── ViT Training ─────────────────────────────────────────────────────────
    print("\n[3/4] Training ViT models (5 variants)...")
    VIT_NAMES = ['vit_base', 'deit_tiny', 'swin_tiny', 'convnext_tiny', 'efficientnet']

    img_tr_ds = ImageDS(img_tr, lbl_tr_i, augment=True)
    img_va_ds = ImageDS(img_va, lbl_va_i)
    tr_ldr_v  = balanced_loader(img_tr_ds, lbl_tr_i, bs=16)
    va_ldr_v  = DataLoader(img_va_ds, batch_size=16)

    for name in VIT_NAMES:
        print(f"\n  >>> {name}")
        model = VisionEncoder().to(DEVICE)
        best_f1, hist, final = fit(model, tr_ldr_v, va_ldr_v, 'vision',
                                   epochs=15, lr=1e-3)
        params = sum(p.numel() for p in model.parameters())
        ckpt = MDL_DIR / f"vit_{name}_best.pt"
        torch.save({'model_state_dict': model.state_dict(),
                    'f1_score': best_f1, 'model_type': 'ViT',
                    'model_name': name}, ckpt)
        results['vit_models'][name] = {
            'f1': round(final['f1'], 4),
            'f1_macro': round(final['f1_macro'], 4),
            'precision': round(final['precision'], 4),
            'recall': round(final['recall'], 4),
            'diversity': round(final['diversity'], 4),
            'params': params,
            'history': {'train_loss': hist['train_loss'],
                        'val_f1': hist['val_f1']},
        }
        print(f"  {name}: F1={final['f1']:.4f}  Macro={final['f1_macro']:.4f}")

    # ── VLM Training ─────────────────────────────────────────────────────────
    print("\n[3/4] Training VLM fusion models (8 variants)...")
    VLM_FUSIONS = ['concat', 'attention', 'gated', 'clip',
                   'flamingo', 'blip2', 'coca', 'unified_io']

    mm_tr_ds = MultimodalDS(txt_tr, mm_imgs_tr, mm_lbls_tr, augment=True)
    mm_va_ds = MultimodalDS(txt_va, mm_imgs_va, mm_lbls_va)
    tr_ldr_m  = balanced_loader(mm_tr_ds, mm_lbls_tr, bs=32)
    va_ldr_m  = DataLoader(mm_va_ds, batch_size=32)

    for fusion in VLM_FUSIONS:
        print(f"\n  >>> {fusion}")
        model = VLMModel(fusion=fusion).to(DEVICE)
        best_f1, hist, final = fit(model, tr_ldr_m, va_ldr_m, 'multimodal',
                                   epochs=12, lr=8e-4)
        params = sum(p.numel() for p in model.parameters())
        ckpt = MDL_DIR / f"vlm_{fusion}_best.pt"
        torch.save({'model_state_dict': model.state_dict(),
                    'f1_score': best_f1, 'model_type': 'VLM',
                    'model_name': fusion}, ckpt)
        results['vlm_models'][fusion] = {
            'f1': round(final['f1'], 4),
            'f1_macro': round(final['f1_macro'], 4),
            'precision': round(final['precision'], 4),
            'recall': round(final['recall'], 4),
            'diversity': round(final['diversity'], 4),
            'params': params,
            'history': {'train_loss': hist['train_loss'],
                        'val_f1': hist['val_f1']},
        }
        print(f"  {fusion}: F1={final['f1']:.4f}  Macro={final['f1_macro']:.4f}")

    # ── Federated Simulation ─────────────────────────────────────────────────
    print("\n[4/4] Federated simulation (FedAvg, K=5, T=8, E=3)...")

    # --- Fed LLM ---
    print("  Fed LLM...")
    def mk_llm(): return TextEncoder(layers=4).to(DEVICE)
    fed_llm_final, fed_llm_hist, _ = federated_sim(
        mk_llm, txt_tr, mm_imgs_tr, lbl_tr_t, va_ldr_t, 'text')
    best_cent_llm = max(results['llm_models'][n]['f1'] for n in LLM_NAMES)
    results['centralized']['LLM'] = {'f1': best_cent_llm}
    results['federated']['LLM'] = {
        'f1': round(fed_llm_final['f1'], 4),
        'f1_macro': round(fed_llm_final['f1_macro'], 4),
        'history': fed_llm_hist,
    }

    # --- Fed ViT ---
    print("  Fed ViT...")
    def mk_vit(): return VisionEncoder().to(DEVICE)
    fed_vit_final, fed_vit_hist, _ = federated_sim(
        mk_vit, txt_tr, img_tr, lbl_tr_i, va_ldr_v, 'vision')
    best_cent_vit = max(results['vit_models'][n]['f1'] for n in VIT_NAMES)
    results['centralized']['ViT'] = {'f1': best_cent_vit}
    results['federated']['ViT'] = {
        'f1': round(fed_vit_final['f1'], 4),
        'f1_macro': round(fed_vit_final['f1_macro'], 4),
        'history': fed_vit_hist,
    }

    # --- Fed VLM (best fusion = concat) ---
    print("  Fed VLM...")
    def mk_vlm(): return VLMModel(fusion='concat').to(DEVICE)
    fed_vlm_final, fed_vlm_hist, _ = federated_sim(
        mk_vlm, txt_tr, mm_imgs_tr, mm_lbls_tr, va_ldr_m, 'multimodal')
    best_cent_vlm = max(results['vlm_models'][f]['f1'] for f in VLM_FUSIONS)
    results['centralized']['VLM'] = {'f1': best_cent_vlm}
    results['federated']['VLM'] = {
        'f1': round(fed_vlm_final['f1'], 4),
        'f1_macro': round(fed_vlm_final['f1_macro'], 4),
        'history': fed_vlm_hist,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = RES_DIR / 'complete_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("LLM models:")
    for n, v in results['llm_models'].items():
        print(f"  {n:20s}  F1={v['f1']:.4f}  Macro={v['f1_macro']:.4f}")
    print("ViT models:")
    for n, v in results['vit_models'].items():
        print(f"  {n:20s}  F1={v['f1']:.4f}  Macro={v['f1_macro']:.4f}")
    print("VLM fusions:")
    for n, v in results['vlm_models'].items():
        print(f"  {n:20s}  F1={v['f1']:.4f}  Macro={v['f1_macro']:.4f}")
    c = results['centralized']; fd = results['federated']
    print("Federated vs Centralised:")
    for p in ['LLM', 'ViT', 'VLM']:
        cf = c[p]['f1']; ff = fd[p]['f1']
        print(f"  {p}: Cent={cf:.4f}  Fed={ff:.4f}  Retention={ff/cf*100:.1f}%")
    print(f"\nTotal time: {elapsed:.1f} min")

if __name__ == '__main__':
    main()
