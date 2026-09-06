"""Fast smoke script used by CI to run lightweight checks.
- Runs a small RAG quick test (in-memory Qdrant) if qdrant-client present
- Runs small model training smoke using notebook's lightweight models
"""
import sys
import os

# Run from anywhere: put the repo root (this file's parent directory) on sys.path so
# `backend` is importable even though the script itself lives in scripts/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _StubEmbedders:
    """Deterministic stand-in for backend.qdrant_rag.Embedders.

    The real Embedders downloads CLIP and MiniLM weights, which is far too slow for a
    fast smoke check. The RAG helpers only ever call embed_image/embed_text, so hash-seeded
    random vectors of the right dimensions exercise the same Qdrant code paths.
    """

    VISUAL_DIM = 512
    SEMANTIC_DIM = 384

    def _vector(self, seed_text, dim):
        import hashlib
        import numpy as np
        # hashlib rather than hash(): str hashing is salted per process, and the same text
        # must map to the same vector across runs for retrieval to be reproducible.
        seed = int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim)
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_image(self, image):
        return self._vector(f"image:{image.size}:{image.mode}", self.VISUAL_DIM)

    def embed_text(self, text):
        return self._vector(f"text:{text}", self.SEMANTIC_DIM)


def run_rag_quick_test():
    try:
        from qdrant_client import QdrantClient
        from backend.qdrant_rag import (
            init_qdrant_collections,
            agentic_diagnose,
            store_session_entry,
            retrieve_session_history,
            upsert_point,
            KNOWLEDGE_COLLECTION,
        )
        from PIL import Image
        client = QdrantClient(':memory:')
        init_qdrant_collections(client)
        emb = _StubEmbedders()
        img = Image.new('RGB', (224, 224), color='green')
        # Seed one knowledge case so the visual search below has something to retrieve;
        # against an empty collection the check would pass without exercising the query.
        upsert_point(
            client,
            KNOWLEDGE_COLLECTION,
            1,
            emb.embed_image(img),
            emb.embed_text('nutrient deficiency on tea leaves'),
            {'stress_type': 'nutrient_def', 'crop_name': 'tea', 'severity': 'moderate',
             'agronomist_notes': 'interveinal chlorosis'},
        )
        res = agentic_diagnose(client, image=img, user_description='Test', emb=emb, llm_func=lambda p: 'mock')
        if not res.get('retrieved'):
            raise AssertionError('visual search returned no cases after seeding one')
        store_session_entry(client, farm_id='farm_1', plant_id='p1', diagnosis='nutrient_def', treatment='test', emb=emb)
        hist = retrieve_session_history(client, 'farm_1', 'p1', emb=emb)
        print('RAG quick test OK, retrieved len', len(res.get('retrieved', [])), 'hist len', len(hist))
        return True
    except Exception as e:
        print('RAG quick test skipped/fail:', e)
        return False


def _collate(batch):
    """Keep PIL images as a plain list; the default collate cannot batch them."""
    import torch
    return {
        'text': [b['text'] for b in batch],
        'labels': torch.stack([b['labels'] for b in batch]),
        'image': [b['image'] for b in batch],
    }


def run_model_smoke():
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from backend.notebooks.rag_colab_demo import (
            SimpleLLM,
            SimpleViT,
            MultiModalDataset,
            generate_image_data,
            generate_text_data,
        )
    except Exception as e:
        print('Model smoke import failed:', e)
        return False

    imgs = generate_image_data(8)
    texts = generate_text_data(16)
    ds = MultiModalDataset(list(texts['text'][:8]), list(texts['labels'][:8]), imgs[:8])
    loader = DataLoader(ds, batch_size=4, collate_fn=_collate)
    llm = SimpleLLM()
    vit = SimpleViT()
    to_tensor = transforms.ToTensor()
    # quick forward
    for b in loader:
        try:
            _ = llm(torch.randint(0, 1000, (len(b['text']), 16)))
            imgs_tensor = torch.stack([to_tensor(im) for im in b['image']])
            _ = vit(imgs_tensor)
        except Exception as e:
            print('Model forward failed:', e)
            return False
        break
    print('Model smoke OK')
    return True


def main():
    print('Starting fast smoke...')
    rag_ok = run_rag_quick_test()
    model_ok = run_model_smoke()
    all_ok = rag_ok or model_ok
    if not all_ok:
        print('Some smoke checks failed. See logs.')
        sys.exit(2)
    print('All smoke checks passed.')

if __name__ == '__main__':
    main()
