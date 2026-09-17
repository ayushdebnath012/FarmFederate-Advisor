"""Synthetic tea-crop -> weather-day mapping: determinism and label independence."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "backend"))

import weather_data as wd  # noqa: E402
import tea_weather_mapping as twm  # noqa: E402
import tea_weather_advisory as twa  # noqa: E402

HAVE_DATA = (REPO / "Weather data").exists() or Path(wd.DEFAULT_CACHE).exists()
ANN = REPO / "tea_results" / "annotation" / "annotations.csv"
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="station data not available")


def _sample_ids():
    ann = pd.read_csv(ANN)
    return list(zip(ann["image_file"].astype(str), ann["box_idx"].astype(int))), ann


def test_assignment_is_deterministic_and_seed_sensitive():
    sid = ("100_jpg.rf.deadbeef.jpg", 0)
    a = twm.assign_index(sid, 974, seed=42)
    assert a == twm.assign_index(sid, 974, seed=42)          # stable across calls
    assert 0 <= a < 974
    # a different seed or a different crop generally lands elsewhere
    assert twm.assign_index(sid, 974, seed=7) != a or twm.assign_index(("x.jpg", 3), 974, 42) != a


def test_assign_index_cannot_see_the_label():
    """Structural guarantee: the label is not a parameter of the assignment."""
    import inspect
    params = set(inspect.signature(twm.assign_index).parameters)
    assert params == {"sample_id", "n_days", "seed"}
    assert not (params & {"label", "cls_id", "disease", "y"})


@pytest.mark.skipif(not ANN.exists(), reason="tea annotations not available")
def test_manifest_covers_every_crop_and_spreads_over_the_record():
    sids, ann = _sample_ids()
    man = twm.build_manifest(sids)
    assert len(man) == len(ann)
    assert man["assigned_date"].nunique() > 100, "assignment collapsed onto few days"
    assert man["tmean"].notna().all()


@pytest.mark.skipif(not ANN.exists(), reason="tea annotations not available")
def test_weather_is_independent_of_class():
    """
    The invariant that keeps this usable: assigned weather must carry no information
    about the label. Two checks -- per-class means agree, and weather-derived risk
    for a crop's own class matches the base rate rather than beating it.
    """
    sids, ann = _sample_ids()
    man = twm.build_manifest(sids)
    m = ann.merge(man, on=["image_file", "box_idx"])

    means = m.groupby("disease")[["tmean", "sunshine_hours", "rh_am"]].mean()
    assert means["tmean"].max() - means["tmean"].min() < 2.0
    assert means["sunshine_hours"].max() - means["sunshine_hours"].min() < 1.5
    assert means["rh_am"].max() - means["rh_am"].min() < 6.0

    own = np.mean([twa.class_risk(r["disease"], twa.weather_for_date(r["assigned_date"]))[0] == "high"
                   for _, r in m.iterrows()])
    base = np.mean([
        np.mean([twa.class_risk(k, twa.weather_for_date(d))[0] == "high" for k in twa.TEA_CLASSES])
        for d in m["assigned_date"].unique()[:150]
    ])
    # if own-class risk beat the base rate materially, a signal was manufactured
    assert own - base < 0.10, f"own-class high-risk {own:.3f} vs base {base:.3f} -- leakage"


def test_weathermap_lookup_matches_manifest():
    wm = twm.WeatherMap()
    sid = ("100_jpg.rf.14fa87ff511c6790858ccbc03bf10be6.jpg", 0)
    assert wm.features_for(sid).shape == (wd.WEATHER_DIM,)
    assert wm.date_for(sid) == twm.build_manifest([sid])["assigned_date"].iloc[0]
    assert wm.dim == wd.WEATHER_DIM


def test_manifest_file_is_labelled_synthetic():
    """The persisted map must announce that it is not capture provenance."""
    p = Path(twm.DEFAULT_MANIFEST)
    if not p.exists():
        pytest.skip("manifest not built")
    head = p.read_text()[:300].upper()
    assert "SYNTHETIC" in head and "NOT CAPTURE PROVENANCE" in head


# ---------------------------------------------------------------------------
# Weather as a third modality in the LEAF model
# ---------------------------------------------------------------------------
def _tea_model(**kw):
    import tea_train as T
    defaults = dict(max_seq_len=32, pretrained_vision=False, vision_backbone="lightweight")
    defaults.update(kw)
    return T.MultiModalClassifier(**defaults)


def _batch(b=4, seq=32, w=None):
    import torch
    out = dict(
        input_ids=torch.randint(1, 30000, (b, seq)),
        attention_mask=torch.ones(b, seq, dtype=torch.long),
        pixel_values=torch.randn(b, 3, 224, 224),
        labels=torch.randint(0, 5, (b,)),
    )
    if w:
        out["weather_features"] = torch.randn(b, w)
    return out


def test_weather_off_leaves_the_reported_architecture_unchanged():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO))
    m = _tea_model()
    assert m.use_weather is False
    assert m.n_modalities == 2 and m.n_blocks == 6
    assert not hasattr(m, "w_enc") and not hasattr(m, "weather_head")
    out = m(**_batch())
    assert out["modality_weights"].shape[-1] == 2
    assert out["weather_logits"] is None
    assert "weather_auxiliary" not in out["loss_components"]


def test_weather_on_adds_a_third_routed_modality():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO))
    m = _tea_model(use_weather=True, weather_dim=wd.WEATHER_DIM)
    assert m.n_modalities == 3 and m.n_blocks == 7
    out = m(**_batch(w=wd.WEATHER_DIM))
    assert out["modality_weights"].shape[-1] == 3
    # reliability weights remain a distribution over the present modalities
    assert torch.allclose(out["modality_weights"].sum(-1), torch.ones(4), atol=1e-5)
    assert out["weather_logits"].shape == (4, 5)
    assert "weather_auxiliary" in out["loss_components"]
    out["loss"].backward()
    assert m.w_enc[1].weight.grad.abs().sum().item() > 0


def test_missing_weather_vector_is_routed_around():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO))
    m = _tea_model(use_weather=True, weather_dim=wd.WEATHER_DIM)
    out = m(**_batch())  # weather enabled but no vector supplied
    assert float(out["modality_weights"][:, 2].abs().max()) < 1e-6


def test_two_wide_modality_mask_from_existing_callers_is_padded():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO))
    m = _tea_model(use_weather=True, weather_dim=wd.WEATHER_DIM)
    b = _batch(w=wd.WEATHER_DIM)
    b["modality_mask"] = torch.tensor([0.0, 1.0]).expand(4, -1)  # vision-warmup style
    out = m(**b)
    assert out["modality_weights"].shape[-1] == 3
