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
