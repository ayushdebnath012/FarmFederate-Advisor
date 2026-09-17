"""Weather-conditioned tea advisory: rules, payload shape, and the no-leakage boundary."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "backend"))

import weather_data as wd  # noqa: E402
import tea_weather_advisory as twa  # noqa: E402

HAVE_DATA = (REPO / "Weather data").exists() or Path(wd.DEFAULT_CACHE).exists()


def _day(**over):
    """A single observed day with sane defaults, overridable per test."""
    base = dict(date=pd.Timestamp("2024-07-15"), tmean=27.0, tmax=31.0, tmin=23.0,
                rh_am=80.0, rh_pm=60.0, rh_mean=70.0, rain_mm=0.0, rain_7d=0.0,
                rainless_days=0.0, sunshine_hours=5.0)
    base.update(over)
    return pd.Series(base)


def test_blister_blight_needs_wet_dull_and_mild():
    # classic high-risk window: humid, almost no sunshine, mild
    hi = _day(rh_mean=90.0, sunshine_hours=1.0, tmean=22.0)
    assert twa.class_risk("LEAF_RUST", hi)[0] == "high"
    # same humidity but hot -> outside Exobasidium's range
    assert twa.class_risk("LEAF_RUST", _day(rh_mean=90.0, sunshine_hours=1.0, tmean=32.0))[0] != "high"
    # same humidity but bright -> leaf wetness does not persist
    assert twa.class_risk("LEAF_RUST", _day(rh_mean=90.0, sunshine_hours=9.0, tmean=22.0))[0] == "low"


def test_blight_tracks_sustained_rain():
    assert twa.class_risk("LEAF_BLIGHT", _day(rain_7d=90.0, rh_am=90.0, tmean=27.0))[0] == "high"
    assert twa.class_risk("LEAF_BLIGHT", _day(rain_7d=30.0, rh_mean=80.0))[0] == "moderate"
    assert twa.class_risk("LEAF_BLIGHT", _day(rain_7d=0.0, rh_mean=50.0))[0] == "low"


def test_hoppers_suppressed_by_heavy_rain():
    warm_humid_dry = _day(tmean=28.0, rh_mean=80.0, rain_mm=0.0, rainless_days=3)
    assert twa.class_risk("LEAF_HOPPERS", warm_humid_dry)[0] == "high"
    washed_off = _day(tmean=28.0, rh_mean=80.0, rain_mm=40.0, rainless_days=0)
    assert twa.class_risk("LEAF_HOPPERS", washed_off)[0] == "low"


def test_looper_favours_warm_settled_spells():
    assert twa.class_risk("LOOPER_CATERPILLARS", _day(tmean=28.0, rainless_days=5, sunshine_hours=7.0))[0] == "high"
    assert twa.class_risk("LOOPER_CATERPILLARS", _day(tmean=18.0, rainless_days=0))[0] == "low"


def test_helopeltis_favours_humid_overcast_warmth():
    assert twa.class_risk("MOSQUITO_BUG", _day(tmean=27.0, rh_mean=85.0, rain_mm=0.0, sunshine_hours=3.0))[0] == "high"
    assert twa.class_risk("MOSQUITO_BUG", _day(tmean=27.0, rh_mean=40.0, sunshine_hours=9.0))[0] == "low"


def test_every_class_scores_and_unknown_class_rejected():
    d = _day()
    for k in twa.TEA_CLASSES:
        level, drivers = twa.class_risk(k, d)
        assert level in ("high", "moderate", "low") and drivers
    with pytest.raises(ValueError):
        twa.class_risk("LEAF_SPOT", d)


def test_tea_classes_match_the_label_schema():
    import json
    schema = json.load(open(REPO / "data_final" / "label_schema.json"))
    assert sorted(twa.TEA_CLASSES) == sorted(schema["class_order"])


@pytest.mark.skipif(not HAVE_DATA, reason="station data not available")
def test_advise_payload_and_out_of_range_date():
    out = twa.advise("MOSQUITO_BUG", "2024-08-10", model_confidence=0.81)
    assert out["weather_available"] is True
    assert out["observation_date"] == "2024-08-10"
    assert out["weather_risk"] in ("high", "moderate", "low")
    assert out["model_confidence"] == 0.81
    assert "did not influence the model" in out["caveat"]
    assert "AMFU Kharagpur" in out["observation"]
    # a date the station never covered degrades gracefully instead of inventing weather
    missing = twa.advise("LEAF_RUST", "1999-01-01")
    assert missing["weather_available"] is False
    assert twa.weather_for_date("1999-01-01") is None


@pytest.mark.skipif(not HAVE_DATA, reason="station data not available")
def test_risk_table_covers_all_classes_and_varies_over_the_year():
    t = twa.risk_table("2024-08-10")
    assert set(t) == set(twa.TEA_CLASSES)
    # the rules must discriminate: not every day can be high risk for every class
    levels = {d: twa.risk_table(d) for d in ("2024-01-15", "2024-04-15", "2024-08-10")}
    seen = {v["risk"] for day in levels.values() for v in day.values()}
    assert len(seen) > 1, "risk rules are degenerate"


def test_advisory_is_not_wired_into_training():
    """The leakage boundary: no training/eval module may import this one."""
    training = ["tea_train.py", "backend/multimodal_train.py", "backend/train_fed_multimodal.py",
                "backend/federated_core.py", "backend/datasets_loader.py"]
    for rel in training:
        p = REPO / rel
        if p.exists():
            assert "tea_weather_advisory" not in p.read_text(), f"{rel} must not import the advisory module"
