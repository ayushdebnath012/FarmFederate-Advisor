"""Weather modality: workbook parsing, agromet rules, corpus grounding and the model branch."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "backend"))

import weather_data as wd  # noqa: E402

XLSX = REPO / "Weather data" / "@ Daily Met data_KGP (42893) Obs Station 2024.xlsx"
HAVE_XLSX = XLSX.exists() or Path(wd.DEFAULT_CACHE).exists()


def _synthetic_daily(n=40):
    """A small daily table with the columns the rules/features expect."""
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    d = pd.DataFrame({
        "date": dates, "tmax": np.linspace(30, 41, n), "tmin": np.full(n, 24.0),
        "rh_am": np.full(n, 70.0), "rh_pm": np.full(n, 30.0), "rain_mm": np.zeros(n),
        "db_am": np.full(n, 27.0), "db_pm": np.full(n, 35.0), "dew_point": np.full(n, 18.0),
        "vapour_pressure": np.full(n, 20.0), "wind_speed": np.full(n, 6.0), "wind_dir": "SW",
        "evaporation": np.full(n, 6.0), "soil_temp_5cm": np.full(n, 30.0),
        "soil_temp_10cm": np.full(n, 29.0), "sunshine_hours": np.full(n, 8.0),
    }).set_index("date")
    return wd.add_derived_indices(d)


def test_cell_coercion_tolerates_typos():
    assert wd._num("23..0", "tmax") == 23.0
    assert wd._num("29./83", "db_temp") == 29.0
    assert np.isnan(wd._num("26.625.1", "rain_mm")) or wd._num("26.625.1", "rain_mm") == 26.625
    assert np.isnan(wd._num(0, "tmin"))          # a typed 0 C is a missing reading at Kharagpur
    assert np.isnan(wd._num(None, "tmax"))
    assert wd._wind_dir("29.6SW") is None and wd._wind_dir("ESE") == "ESE"


def test_title_regex_matches_all_sheet_variants():
    for title, exp in [
        ("JAN 2024(05.30 P.M)", (1, 2024, "pm")), ("FEB-2024(8.30A.M)", (2, 2024, "am")),
        ("APRIL- 2024 (5.30P.M)", (4, 2024, "pm")), ("JULY -2025 (5.30 P.M)", (7, 2025, "pm")),
        ("SEPT 2024(8.30 A.M)", (9, 2024, "am")), ("MARCH-2025(8.30 A.M)", (3, 2025, "am")),
    ]:
        assert wd._parse_title([np.nan, title]) == exp
    assert wd._parse_title(["MEAN", 20.5]) is None


def test_derived_indices_and_rules_on_synthetic_days():
    d = _synthetic_daily()
    assert d["rainless_days"].iloc[-1] == len(d)          # never rained
    assert (d["soil_moisture_pct"].diff().dropna() <= 0).all()  # bucket only drains
    assert d["vpd_kpa"].between(0, 8).all()
    last = d.iloc[-1]
    labs = wd.weather_risk_labels(last)
    assert wd.LABEL_TO_ID["heat_stress"] in labs           # Tmax 41 C
    assert wd.LABEL_TO_ID["water_stress"] in labs          # long rainless spell, high VPD, warm
    assert wd.LABEL_TO_ID["disease_risk"] not in labs
    wet = last.copy()
    wet["rain_mm"], wet["rain_7d"], wet["rh_am"], wet["rh_pm"], wet["tmean"] = 30.0, 60.0, 92.0, 80.0, 27.0
    assert wd.LABEL_TO_ID["disease_risk"] in wd.weather_risk_labels(wet)


def test_feature_vector_and_text_avoid_keyword_leak():
    d = _synthetic_daily()
    row = d.iloc[-1]
    vec = wd.weather_feature_vector(row)
    assert vec.shape == (wd.WEATHER_DIM,) and vec.dtype == np.float32 and np.abs(vec).max() <= 4.0
    line = wd.weather_summary_line(row, imputed=["rh_am"])
    assert line.startswith("WEATHER (") and "RH ~70%" in line
    sensors = wd.sensor_line_from_weather(row, np.random.default_rng(0))
    assert sensors.startswith("SENSORS:") and "\nWEATHER (" in sensors
    # words the weak labeller keys on must not appear in the machine lines
    for bad in ("dry", "hot", "moisture", "wilt", "rot", "rust"):
        assert bad not in line.lower()


@pytest.mark.skipif(not HAVE_XLSX, reason="KGP workbook not available")
def test_real_workbook_parses_into_daily_records():
    daily = wd.load_kgp_weather()
    assert len(daily) > 700 and daily["observed"].sum() > 700
    assert daily["tmax"].between(15, 48).all()
    assert daily["date"].is_monotonic_increasing and daily["date"].diff().dropna().eq(pd.Timedelta("1D")).all()
    sampler = wd.WeatherSampler(daily)
    support = sampler.label_support()
    assert all(v > 20 for v in support.values()), support
    row, feat, labs, imp = sampler.sample(prefer_labels=[wd.LABEL_TO_ID["heat_stress"]], p_match=1.0)
    assert wd.LABEL_TO_ID["heat_stress"] in labs and feat.shape == (wd.WEATHER_DIM,)


def test_weak_labels_ignore_sensor_and_weather_lines():
    import datasets_loader as dl
    fused = ("SENSORS: soil_moisture=30.0%, soil_pH=6.5, temp=29.0°C, humidity=60%, VPD=1.4 kPa, "
             "rainfall_24h=0.0mm (trend: →).\nWEATHER (AMFU Kharagpur/IMD 42893, 2024-05-01): Tmax=39.0°C.\n"
             "LOG: Aphids and honeydew on the undersides of tomato leaves in the field.")
    labs = dl.weak_labels(fused)
    assert labs == [dl.LABEL_TO_ID["pest_risk"]]          # no water_stress from "soil_moisture"


@pytest.mark.skipif(not HAVE_XLSX, reason="KGP workbook not available")
def test_corpus_carries_weather_column_in_full_mode_only():
    import datasets_loader as dl
    dl.configure_weather("full")
    df = dl.build_text_corpus_mix(mix_sources="localmini,weather", max_per_source=100, max_samples=50)
    assert list(df.columns) == ["text", "labels", "weather"]
    assert all(len(w) == wd.WEATHER_DIM for w in df["weather"])
    assert df["text"].str.contains("WEATHER (AMFU Kharagpur", regex=False).all()
    dl.configure_weather("text-only")
    df = dl.build_text_corpus_mix(mix_sources="localmini", max_per_source=100, max_samples=20)
    assert list(df.columns) == ["text", "labels"] and df["text"].str.contains("WEATHER (", regex=False).all()
    dl.configure_weather("none")
    df = dl.build_text_corpus_mix(mix_sources="localmini", max_per_source=100, max_samples=20)
    assert list(df.columns) == ["text", "labels"] and not df["text"].str.contains("WEATHER (", regex=False).any()


def test_model_weather_branch_shapes_and_backward_compat(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import RobertaConfig, RobertaModel, ViTConfig, ViTModel
    from multimodal_model import MultiModalModel

    tdir, vdir = str(tmp_path / "t"), str(tmp_path / "v")
    RobertaModel(RobertaConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32, max_position_embeddings=64, type_vocab_size=1)).save_pretrained(tdir)
    ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                       image_size=32, patch_size=16)).save_pretrained(vdir)
    model = MultiModalModel(text_model_name=tdir, image_model_name=vdir).eval()
    ids = torch.randint(2, 100, (3, 8))
    mask = torch.ones_like(ids)
    px = torch.randn(3, 3, 32, 32)
    w = torch.randn(3, wd.WEATHER_DIM)
    with torch.no_grad():
        out = model(ids, mask, px, weather_features=w, return_attention=True)
        assert out.logits.shape == (3, 5)
        assert out.attention_weights["text_to_image"].shape == (3, 1, 2)  # image + weather tokens
        assert model(ids, mask, px).logits.shape == (3, 5)                  # weather-free path
    # a checkpoint trained before the weather branch existed still loads
    old = {k: v for k, v in model.state_dict().items() if not k.startswith("weather")}
    missing, unexpected = MultiModalModel(text_model_name=tdir, image_model_name=vdir).load_state_dict(old, strict=False)
    assert not unexpected and all(k.startswith("weather") for k in missing)
