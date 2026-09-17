# FarmFederate / LEAF

Tea pest and disease classification using paired leaf images and field notes,
with centralized comparisons, simulated federated learning, and advisory retrieval.

## Active code

| Location | Purpose |
| --- | --- |
| `tea_train.py` | Crop/note datasets, source-grouped splits, text leakage masking, and the ResNet/Transformer multimodal model |
| `experiments/tea_clubbed_tables.py` | Text, image, and ViT/BERT fusion comparisons with centralized and federated heads |
| `experiments/tea_federated_adaptation.py` | Adapt the selected multimodal checkpoint across simulated clients |
| `experiments/tea_federated_all_systems.py` | Federated frozen-encoder probes and local-only controls |
| `experiments/tea_federated_robustness.py` | Label skew, dropout, stale updates, and poisoning experiments |
| `experiments/tea_architecture_ablation.py` | Component ablations |
| `experiments/advisory_retrieval_eval.py` | Offline advisory retrieval evaluation |
| `experiments/make_compact_figures.py` | Generate paper figures from saved experiment results |
| `backend/` | API, RAG modules, and standalone Colab/Kaggle training workflows |
| `frontend/` | Flutter application |
| `tests/` | Existing model, dataset, runtime, and RAG tests |

The frozen ViT/BERT comparisons concatenate encoder features and train a separate
head. They are distinct from the ResNet/Transformer cross-attention model in
`tea_train.py`. Consult each experiment's arguments and implementation before
comparing runs; the heads and evaluation protocols differ.

## Data and results

- `Real Dataset/`: source photographs and YOLO oriented bounding-box annotations.
- `data_final/`: canonical sorted-image/text bundle and `label_schema.json`.
- `tea_results/annotation/annotations.csv`: crop-linked notes used by the audited
  tea experiments.
- `tea_results/`: saved results and run metadata. Keep these for reproducibility.
- `experiments/farm_results_*.json`: recorded architecture and aggregation sweeps.

The five labels are `LEAF_BLIGHT`, `LEAF_HOPPERS`, `LEAF_RUST`,
`LOOPER_CATERPILLARS`, and `MOSQUITO_BUG`.

The former nested `data_final/data_final/` bundle was an exact duplicate and has
been consolidated into `data_final/`. The augmented, studio, and other dataset
collections remain separate because they serve different experiments.

## Weather modality (AMFU Kharagpur station data)

`Weather data/` holds the IMD-style daily observation workbook of the AMFU
Kharagpur observatory (station 42893; 08:30 and 17:30 IST readings, Jan 2024 –
Aug 2026). `backend/weather_data.py` parses it into one record per day (cached at
`backend/data/weather/kgp_42893_daily.csv`), derives rain accumulations, rainless
spells, VPD, heat-day counts and a bucket-model soil-moisture proxy, and exposes
the day three ways to the crop-stress multimodal model in `backend/`:

- **Text**: every training sample's `SENSORS:` line is derived from a real day and
  followed by a `WEATHER (...)` line; a `weather` corpus source adds farmer logs
  written from the observed conditions.
- **Features**: a 19-d station vector (`WEATHER_FEATURES`) feeds a weather branch in
  `MultiModalModel` (a token in the cross-attention key/values plus a gated
  residual). Checkpoints trained without it still load.
- **Labels**: agromet-rule risk labels (`weather_risk_labels`) are merged with the
  keyword weak labels of the paired log.

```sh
cd backend
python multimodal_train.py --weather-mode full        # text + features + rule labels
python multimodal_train.py --weather-mode text-only   # ablation: no numeric branch
python multimodal_train.py --weather-mode none        # legacy synthetic sensors
FARMFED_WEATHER_MODE=full python train_fed_multimodal.py
```

`FARMFED_WEATHER_XLSX=<path>` points at a different workbook; `scripts/remote_train.sh`
runs the same commands on the IIT GPU box through serveo.

## Current paper

The primary manuscript is [overleaf_final/main.tex](overleaf_final/main.tex), with
its compiled [PDF](overleaf_final/main.pdf). The alternate standalone package is
`overleaf_final_slim/`. Both packages include their own required figure photos
under `plots/photos/`.

Build with a TeX installation providing pdfLaTeX, IEEEtran, TikZ, and FontAwesome 5:

```sh
cd overleaf_final
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Use the same commands in `overleaf_final_slim/` for that version. The bibliography
is embedded in the manuscript. LaTeX intermediates are ignored by Git.

## Other maintained entry points

- `Dockerfile` and `render.yaml` use `backend/demo_server.py`.
- `Dockerfile.local` and `docker-compose.yml` describe the local service setup.
- `notebooks/` and the standalone scripts in `backend/` support notebook workflows.
- `paper/`, the root reports/presentations, and `FarmFederate_Globecom/` are
  historical or separate deliverables, not the current LEAF manuscript.
- `FarmFederate_App_Copyright_Ready/` is a distinct application submission bundle;
  it differs from `frontend/` and is retained intentionally.

See [the cleanup record](docs/CODEBASE_CLEANUP.md) for removals and recovery details.
