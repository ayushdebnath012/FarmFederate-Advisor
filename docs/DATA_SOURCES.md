# DATA_SOURCES.md

This document lists candidate data sources and instructions for acquiring datasets used by FarmFederate.

## Image datasets by label

- water_stress
  - Candidate: `ashokpant/drought-detection` (Kaggle) — https://www.kaggle.com/datasets/ashokpant/drought-detection
  - Note: public drought datasets are less common; if you have institutional sources or UAV thermal data, place them in `drought/`.

- nutrient_def
  - Candidate: `PlantDoc` (PlantDoc repository / Kaggle mirror). Try `pratik2901/plantdoc-dataset` and GitHub mirrors.
  - Local folder: `plantdoc/`

- pest_risk
  - Candidate: `IP102` (large insect dataset) — GitHub: `PKU-ICST-MIPL/IP102` (mirror archives sometimes available). Local folder: `ip102/`
  - Fallback: `plant_seedlings` datasets or `Plant_Seedlings` Kaggle competition.

- disease_risk
  - Candidates: `PlantVillage` (emmarex/plantdisease), `plant-pathology-2020-fgvc7` (competition), `Crop_Disease`.
  - Local folders: `plantvillage/`, `plant_pathology/`, `crop_disease/`

- heat_stress
  - Candidate sources: thermal imaging datasets are rare and often in research repos. Check university pages or contact dataset owners. Place in `heat_stress/`.

## How to provide data to the script

- Preferred: place unzipped dataset folder under the expected path (e.g., `plantvillage/PlantVillage`, `plant_pathology/images`, `plant_seedlings/`). The script will automatically discover images recursively.
- Kaggle: if you want the script to download automatically, configure the Kaggle CLI beforehand:
  - Create `~/.kaggle/kaggle.json` with your API credentials, or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
  - You can also run `kaggle datasets download -d <id> -p <dest> --unzip` yourself and place results in the dest folder.

## Running acquisition offline (recommended test first)

- Preview downloads: `python FarmFederate_Kaggle_Complete.py --dry-run`
- Attempt acquisition only: `python FarmFederate_Kaggle_Complete.py --download-only`
- Generate local folder report: `python scripts/check_datasets.py`

## Notes

- Some Kaggle datasets require you to "join" a competition or accept terms before downloading (e.g., certain plant pathology competitions). The script may get 401/403 in such cases; manual download may be required.
- If a required dataset is not publicly available, consider providing a small representative sample in the expected folder to enable testing and development.
