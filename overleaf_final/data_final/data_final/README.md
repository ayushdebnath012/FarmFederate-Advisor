# FarmFederate Colab Data Bundle

This bundle contains the real image and separately supplied symptom-text datasets used by
`backend/FarmFederate_Colab_Complete.py`.

## Contents

- `real_dataset_sorted/`: 200 real tea images in five class folders.
- `text_data/annotations.csv`: 3,000 class-labelled text observations.
- `text_data/<CLASS>/text.csv`: 600 observations for each class.

Images and texts share class labels but are not verified observations of the
same leaf. The script therefore reports VLM evaluation as label-aligned proxy
pairs and never as sample-linked multimodal field data.

Class order:

0. `LEAF_BLIGHT`
1. `LEAF_HOPPERS`
2. `LEAF_RUST`
3. `LOOPER_CATERPILLARS`
4. `MOSQUITO_BUG`

Label mapping version: `farmfederate_stress_v2_direct_yolo_ids`.
Raw YOLO IDs map directly as `0=LEAF_BLIGHT`, `1=LEAF_HOPPERS`,
`2=LEAF_RUST`, `3=LOOPER_CATERPILLARS`, and `4=MOSQUITO_BUG`.

## Google Colab

Upload `data_final.zip` to `MyDrive/FarmFederate/data_final.zip`. The training
script automatically finds and extracts that location after Google Drive is
mounted. It can also be supplied explicitly:

```bash
python backend/FarmFederate_Colab_Complete.py --train \
  --data-final-zip "/content/drive/MyDrive/FarmFederate/data_final.zip"
```
