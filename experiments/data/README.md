# Data Readme

## Datasets

Datasets are self-contained CityLearn environments that can be used for training. When ready, move files for buildings, weather, and carbon intensity into a directory under `datasets` and reference them in a `schema.json` to use them for training.

## Regenerating Data

Go through and run these Jupyter notebooks in order:

1. 01_preprocess/preprocess_building_real.ipynb
2. 02_standard/standard.ipynb
3. 03_clean/clean_real_building.ipynb
4. 00_baseline.ipynb
5. 01_dirty__building_power_dips.ipynb
6. 02_dirty__building_outliers.ipynb
7. (optional) visualise_building_data.ipynb - to see what is contained within building data

Then, move these files:

1. 03_clean/Building_01_baseline.csv -> datasets/baseline/Building_1_pristine.csv
2. 03_clean/Building_01_dirty_01_power_dips.csv -> datasets/baseline/building_1_missing.csv
3. 03_clean/Building_01_dirty_02_power_outliers.csv -> datasets/baseline/building_1_outliers.csv

Then you can re-run experiments.