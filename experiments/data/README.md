# Data Readme

The data we need is basically:

* building data
* carbon intensity data (how dirty the grid is)
* weather data (for making predictions)

This allows CityLearn to learn what carbon intensity the grid is, and what the weather will be like, and use that information to decide whether to store energy or use it.

`.ipynb` files that start with `0x` indicate what processes to run, and in what order. Consider it a poor substitute for a proper data pipeline. You are the pipeline. Be the pipeline.

## Hourly

Data is captured or processed into hourly timesteps across 2023-01-01 00:00 to 2024-01-01 00:00

## Datasets

Datasets are self-contained CityLearn environments that can be used for training. When ready, move files for buildings, weather, and carbon intensity into a directory under `datasets` and reference them in a `schema.json` to use them for training.

## Building

There exists a building on the Aran Islands. It has been anonymized.

## Battery

Battery data is provided for buildings, but in the case of letting CityLearn _control_ the battery, it should not be used in the simulation (aside from the size, efficiency, etc. of the battery itself so CityLearn knows what kind of battery it is controlling).

## CSV Compression

CSVs can get quite large. Use `csv_compressor.ipynb` to convert to parquet data format.


