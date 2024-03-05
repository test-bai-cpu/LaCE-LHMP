# LaCE-LHMP

## Data
Use first 10 days of [ATC dataset](https://dil.atr.jp/crest2010_HRI/ATC_dataset/), first day for train and the rest for evaluation. The data is in `atc_data/middle_area` folder.

Use middle area of ATC dataset. This middle area has dimensions where x ranges from -25m to 0m and y ranges from -10m to 15m, amounting to an area of 625m<sup>2</sup>, which is shown in the figure below.

<img src="figures/middle_area1.png" alt="Alt text for the figure" title="Optional title" width="200" />

## Baseline
Use [CLiFF-LHMP](https://ieeexplore.ieee.org/document/10342031) and [Trajectron++](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_40) as baselines. The config file of training Trajectron++ is attached in `/baselines/trajectron++_config.json`


## LaCE-LHMP
