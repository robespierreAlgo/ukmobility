# UK mobility simulation repository

This repository contains the code required to run the epidemic simulations used in the paper from the final preprocessed network inputs.

The full study relied on substantial upstream preprocessing of mobile-network, antenna, and building data. That earlier preparation is not included here. Instead, the repository starts from the final simulation input files in `people_on_builds/`, so the model can be run directly and the outputs in the paper can be reproduced without rebuilding the full preprocessing chain.

## Repository layout

```text
ukmobility_repo_paper_ready/
  README.md
  requirements.txt
  configs/
    example_config.json
  scripts/
    00_validate_inputs.py
    01_run_simulation.py
    _common.py
  src/
    simulation_generic.py
  data/
    processed/
      people_on_builds/
    reference/
      daily_cases_read_cam_cov.csv
      mobility_city_alpha_delta.csv   # optional
```

## Required input files

Place the preprocessed simulation inputs and reference files in this layout:

```text
data/
  processed/
    people_on_builds/
      Reading_09.json
      Cambridge_09.json
      Coventry_09.json
  reference/
    daily_cases_read_cam_cov.csv
    mobility_city_alpha_delta.csv   # optional
```

Required:
- `data/processed/people_on_builds/*.json`
- `data/reference/daily_cases_read_cam_cov.csv`
- `src/simulation_generic.py` is already bundled in the repository

Optional:
- `data/reference/mobility_city_alpha_delta.csv` if you want to run with mobility enabled

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a local config file:

```bash
cp configs/example_config.json configs/local_config.json
```

Edit `configs/local_config.json` if your `data/` folder lives somewhere else.

## Check that the inputs are present

```bash
python scripts/00_validate_inputs.py   --config configs/local_config.json   --cities Reading,Cambridge,Coventry   --months 09
```

## Run the simulations

The commands below run the full set of paper-facing simulations and keep outputs in separate folders so nothing is overwritten.

### Alpha wave, no mobility

```bash
python scripts/01_run_simulation.py   --config configs/local_config.json   --cities Reading,Cambridge,Coventry   --months 09   --wave alpha   --mobility off   --run-tag alpha_nomob
```

### Alpha wave, with mobility

```bash
python scripts/01_run_simulation.py   --config configs/local_config.json   --cities Reading,Cambridge,Coventry   --months 09   --wave alpha   --mobility on   --run-tag alpha_mob
```

### Delta wave, no mobility

```bash
python scripts/01_run_simulation.py   --config configs/local_config.json   --cities Reading,Cambridge,Coventry   --months 09   --wave delta   --mobility off   --run-tag delta_nomob
```

### Delta wave, with mobility

```bash
python scripts/01_run_simulation.py   --config configs/local_config.json   --cities Reading,Cambridge,Coventry   --months 09   --wave delta   --mobility on   --run-tag delta_mob
```

If you do not have the mobility file, simply skip the two `--mobility on` runs.

## Outputs

Each run writes to its own subfolder under `results/`.

Example:

```text
results/
  alpha_nomob/
    plots/
      Reading_alpha_onewave.png
      Cambridge_alpha_onewave.png
      Coventry_alpha_onewave.png
    tables/
      quant_eval_alpha.csv
      quant_eval_tables/
  alpha_mob/
    plots/
    tables/
  delta_nomob/
    plots/
      Reading_delta_onewave.png
      Cambridge_delta_onewave.png
      Coventry_delta_onewave.png
    tables/
      quant_eval_delta.csv
      quant_eval_tables/
  delta_mob/
    plots/
    tables/
```

The main outputs are:
- one epidemic curve plot per city and run
- one summary CSV per wave
- readable evaluation tables in CSV, TXT, PNG, and PDF under `tables/quant_eval_tables/`

## What the JSON inputs contain

Each `people_on_builds` file is the final preprocessed simulation input. It contains the region-level structure used by the model, including the resident devices assigned to each region, visiting-device assignments, and the buildings used to define workplace, education, shopping, and restaurant contact settings.

## Notes

- The month is taken from the filename suffix, for example `Reading_09.json`.
- The repository intentionally starts from the final preprocessed simulation inputs rather than the raw mobility-processing stages used earlier in the study.
