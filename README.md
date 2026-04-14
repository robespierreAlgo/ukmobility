# UK mobility simulation repository

This repository contains the code used to run the epidemic simulations for the paper from final preprocessed network inputs.

The full study relied on substantial upstream preprocessing of mobile-network, antenna, and building data. That earlier raw-data preparation is **not** included here. The public repository starts from already processed simulation inputs and reference files, so the epidemic model can be run without rebuilding the full original pipeline.

In addition to running directly from released `people_on_builds` inputs, the repository also includes an **optional** intermediate rebuild step: if you have compatible `filtered_pre_model` files and region dictionaries, the code can reconstruct `people_on_builds` locally using Gurobi.

## Scope of this repository

This repository is intended to support the paper by providing:

- the simulation code
- the paper-facing run scripts
- the evaluation and plotting pipeline
- configuration for local execution
- optional rebuilding of `people_on_builds` from intermediate processed inputs

This repository does **not** include:

- raw mobile-network data
- the original full preprocessing chain from raw data to intermediate processed files
- a bundled Gurobi license

## Repository layout

```text
ukmobility/
├── README.md
├── requirements.txt
├── run_paper.py
├── configs/
│   └── local_config.json
├── scripts/
│   ├── 00_validate_inputs.py
│   ├── 01_build_people_on_builds.py
│   └── 01_run_simulation.py
└── src/
    ├── people_on_builds_builder.py
    └── simulation_generic.py
```

## Input data overview

There are two supported ways to use the repository.

### 1. Minimal run path: start from released `people_on_builds`

This is the simplest path and the intended one for paper reproduction.

Required inputs:

```text
data/
├── processed/
│   └── people_on_builds/
│       ├── Reading_09.json
│       ├── Cambridge_09.json
│       └── Coventry_09.json
└── reference/
    └── daily_cases_read_cam_cov.csv
```

Optional input:

```text
data/
└── reference/
    └── mobility_city_alpha_delta.csv
```

Use this path if you already have the final simulation inputs.

---

### 2. Optional rebuild path: start from intermediate processed inputs

If you do **not** already have `people_on_builds/*.json`, but you do have the compatible intermediate processed files, the repository can build them locally before simulation.

Additional inputs for this path:

```text
data/
└── processed/
    ├── filtered_pre_model/
    │   ├── Reading_09.json
    │   ├── Cambridge_09.json
    │   └── Coventry_09.json
    ├── per_region_dist/
    │   └── region_dicts/
    │       ├── ...Reading....json
    │       ├── ...Cambridge....json
    │       └── ...Coventry....json
    └── model_output/
```

This rebuild step requires:

- `gurobipy`
- a valid local Gurobi license configuration

## Installation

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The public `requirements.txt` covers the simulation and plotting stack.

If you want to use the optional rebuild step from `filtered_pre_model`, you must also install:

```bash
pip install gurobipy
```

and make a valid Gurobi license available locally.

## Configuration

The repository includes `configs/local_config.json`.

Despite the name, this file should be understood as an **example local setup** showing the expected path structure. You can use it directly if your folder layout matches it, or edit it for your own machine.

Current default structure expected by the example config:

```text
../data/
├── processed/
│   ├── people_on_builds/
│   ├── filtered_pre_model/
│   ├── per_region_dist/
│   │   └── region_dicts/
│   └── model_output/
├── reference/
│   ├── daily_cases_read_cam_cov.csv
│   └── mobility_city_alpha_delta.csv
└── gurobi_wls.txt   # only needed for optional Gurobi rebuilds
```

The most important paths in the config are:

- `processed_people_on_builds`
- `filtered_pre_model`
- `region_dicts`
- `model_output_root`
- `real_cases_csv`
- `mobility_csv`
- `results_root`

The committed `local_config.json` is only an example. Adjust paths as needed for your own local environment.

## Validate the setup

Before running the model, check that the expected inputs are present and readable:

```bash
python scripts/00_validate_inputs.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09
```

This checks the configured files and summarizes the available simulation inputs.

## Run the simulations

The public artifact currently supports **one month per run**. In the example setup, this is month `09`, corresponding to the filenames `Reading_09.json`, `Cambridge_09.json`, and `Coventry_09.json`.

### Direct paper-facing entry point

To run the four standard scenarios:

```bash
python run_paper.py --config configs/local_config.json --scenario all
```

You can also run only one scenario:

```bash
python run_paper.py --config configs/local_config.json --scenario alpha_nomob
python run_paper.py --config configs/local_config.json --scenario alpha_mob
python run_paper.py --config configs/local_config.json --scenario delta_nomob
python run_paper.py --config configs/local_config.json --scenario delta_mob
```

If you do not have the mobility file, use only the `*_nomob` scenarios.

---

### Run scenarios manually

You can also call the simulation script directly.

#### Alpha wave, no mobility

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave alpha \
  --mobility off \
  --run-tag alpha_nomob
```

#### Alpha wave, with mobility

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave alpha \
  --mobility on \
  --run-tag alpha_mob
```

#### Delta wave, no mobility

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave delta \
  --mobility off \
  --run-tag delta_nomob
```

#### Delta wave, with mobility

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave delta \
  --mobility on \
  --run-tag delta_mob
```

## Optional: rebuild `people_on_builds` before simulation

If you have `filtered_pre_model` and `region_dicts`, but not final `people_on_builds` files, you can build them first.

### Build explicitly

```bash
python scripts/01_build_people_on_builds.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09
```

### Or let the simulation script build them automatically

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave alpha \
  --mobility off \
  --build-people auto \
  --run-tag alpha_nomob
```

To force rebuilding even if files already exist:

```bash
python scripts/01_run_simulation.py \
  --config configs/local_config.json \
  --cities Reading,Cambridge,Coventry \
  --months 09 \
  --wave alpha \
  --mobility off \
  --build-people auto \
  --force-rebuild-people \
  --run-tag alpha_nomob
```

## Outputs

Each run writes results into its own subfolder under the configured `results_root`.

Example layout:

```text
results/
├── alpha_nomob/
│   ├── plots/
│   │   ├── reading_alpha_onewave.png
│   │   ├── cambridge_alpha_onewave.png
│   │   └── coventry_alpha_onewave.png
│   └── tables/
│       ├── quant_eval_alpha.csv
│       └── quant_eval_tables/
├── alpha_mob/
├── delta_nomob/
└── delta_mob/
```

Main outputs include:

- one epidemic-curve plot per city and scenario
- one wave-level summary CSV
- readable evaluation tables in CSV, TXT, PNG, and PDF
- scenario-specific output folders so runs do not overwrite one another

## What the JSON inputs contain

Each `people_on_builds` JSON file contains the final processed simulation input for one city and one month.

At region level, this includes:

- resident devices assigned to home regions
- visiting-device assignments between regions
- building groupings used to define contact settings
- contact-relevant building categories such as education, office, shop, and restaurant

These files are the direct inputs used by the epidemic simulation.

## Notes on reproducibility

- The public repository is designed to reproduce the simulation runs from released processed inputs, not from raw telco data.
- The optional Gurobi-based rebuild step starts from intermediate processed inputs, not from raw data.
- The repository currently supports one month per run.
- Mobility-enabled runs require `mobility_city_alpha_delta.csv`; mobility-disabled runs do not.
- The exact local folder layout can vary, as long as the paths in `configs/local_config.json` are updated accordingly.

## License

See `LICENSE`.
