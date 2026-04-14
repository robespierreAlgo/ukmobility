#!/usr/bin/env python3
"""Run the UK mobility simulation from preprocessed people_on_builds inputs."""


import os
import sys
import copy
import argparse
import warnings

from _common import add_common_args, canonical_city_name, ensure_dir, get_cities, get_single_month, load_config, resolve_optional_path, resolve_path, resolved_gurobi_options

from collections import Counter
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
import matplotlib.dates as mdates


# --- Quantitative evaluation helpers ---------------------------------
METRICS = []

def compute_quant_eval(
    *,
    city: str,
    wave: str,
    dates: pd.DatetimeIndex,
    sim_plot: np.ndarray,   # series plotted (e.g., smoothed)
    obs_plot: np.ndarray,   # same transform as sim_plot
    sim_raw: np.ndarray,    # unsmoothed daily (scaled) for totals/peak
    obs_raw: np.ndarray,    # unsmoothed daily for totals/peak
    mape_min_real: float = 5.0,
) -> dict:
    sim_plot = np.asarray(sim_plot, dtype=float)
    obs_plot = np.asarray(obs_plot, dtype=float)
    sim_raw  = np.asarray(sim_raw,  dtype=float)
    obs_raw  = np.asarray(obs_raw,  dtype=float)

    # Errors on the plotted (visual) series
    err = sim_plot - obs_plot
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # MAPE with guard for small observed values (based on raw obs threshold)
    mask = obs_raw >= float(mape_min_real)
    if np.any(mask):
        mape = float(np.mean(np.abs((sim_plot[mask] - obs_plot[mask]) / obs_plot[mask])) * 100.0) \
               if np.all(obs_plot[mask] != 0) else float("nan")
    else:
        mape = float("nan")

    # Pearson correlation on plotted series
    if np.std(sim_plot) > 0 and np.std(obs_plot) > 0:
        pearson_r = float(np.corrcoef(sim_plot, obs_plot)[0, 1])
    else:
        pearson_r = float("nan")

        # Peak timing shift (use 7-day rolling mean on RAW daily series)
    sim7 = pd.Series(sim_raw, index=dates).rolling(7, min_periods=1).mean()
    obs7 = pd.Series(obs_raw, index=dates).rolling(7, min_periods=1).mean()
    d_sim = sim7.idxmax()
    d_obs = obs7.idxmax()
    peak_shift_days = int((d_sim - d_obs).days)
    peak_date_sim = str(pd.Timestamp(d_sim).date())
    peak_date_obs = str(pd.Timestamp(d_obs).date())

    # Total cases error over the wave (RAW sums)
    tot_sim = float(sim_raw.sum())
    tot_obs = float(obs_raw.sum())
    total_pct_err = float((tot_sim - tot_obs) / tot_obs * 100.0) if tot_obs > 0 else float("nan")

    return dict(
        wave=wave,
        city=city,
        start=str(dates.min().date()),
        end=str(dates.max().date()),
        mae=mae,
        rmse=rmse,
        mape_ge5=mape,
        pearson_r=pearson_r,
        peak_date_sim=peak_date_sim,
        peak_date_obs=peak_date_obs,
        peak_shift_days=peak_shift_days,
        total_pct_err=total_pct_err,
    )
    
def _pretty_metrics_tables(df: pd.DataFrame):
    """Return two human-readable tables: errors + shape/timing."""
    d = df.copy()

    # nice formatting
    d["MAE"] = d["mae"].map(lambda x: f"{x:.1f}")
    d["RMSE"] = d["rmse"].map(lambda x: f"{x:.1f}")
    d["MAPE (real=5)"] = d["mape_ge5"].map(lambda x: ("" if pd.isna(x) else f"{x:.1f}%"))
    d["Total err"] = d["total_pct_err"].map(lambda x: ("" if pd.isna(x) else f"{x:+.1f}%"))

    d["Pearson r"] = d["pearson_r"].map(lambda x: ("" if pd.isna(x) else f"{x:.3f}"))
    d["Peak (obs, 7d)"] = d["peak_date_obs"].astype(str)
    d["Peak (sim, 7d)"] = d["peak_date_sim"].astype(str)
    d["Peak shift"] = d["peak_shift_days"].map(lambda x: f"{int(x):+d} d")

    # index and compact columns
    d = d.set_index("city")

    err_tbl = d[["MAE", "RMSE", "MAPE (real=5)", "Total err"]].copy()
    shape_tbl = d[["Pearson r", "Peak (obs, 7d)", "Peak (sim, 7d)", "Peak shift"]].copy()

    return err_tbl, shape_tbl


def _save_table_figure(df: pd.DataFrame, path: str, title: str):
    """Save a clean matplotlib table as PNG/PDF."""
    import matplotlib.pyplot as plt

    # height grows with rows; width with cols (readable default)
    nrows, ncols = df.shape
    fig_w = max(10.0, 1.6 * ncols + 4.0)
    fig_h = max(2.5, 0.55 * nrows + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        rowLabels=list(df.index),
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(22)
    tbl.scale(1.0, 1.7)
    ax.set_title(title, fontsize=28, pad=24)
    for cell in tbl.get_celld().values():
        cell.set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_quant_eval_tables(metrics_rows: list, out_root: str, wave_slug: str):
    """
    Save:
      - raw metrics CSV
      - two readable tables (errors + shape) as CSV/TXT/PNG/PDF
    into: out_root/quant_eval_tables/<wave_slug>/
    """
    if not metrics_rows:
        return

    df = pd.DataFrame(metrics_rows)
    out_dir = os.path.join(out_root, "quant_eval_tables", wave_slug)
    os.makedirs(out_dir, exist_ok=True)

    # raw metrics (machine-readable)
    df.to_csv(os.path.join(out_dir, f"quant_eval_{wave_slug}_raw.csv"), index=False)

    # pretty tables
    err_tbl, shape_tbl = _pretty_metrics_tables(df)

    # also add an average row (quick overview)
    # (only for numeric metrics; keep blanks for dates)
    try:
        avg = df.copy()
        avg_row = {
            "city": "MEAN",
            "mae": avg["mae"].mean(),
            "rmse": avg["rmse"].mean(),
            "mape_ge5": avg["mape_ge5"].mean(skipna=True),
            "total_pct_err": avg["total_pct_err"].mean(skipna=True),
            "pearson_r": avg["pearson_r"].mean(skipna=True),
            "peak_date_obs": "",
            "peak_date_sim": "",
            "peak_shift_days": int(round(avg["peak_shift_days"].mean())),
        }
        df2 = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        err_tbl, shape_tbl = _pretty_metrics_tables(df2)
    except Exception:
        pass

    # save as CSV + TXT
    err_tbl.to_csv(os.path.join(out_dir, f"quant_eval_{wave_slug}_errors.csv"))
    shape_tbl.to_csv(os.path.join(out_dir, f"quant_eval_{wave_slug}_shape.csv"))

    with open(os.path.join(out_dir, f"quant_eval_{wave_slug}_errors.txt"), "w") as f:
        f.write(err_tbl.to_string())

    with open(os.path.join(out_dir, f"quant_eval_{wave_slug}_shape.txt"), "w") as f:
        f.write(shape_tbl.to_string())

    # save figures (PNG + PDF)
    _save_table_figure(
        err_tbl,
        os.path.join(out_dir, f"quant_eval_{wave_slug}_errors.png"),
        title=f"Quantitative evaluation — Errors ({wave_slug})",
    )
    _save_table_figure(
        shape_tbl,
        os.path.join(out_dir, f"quant_eval_{wave_slug}_shape.png"),
        title=f"Quantitative evaluation — Shape & timing ({wave_slug})",
    )

    _save_table_figure(
        err_tbl,
        os.path.join(out_dir, f"quant_eval_{wave_slug}_errors.pdf"),
        title=f"Quantitative evaluation — Errors ({wave_slug})",
    )
    _save_table_figure(
        shape_tbl,
        os.path.join(out_dir, f"quant_eval_{wave_slug}_shape.pdf"),
        title=f"Quantitative evaluation — Shape & timing ({wave_slug})",
    )

    print(f"[eval] tables written to {out_dir}")
# ---------------------------------------------------------------------


# Config ----------------------------------------------------------------
PARSER = add_common_args(argparse.ArgumentParser(description='Run the UK mobility simulation from processed one-month inputs.'), needs_months=True)
PARSER.add_argument('--wave', choices=['alpha', 'delta'], default='alpha')
PARSER.add_argument('--start', type=str, help='Override wave start date (YYYY-MM-DD)')
PARSER.add_argument('--end', type=str, help='Override wave end date (YYYY-MM-DD)')
PARSER.add_argument('--imports', choices=['local', 'external'], default='external',
                    help="Import model: 'local' (share × local infections) or 'external' (share × external prevalence)")
PARSER.add_argument('--mobility', choices=['auto', 'on', 'off'], default='off',
                    help='Use Google/city mobility multipliers: auto, on, or off.')
PARSER.add_argument('--run-tag', type=str, default='',
                    help='Optional output subfolder name, e.g. alpha_nomob or delta_mob.')
PARSER.add_argument('--build-people', choices=['auto', 'on', 'off'], default='auto',
                    help='Whether to build people_on_builds from filtered_pre_model using Gurobi before simulation.')
PARSER.add_argument('--force-rebuild-people', action='store_true',
                    help='Rebuild people_on_builds even if output files already exist.')
ARGS = PARSER.parse_args()
CFG = load_config(ARGS.config)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.append(SRC_DIR)
try:
    from simulation_generic import ContactEpsimGeneric  # noqa: E402
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        f'Could not import simulation_generic.py from {SRC_DIR}. '
        'Keep the bundled src/ directory intact before running the simulation.'
    ) from exc

CSV = str(resolve_path(CFG, 'real_cases_csv'))
REAL = pd.read_csv(CSV, parse_dates=['date']).set_index('date')
MOBILITY_FILE = resolve_optional_path(CFG, 'mobility_csv')
PEOPLE = str(resolve_path(CFG, 'processed_people_on_builds'))
FILTERED_PRE_MODEL = resolve_optional_path(CFG, 'filtered_pre_model')
REGION_DICTS = resolve_optional_path(CFG, 'region_dicts')
MODEL_OUTPUT_ROOT = resolve_optional_path(CFG, 'model_output_root')
MONTH = get_single_month(getattr(ARGS, 'months', None), CFG)
RESULTS_ROOT = resolve_path(CFG, 'results_root')
if getattr(ARGS, 'run_tag', ''):
    RESULTS_ROOT = RESULTS_ROOT / ARGS.run_tag
PLOTS_DIR = str(ensure_dir(RESULTS_ROOT / 'plots'))
OUTPUT_DIR = str(ensure_dir(RESULTS_ROOT / 'tables'))


def maybe_build_people_inputs(cities: list[str]) -> None:
    if ARGS.build_people == 'off' and not ARGS.force_rebuild_people:
        return
    missing = []
    for city in cities:
        city_file = canonical_city_name(city)
        json_path = os.path.join(PEOPLE, f'{city_file}_{MONTH}.json')
        if ARGS.force_rebuild_people or not os.path.exists(json_path):
            missing.append(city)
    if not missing and not ARGS.force_rebuild_people and ARGS.build_people != 'on':
        return
    if FILTERED_PRE_MODEL is None or REGION_DICTS is None:
        raise RuntimeError(
            'filtered_pre_model and region_dicts must be configured to build people_on_builds automatically.'
        )
    try:
        from people_on_builds_builder import ensure_people_on_builds_inputs
    except Exception as exc:
        raise RuntimeError(
            'Could not import people_on_builds_builder. Keep src/people_on_builds_builder.py in the repository.'
        ) from exc

    ensure_people_on_builds_inputs(
        cities=cities,
        months=[MONTH],
        processed_people_dir=PEOPLE,
        filtered_pre_model_dir=FILTERED_PRE_MODEL,
        region_dicts_dir=REGION_DICTS,
        model_output_dir=MODEL_OUTPUT_ROOT,
        build_mode=ARGS.build_people,
        force=ARGS.force_rebuild_people,
        solver_options=resolved_gurobi_options(CFG),
        verbose=True,
    )


# [Ref: CoMix contact studies & tier impact analyses]
# Alpha-era tiers (unchanged)
TIER_SCHEDULE_ALPHA = {
     "reading":   [("2020-12-02", 3), ("2020-12-20", 4)],
     "cambridge": [("2020-12-02", 2), ("2020-12-26", 4)],
     "coventry":  [("2020-12-02", 2), ("2020-12-31", 4)],
 }

# Simple delta-era easing (Step 3 → Step 4 on 2021-07-19). Same for all cities for now.
TIER_SCHEDULE_DELTA = {
    "reading":   [("2021-05-01", 2), ("2021-07-19", 1)],
    "cambridge": [("2021-05-01", 2), ("2021-07-19", 1)],
    "coventry":  [("2021-05-01", 2), ("2021-07-19", 1)],
}

# Default (overridden in __main__ based on --wave)
TIER_SCHEDULE = TIER_SCHEDULE_ALPHA
CITY_POP_REAL = {"reading": 230_000, "cambridge": 125_000, "coventry": 370_000}

WAVE_START, WAVE_END = "2020-12-01", "2021-03-01"
WAVE_NAME  = "Alpha wave"
WAVE_SLUG  = "alpha"
INC_DAYS   = 4                  # [Ref: Lauer 2020; Hart et al. (UK generation time)]
SMOOTH     = 5                  # rolling window for plotting

SEED = 42
DELTA_VARIANT_MULT = 1.55  # Delta vs Alpha per-contact risk ~1.4–1.7 → pick mid
# Delta replacement (default) and slower takeover for SE/EoE cities (PHE VTBs)
DELTA_SHARE_KNOTS_DEFAULT = [
    ("2021-05-01", 0.02),
    ("2021-06-15", 0.30),
    ("2021-07-01", 0.60),
    ("2021-07-15", 0.95),
]
DELTA_SHARE_KNOTS_BY_CITY = {}

# --- Hold & release a fraction of bridging (non-household) edges -----
# We temporarily “freeze” some weak-tie edges (shops/rest) over the Christmas lull,
# then let them come back from 28 Dec to reflect Boxing-Day retail footfall, returns/returns,
# and early-Jan back-to-work mixing. We keep ONE value across cities because we don’t
# wire in city-level footfall series.
# Coventry likely had more essential retail/logistics footfall than Cambridge (students away),
# but without per-city mobility we don’t differentiate here.
# [Ref: ONS/Google mobility/footfall indicators]
BRIDGE_HOLD_FRAC = {"reading":0.60, "cambridge":0.60, "coventry":0.6}
BRIDGE_RELEASE_START = pd.Timestamp("2020-12-26") 
BRIDGE_RELEASE_DAYS  = 7       # release evenly over ~first 2 weeks of Jan

# --- Import method switch ------------------------------------------------------
# 'local'  : share(t) × local infections (detections / ascertainment)
# 'external': fixed share × external prevalence (detections in other cities)
IMPORT_METHOD = "local"  # default; can be overridden via --imports

# External-prevalence shares (replicates prior Delta ~1%)
IMPORT_EXTERNAL_SHARE_KNOTS = {
    # Small but rising inter-city seeding share as mobility recovers (DfT) and holidays arrive
    "alpha": [("2020-12-01", 0.006)],
    "delta": [
        ("2021-05-01", 0.008),
        ("2021-06-15", 0.010),
        ("2021-07-19", 0.016),  # Step 4
        ("2021-08-01", 0.015),  # August holidays (domestic travel peak)
    ],
}

# Travel volume / mobility index for inter-city mixing (DfT transport use)
TRAVEL_VOL_KNOTS = [
    ("2021-08-01", 1.00),
]

# Import seeding (same method for alpha & delta) 
# One method: imports = share(t) × local infections (de-biased by ascertainment)
# Import-share schedules are small heuristic seeding terms. Their timing is intended
# to be consistent with reopening phases and known travel windows, including the
# December 2020 university travel window and increased summer mobility. The exact
# percentages are not directly estimated here and should be treated as scenario
# assumptions rather than measured import rates.
IMPORT_SHARE_KNOTS_ALPHA = [
    ("2020-12-20", 0.012),
    ("2020-12-26", 0.018),
    ("2021-01-01", 0.028),
    ("2021-01-18", 0.012),
    ("2021-02-15", 0.006),
]
IMPORT_SHARE_KNOTS_DELTA = [
    ("2021-05-01", 0.004),
    ("2021-06-10", 0.007),
    ("2021-07-19", 0.012),
    ("2021-08-16", 0.008),
]

IMPORT_SHARE_SCHEDULE = {
    "alpha": IMPORT_SHARE_KNOTS_ALPHA,
    "delta": IMPORT_SHARE_KNOTS_DELTA,
}

# (optional) simple purpose split if you want to target where imports land
PURPOSE_SPLIT = {"vfr": 0.70, "business": 0.25, "holiday": 0.05}

# ── Common sector multipliers after 5 Jan 2021 lockdown ──
# These are latent sector-specific multipliers applied after the national lockdown
# announced on 5 Jan 2021.
# Evidence base:
# - England lockdown from 5 Jan: work from home where possible; non-essential retail closed;
#   hospitality takeaway/delivery only; schools open mainly to vulnerable children and
#   children of critical workers.
# - DfE attendance data show very low in-person attendance through Jan 2021, supporting
#   strong suppression of school-layer mixing.
CITY_LOCK = {
    "reading":   {"sc":0.45, "ofc":0.85, "shop":0.90, "rest":0.70, "ramp_end":"2021-01-15"},
    "cambridge": {"sc":0.45, "ofc":0.85, "shop":0.90, "rest":0.70, "ramp_end":"2021-01-15"},
    "coventry":  {"sc":0.45, "ofc":0.90, "shop":0.94, "rest":0.70, "ramp_end":"2021-01-18"},    # more on-site adult mixing than the other two cities
}

# ── Ascertainment multipliers (reporting only) ──
# These modify observed detection/reporting, not underlying infections.
# Coventry receives a modest uplift because local documents show an earlier and broader
# community-testing offer than the comparators: widespread LFD rollout from 16 Dec 2020,
# 16,522 community-site tests already recorded by 1 Feb 2021, and later seven sites
# plus a mobile unit / community-collect distribution.
ASC_MULT_KNOTS_ALPHA = {
    "reading":   [("2020-12-01", 1.00)],
    "cambridge": [("2020-12-01", 1.00)],
    "coventry":  [
        ("2020-12-01", 1.00),
        ("2021-01-04", 1.08),  
        ("2021-01-18", 1.12),
        ("2021-02-15", 1.08),
        ("2021-03-01", 1.00),
    ],
}

ASC_MULT_KNOTS_DELTA = {
    "reading":   [("2021-05-01", 1.00)],
    "cambridge": [("2021-05-01", 1.00)],
    "coventry": [("2021-05-01", 1.00)],
}


# ── Adult non-household contact caps (latent daily bridge budget) ──
# These caps represent latent daily limits on activated adult non-household edges,
# rather than direct survey means. Values are aligned to major policy phases and
# informed by CoMix contact evidence and city economic structure.

# Reference shorthand used below:
# [CW42]  CoMix week 42: low adult contacts during Lockdown 3; only a small post-Christmas work-contact rise
# [CW70]  CoMix week 70: no obvious immediate increase in reported contacts after 19 July 2021
# [CW76]  CoMix week 76 / workplace analysis: workplace attenders had substantially more contacts than WFH adults
# [CFC]   Centre for Cities: Coventry more on-site / manufacturing-oriented than Reading and Cambridge
# [ONS]   ONS: Cambridge shows stronger term-time seasonality, so extra summer effects are better handled elsewhere
COMMON_NH_PERIODS = [
    ("2020-12-10", "2020-12-27"),  # pre-Christmas restrictions
    ("2020-12-26", "2021-01-05"),  # Christmas / pre-full-lockdown rebound
    ("2021-01-06", "2021-01-24"),  # early Lockdown 3
    ("2021-01-25", "2021-02-07"),  # mid Lockdown 3
    ("2021-02-08", "2099-01-01"),  # late Lockdown 3
]

NH_CAP_VALUES_ALPHA = {
    # CoMix supports a temporary Christmas / early-January increase before lower lockdown mixing sets in [CW42].
    "reading":   [4, 5, 3, 3, 2],

    # Cambridge is kept slightly lower after Christmas, consistent with a more WFH-heavy structure
    # and stronger student-related seasonality [CFC, ONS].
    "cambridge": [5, 4, 3, 2, 2],

    # Coventry is assigned a modest positive offset, consistent with a more on-site /
    # manufacturing-oriented employment mix [CFC], but values are reduced once Lockdown 3 begins [CW42].
    "coventry":  [5, 6, 4, 3, 2],
}


DELTA_NH_PERIODS = [
    ("2021-05-17", "2021-07-18"),  # Step 3
    ("2021-07-19", "2021-08-15"),  # Step 4
    ("2021-08-16", "2099-01-01"),  # post-isolation-rule change
]

NH_CAP_VALUES_DELTA = {
    # Step 3 baseline with a modest increase under Step 4 / late summer reopening [CW70, CW76].
    "reading":   [3, 4, 4],

    # Kept aligned with Reading; additional Cambridge-specific summer effects are better represented
    # through student-presence adjustments rather than inside the NH cap itself [ONS].
    "cambridge": [3, 4, 4],

    # Coventry retains a higher cap across Delta, reflecting a more on-site employment structure [CFC].
    # The post-16-Aug increase is consistent with gradual contact growth and higher workplace-contact levels [CW70, CW76].
    "coventry":  [4, 5, 6],
}

# Cambridge-only student weighting. Cambridge has a student population of over 40,000
# in a city of roughly 145,700 residents, so a weight around 0.3 is a reasonable
# approximation for student-driven seasonality in city activity.
STUDENT_CITY_SHARE = {"cambridge": 0.3}

# [Ref: DfE student travel & Jan return guidance]
STUDENT_PRESENCE = {
    "cambridge": [("2020-12-05",0.30), ("2021-01-05",0.18), ("2021-02-28",0.25)]
}


# reporting kernel (center ~3 days)
delay_probs = np.array([0.30, 0.34, 0.22, 0.10, 0.04])
delay_probs /= delay_probs.sum()
LAG = len(delay_probs)

rng  = default_rng(SEED)

# ─── daily random-mix layer (adds edges instead of deleting) ─────────
HH_CAP       = 3          # fewer household contacts slows spread
NH_RAMP_DAYS  = 10
          
# Bootstrap the first few days to match observed curve exactly
BOOT_DAYS = LAG  

# Bootstrap initialization for early-window alignment to observed detections
FORCE_EARLY_MATCH = True
HANDOFF_BOOST = 1

NO_DIP_GUARD   = True
NO_DIP_WINDOW  = LAG          # guard this many days after BOOT_DAYS
NO_DIP_FLOOR   = "day0"       # 'day0' | 'last_boot' | 'mean_boot'

STUDENT_DOMESTIC_SEED      = True
STUDENT_SEED_WINDOW        = (pd.Timestamp("2020-12-12"), pd.Timestamp("2020-12-18"))

# Small heuristic term for student-movement-related introductions around the December
# university travel window. Government guidance created a staggered student travel
# window and mass testing before departure
STUDENT_SEEDS_PER_100K_DAY = 1.0   

# base β’s (per‐contact infection probabilities)
# [Ref: Madewell et al. (household SAR); UK hospitality/retail risk review]
BASE_BETAS = dict(
    household_dict    = {0: 0.60},
    school_dict       = {0: 0.51},
    office_dict       = {0: 0.544},
    shop_dict         = {0: 0.020},
    rest_dict         = {0: 0.025},
    detect_school     = 0.15,
    detect_adult      = 0.45,
    location          = 0.012,
    interhousehold    = 0.08,
)

# WHY:
# Holiday backlog slows detections (fewer swabs, lab holidays), then a catch-up wave in early Jan.
# Shapes approximate national reporting artefacts; we do not ingest local lab turnaround data.
# Good for getting the “kink” timing right; not a statement about local test ops.  
# --- Holiday reporting backlog → slower detection kernel ---------------
HOLIDAY_BACKLOG = (pd.Timestamp("2020-12-25"), pd.Timestamp("2020-12-26"))  # 2 days only
DELAY_DEFAULT = np.array([0.18, 0.30, 0.27, 0.17, 0.08])

# Keep early-day mass non-trivial so detections don't shift too late
DELAY_BACKLOG = np.array([0.12, 0.28, 0.27, 0.20, 0.13])

# Catch-up ends sooner; don’t drag into mid-Jan
DELAY_CATCHUP = np.array([0.30, 0.30, 0.20, 0.12, 0.08])
DELAY_CATCHUP = DELAY_CATCHUP / DELAY_CATCHUP.sum()
DELAY_CATCHUP_THROUGH = pd.Timestamp("2021-01-05")

# Share of exposed contacts who actually quarantine (coverage of trace+adherence).
# Rough, conservative schedule: a bit lower over the holidays.
Q_COVER_KNOTS_ALPHA = [
    ("2020-12-01", 0.55),
    ("2020-12-21", 0.50),
    ("2021-01-04", 0.45),
    ("2021-01-18", 0.50),
    ("2021-02-01", 0.55),
]

# Delta quarantine coverage:
# - broadly steady early summer,
# - lower after 19 Jul easing,
# - further dip from 16 Aug (no legal self-isolation for fully-vaccinated contacts),
# - slight recovery into September (operational adaptation).
Q_COVER_KNOTS_DELTA = [
    ("2021-05-01", 0.55),
    ("2021-07-19", 0.48),
    ("2021-08-16", 0.35),
]

QD_START = pd.Timestamp("2020-12-14")
QD_END   = pd.Timestamp("2021-01-25")  # longer ramp to soften January lift

# --- Delta vaccination coverage and effectiveness ---
# Coverage values are cumulative shares in [0, 1].
VAX1_COV_KNOTS = [
    ("2021-05-01", 0.55),
    ("2021-06-01", 0.60),
    ("2021-07-01", 0.68),
    ("2021-08-01", 0.72),
    ("2021-08-31", 0.75),
]
VAX2_COV_KNOTS = [
    ("2021-05-01", 0.20),
    ("2021-06-01", 0.40),
    ("2021-07-01", 0.50),
    ("2021-08-01", 0.60),
    ("2021-08-31", 0.64),
]

# Sources: UKHSA/PHE COVID-19 vaccine surveillance reports (May–Aug 2021);
# Eyre DW et al., N Engl J Med 2022;386:744–756.
VE1_DELTA = 0.35
VE2_DELTA_KNOTS = [
    ("2021-05-01", 0.65),
    ("2021-06-15", 0.60),
    ("2021-07-15", 0.55),
    ("2021-08-15", 0.48),
]

# General post-second-dose effectiveness lag:
# UK/PHE-UKHSA and ONS Delta-period analyses commonly define two-dose protection
# from 14 days after the second dose, so we apply a universal 14-day lag to the
# two-dose coverage series before converting coverage into effective protection.
VAX2_EFFECT_LAG_DAYS = 14
REDUCE_TRANSMISSIBILITY_IF_INFECTED = 0.46  # ≈ 35% cut in *onward* transmission once infected (Eyre et al., 2022)

# ---- Event pulses: data-backed, city-specific -------------------------
EVENT_PULSES = [
    # Euro 2020 (UKHSA/PHE + media reports): strong, all cities (Jul 7–12 big, 13–18 taper, 19–21 tail).
    # Sources: UKHSA/PHE analyses and reporting — see docs/citations.md
    dict(name="euro2020_big",   cities="ALL", start="2021-07-07", end="2021-07-12",
         mult={"rest":1.18, "shop":1.05}, nh_cap=1.06),
    dict(name="euro2020_taper", cities="ALL", start="2021-07-13", end="2021-07-18",
         mult={"rest":1.08, "shop":1.03}, nh_cap=1.03),
    dict(name="euro2020_tail",  cities="ALL", start="2021-07-19", end="2021-07-21",
         mult={"rest":1.03, "shop":1.01}, nh_cap=1.01),
]

PULSES_ENABLED = False

def apply_event_pulses(date: pd.Timestamp, city: str, prm: dict):
    if not PULSES_ENABLED:
        return prm
    for p in EVENT_PULSES:
        if p["cities"] != "ALL" and city not in p["cities"]:
            continue
        if pd.Timestamp(p["start"]) <= date <= pd.Timestamp(p["end"]):
            # 1) apply baseline
            for k, v in p.get("mult", {}).items():
                if k in prm["contact_mult"]:
                    prm["contact_mult"][k] *= float(v)
            if "nonhousehold_contacts" in prm and "nh_cap" in p:
                prm["nonhousehold_contacts"] = int(round(
                    prm["nonhousehold_contacts"] * float(p["nh_cap"])
                ))
    return prm


# ---------- Google Mobility integration ----------
# Expected file schema (wide, daily):
# date,city,mobility_retail_and_recreation,mobility_grocery_and_pharmacy,
# mobility_parks,mobility_transit_stations,mobility_workplaces,mobility_residential
# 'city' should be one of: reading, cambridge, coventry (lowercase), after your proxy-averaging step.
# Mobility control: "auto" (default), "on", or "off"
USE_MOBILITY = "off"
MOBILITY_BY_CITY = {}

def _mob_lin_mult(pct: float, alpha: float = 0.8, lo: float = 0.5, hi: float = 1.25) -> float:
    """
    Map % change vs baseline (Google) to a contact multiplier.
    Linear elasticity (alpha ~0.8) and clamps to avoid extremes.
    Example: pct = -20 → 1 + 0.8*(-0.20) = 0.84.
    """
    try:
        return float(np.clip(1.0 + alpha * (float(pct) / 100.0), lo, hi))
    except Exception:
        return 1.0

def _normalize_city_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def load_mobility(path: str):
    """
    Load mobility and build per-city daily multipliers:
      - office  → m_ofc
      - retail+grocery → m_shop
      - retail (as leisure) → m_rest
      - transit → travel (used to scale external-imports)
    Accepts two schemas:
      (A) Pre-aggregated per-city CSV: columns include city,date and either
          m_ofc,m_shop,m_rest,travel  (preferred), OR raw Google columns mobility_*.
      (B) Raw Google shape already filtered/averaged to cities: city,date + mobility_* columns.
    """
    global MOBILITY_BY_CITY

    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception as e:
        warnings.warn(f"[mobility] Could not read {path}: {e}")
        MOBILITY_BY_CITY = {}
        return

    # hygiene
    if "city" not in df.columns or "date" not in df.columns:
        warnings.warn(f"[mobility] Expected 'city' and 'date' columns in {path}.")
        MOBILITY_BY_CITY = {}
        return

    df["city"] = _normalize_city_series(df["city"])
    df = df[df["city"].isin(["reading", "cambridge", "coventry"])].copy()
    if df.empty:
        warnings.warn("[mobility] No rows for reading/cambridge/coventry after city filter.")
        MOBILITY_BY_CITY = {}
        return

    # If ready-made multipliers exist, just take them
    ready_cols = {"m_ofc", "m_shop", "m_rest", "travel"}
    if ready_cols.issubset(set(df.columns)):
        # ensure numeric + 7d smooth
        def _prep(g: pd.DataFrame) -> pd.DataFrame:
            g = g.set_index("date").sort_index()
            g = g[["m_ofc", "m_shop", "m_rest", "travel"]].astype(float)
            g = g.rolling(7, min_periods=1).mean()
            return g

        MOBILITY_BY_CITY = {c: _prep(g) for c, g in df.groupby("city", sort=False)}
        print(f"[mobility] Loaded precomputed multipliers from {path} "
              f"for cities: {', '.join(MOBILITY_BY_CITY.keys())}.")
        return

    # Otherwise, compute multipliers from Google columns if present
    need_cols = {
        "mobility_retail_and_recreation",
        "mobility_grocery_and_pharmacy",
        "mobility_transit_stations",
        "mobility_workplaces",
    }
    if not need_cols.issubset(set(df.columns)):
        missing = sorted(need_cols - set(df.columns))
        warnings.warn(f"[mobility] Missing columns in {path}: {missing}. No mobility will be applied.")
        MOBILITY_BY_CITY = {}
        return

    # 7-day smooth, per city
    num_cols = [
        c for c in [
            "mobility_retail_and_recreation",
            "mobility_grocery_and_pharmacy",
            "mobility_transit_stations",
            "mobility_workplaces",
            "mobility_residential",
            "mobility_parks",
        ]
        if c in df.columns
    ]

    df = df[["city", "date"] + num_cols].sort_values(["city", "date"]).copy()

    smoothed = []
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("date").copy()
        g[num_cols] = g[num_cols].apply(pd.to_numeric, errors="coerce")
        g[num_cols] = g[num_cols].rolling(7, min_periods=1).mean()
        smoothed.append(g)

    df = pd.concat(smoothed, ignore_index=True)

    out = {}
    for city, g in df.groupby("city", sort=False):
        g = g.set_index("date").sort_index()

        if "mobility_residential" not in g.columns:
            g["mobility_residential"] = 0.0
        if "mobility_parks" not in g.columns:
            g["mobility_parks"] = 0.0

        # Build multipliers with reasonable elasticities (literature ~0.6–1.0)
        # Add a residential dampener: if residential is +12%, knock ~0.12*gamma off shop/rest.
        gamma_res = 0.8  # strength of the "more at home → less going out" effect

        m_ofc  = g["mobility_workplaces" ].apply(lambda p: _mob_lin_mult(p, alpha=0.9))
        m_shop_core = (
            0.8 * g["mobility_retail_and_recreation"].apply(lambda p: _mob_lin_mult(p, alpha=0.8)) +
            0.2 * g["mobility_grocery_and_pharmacy"].apply(lambda p: _mob_lin_mult(p, alpha=0.6))
        )
        m_rest_core = g["mobility_retail_and_recreation"].apply(lambda p: _mob_lin_mult(p, alpha=0.9))
        travel = g["mobility_transit_stations"].apply(lambda p: _mob_lin_mult(p, alpha=1.0, lo=0.5, hi=1.35))

        # residential is "+% at home vs baseline" → damp shop/rest (not ofc: workplaces already uses its own series)
        res_damp = (1.0 - gamma_res * (g["mobility_residential"] / 100.0)).clip(0.6, 1.0)

        # add a parks-based *risk* discount that only applies to leisure settings:
        eta_parks = 0.30  # 30% of parks% treated as safer substitution
        risk_damp  = (1.0 - eta_parks * (g["mobility_parks"] / 100.0)).clip(0.7, 1.0)

        m_shop = (m_shop_core * res_damp * risk_damp).astype(float)
        m_rest = (m_rest_core * res_damp * risk_damp).astype(float)

        out[city] = pd.DataFrame(
            {"m_ofc": m_ofc.astype(float),
             "m_shop": m_shop.astype(float),
             "m_rest": m_rest.astype(float),
             "travel": travel.astype(float)},
            index=g.index
        )


    MOBILITY_BY_CITY = out
    print(f"[mobility] Built multipliers for cities: {', '.join(MOBILITY_BY_CITY.keys())} from {path}.")

def init_mobility():
    """Load mobility depending on USE_MOBILITY."""
    global MOBILITY_BY_CITY
    if USE_MOBILITY == "off":
        MOBILITY_BY_CITY = {}
        print("[mobility] Disabled via flag (--mobility=off).")
        return
    try:
        if MOBILITY_FILE is None:
            print("[mobility] No mobility_csv configured; continuing without mobility.")
            return
        load_mobility(str(MOBILITY_FILE))
        if not MOBILITY_BY_CITY:
            print(f"[mobility] Loaded {str(MOBILITY_FILE)} but no usable rows; continuing without mobility.")
        else:
            print(f"[mobility] Using {str(MOBILITY_FILE)} for mobility multipliers.")
    except Exception as e:
        print(f"[mobility] Failed to load {str(MOBILITY_FILE)}: {e}\nContinuing without mobility.")
        MOBILITY_BY_CITY = {}
        
def _travel_volume_mult(date: pd.Timestamp, city: Optional[str] = None) -> float:
    if USE_MOBILITY != "off" and city and MOBILITY_BY_CITY.get(city) is not None:
        m = MOBILITY_BY_CITY[city]
        if date in m.index:
            return float(m.loc[date, "travel"])
        m2 = m.reindex(m.index.union([date])).sort_index().ffill()
        return float(m2.loc[date, "travel"])
    # fallback to coarse schedule
    return _interp_share(date, TRAVEL_VOL_KNOTS)

def _interp_cov(date, knots):
    return _interp_share(date, knots)  # you already have _interp_share()

def _vax_multipliers(date: pd.Timestamp, city: str):
    cov2   = _interp_cov(date - pd.Timedelta(days=VAX2_EFFECT_LAG_DAYS), VAX2_COV_KNOTS)
    cov1_t = _interp_cov(date, VAX1_COV_KNOTS)
    cov1   = max(0.0, cov1_t - cov2)
    VE2_eff = _interp_share(date, VE2_DELTA_KNOTS)
    sus_mult = max(0.0, 1.0 - (VE1_DELTA * cov1 + VE2_eff * cov2))
    inf_mult = 1.0 - REDUCE_TRANSMISSIBILITY_IF_INFECTED * (cov1 + cov2)
    return sus_mult, inf_mult


def _quarantine_base_days(date: pd.Timestamp) -> int:
    # 14 → 10, but slowly (Dec 14 … Jan 25)
    if date <= QD_START: return 14
    if date >= QD_END:   return 10
    t = (date - QD_START).days / max((QD_END - QD_START).days, 1)
    return int(round(14 - 4*t))

def quarantine_days_for(date: pd.Timestamp) -> int:
    cov = _interp_share(date, Q_COVER_KNOTS)  # 0..1
    base = _quarantine_base_days(date)        # 14..10
    # Only the covered fraction “feels” the shorter duration;
    # the rest effectively remain at 14.
    return int(round((1 - cov) * 14 + cov * base))

def _delta_share(date: pd.Timestamp, city: str) -> float:
    knots = DELTA_SHARE_KNOTS_BY_CITY.get(city, DELTA_SHARE_KNOTS_DEFAULT)
    return _interp_share(date, knots)

def _build_cap_schedule(periods, values_map):
    """Build [(start_date, cap), ...] per city from period list + cap arrays."""
    sched = {}
    for city, caps in values_map.items():
        assert len(caps) == len(periods), "cap list length must match periods"
        seq = [(start, cap) for (start, _end), cap in zip(periods, caps)]
        # guarantee a far-future guard
        if periods[-1][1] != "2099-01-01":
            seq.append(("2099-01-01", caps[-1]))
        sched[city] = seq
    return sched

# Build both; select by wave in __main__
NH_CAP_SCHEDULE_ALPHA = _build_cap_schedule(COMMON_NH_PERIODS, NH_CAP_VALUES_ALPHA)
NH_CAP_SCHEDULE_DELTA = _build_cap_schedule(DELTA_NH_PERIODS, NH_CAP_VALUES_DELTA)
NH_CAP_SCHEDULE = NH_CAP_SCHEDULE_ALPHA   # default; overwritten in __main__

def delay_kernel_for(date: pd.Timestamp, city: str) -> np.ndarray:
    if HOLIDAY_BACKLOG[0] <= date <= HOLIDAY_BACKLOG[1]:
        k = DELAY_BACKLOG
    elif date <= DELAY_CATCHUP_THROUGH:
        k = DELAY_CATCHUP
    else:
        k = DELAY_DEFAULT
    k = k.astype(float)
    k /= k.sum()
    return k

def ascertain(date: pd.Timestamp, city: str) -> float:
    if WAVE_SLUG == "delta":
        # Conservative: higher than winter due to widespread LFD use
        if date <  pd.Timestamp("2021-06-01"): base = 0.45
        elif date < pd.Timestamp("2021-07-19"): base = 0.50
        elif date < pd.Timestamp("2021-08-16"): base = 0.55  # before isolation rule change
        else:                                base = 0.50
        return min(base * _asc_mult(date, city), 0.85)
    # ---- Alpha (unchanged) ----
    if date <= pd.Timestamp("2020-12-23"): base = 0.33
    elif date <= pd.Timestamp("2021-01-02"): base = 0.25
    elif date <= pd.Timestamp("2021-01-12"): base = 0.42
    elif date <= pd.Timestamp("2021-01-31"): base = 0.45
    else: base = 0.40
    return min(base * _asc_mult(date, city), 0.80)
    
def _school_nodes(sim):
    out = []
    for u in sim.G:
        for v in sim._nbr_cache.get(u, ()):
            if "school" in str(sim.G[u][v].get("contact_type","")).lower():
                out.append(u); break
    return list(set(out))

def tier_for(city: str, date: pd.Timestamp) -> int:
    t = 2
    for d, v in TIER_SCHEDULE[city]:
        if date >= pd.Timestamp(d): t = v
        else: break
    return t

def _ramp(start, end, date, a, b):
    if date <= start: return a
    if date >= end:   return b
    t = (date - start).days / max((end - start).days, 1)
    return a + t*(b - a)

def _piecewise_cap(date, sched):
    cap = 2
    for d, v in sched:
        if date < pd.Timestamp(d): break
        cap = v
    return cap

def _asc_mult(date, city):
    pts = [(pd.Timestamp(d), f) for d, f in ASC_MULT_KNOTS[city]]; pts.sort()
    if date <= pts[0][0]: return pts[0][1]
    for i in range(1, len(pts)):
        if date < pts[i][0]:
            (d0,f0),(d1,f1)=pts[i-1],pts[i]
            t=(date-d0).days/max((d1-d0).days,1)
            return f0 + t*(f1-f0)
    return pts[-1][1]

def _presence(city, date):
    pts = [(pd.Timestamp(d), v) for d, v in STUDENT_PRESENCE.get(city, [])]
    if not pts: return 1.0
    pts.sort()
    if date <= pts[0][0]: return pts[0][1]
    for i in range(1, len(pts)):
        if date < pts[i][0]:
            (d0,v0),(d1,v1)=pts[i-1],pts[i]
            t=(date-d0).days/max((d1-d0).days,1)
            return v0 + t*(v1-v0)
    return pts[-1][1]

def bridge_open_frac(date: pd.Timestamp) -> float:
    if WAVE_SLUG == "alpha":
        # Dec tighter, early-Jan reopening pulse
        if date <= pd.Timestamp("2020-12-20"): return 0.85
        if date <= pd.Timestamp("2020-12-27"): return 0.70
        if date <= pd.Timestamp("2021-01-05"): return 0.95
        if date <= pd.Timestamp("2021-01-15"): return 1.00
        return 1.00
    else:
        return 1.00
        
def apply_calendar_multipliers(date: pd.Timestamp, prm: dict, city: str) -> dict:
    # - Schools: taper → holiday low → city lock targets after 5 Jan. Reflects national policy.
    # - Adult venues (office/shop/rest): city targets approximate differing WFH/retail mixes:
    #     Cambridge lower (WFH + fewer students on site), Coventry looser and longer (industry open),
    #     Reading in between (commuter/retail). These are priors used to match shape,
    #     not direct measurements. 
    # [Ref: Commons Library briefing; UKHSA festive guidance]
    m = prm["contact_mult"]
    
    # Schools: pre-holiday taper, holidays low, then Jan ramp to city target
    if WAVE_SLUG == "alpha":
        # Schools: pre-Xmas taper → holidays → Jan ramp to city target
        PRE0, PRE1 = pd.Timestamp("2020-12-07"), pd.Timestamp("2020-12-18")
        HOL0, HOL1 = pd.Timestamp("2020-12-19"), pd.Timestamp("2021-01-03")
        L0         = pd.Timestamp("2021-01-05")
        L1         = pd.Timestamp(CITY_LOCK[city]["ramp_end"])
        if PRE0 <= date <= PRE1:
            sc_mult = _ramp(PRE0, PRE1, date, 1.00, 0.70)
        elif HOL0 <= date <= HOL1:
            sc_mult = 0.30
        elif L0 <= date <= L1:
            sc_mult = _ramp(L0, L1, date, 1.00, CITY_LOCK[city]["sc"])
        elif date > L1:
            sc_mult = CITY_LOCK[city]["sc"]
        else:
            sc_mult = 1.00
        m["sc"] *= sc_mult

    # Xmas bubble — do not touch schools
    if pd.Timestamp("2020-12-24") <= date <= pd.Timestamp("2020-12-26"):
        m["hh"] *= 1.05; m["ofc"] *= 0.96; m["shop"] *= 0.92; m["rest"] *= 0.85

    # Offices/shops/rest:
    if WAVE_SLUG == "alpha":
        def piece(k):
            L0 = pd.Timestamp("2021-01-05")
            L1 = pd.Timestamp(CITY_LOCK[city]["ramp_end"])
            tgt = CITY_LOCK[city][k]
            if L0 <= date <= L1: return _ramp(L0, L1, date, 1.0, tgt)
            if date > L1:        return tgt
            return 1.0
        m["ofc"]  *= piece("ofc")
        m["shop"] *= piece("shop")
        m["rest"] *= piece("rest")
    else:
        STEP3, STEP4 = pd.Timestamp("2021-05-17"), pd.Timestamp("2021-07-19")
        MAY_HALF_TERM_START, MAY_HALF_TERM_END = pd.Timestamp("2021-05-31"), pd.Timestamp("2021-06-06")
        SUMMER_START, SUMMER_END = pd.Timestamp("2021-07-20"), pd.Timestamp("2021-08-31")

        # Delta school calendar only; adult sectors are handled by the baseline
        # transmission schedule, tier schedule, NH caps, and optional mobility.
        if MAY_HALF_TERM_START <= date <= MAY_HALF_TERM_END:
            m["sc"] *= 0.25
        elif SUMMER_START <= date <= SUMMER_END:
            m["sc"] *= 0.65
        elif STEP3 <= date < STEP4:
            m["sc"] *= 0.75

    # Student presence attenuation (Cambridge only, Alpha only)
    if WAVE_SLUG == "alpha" and city in STUDENT_CITY_SHARE:
        pres  = _presence(city, date)
        share = STUDENT_CITY_SHARE[city]
        att   = 1.0 - share*(1.0 - pres)
        m["ofc"]  *= att
        m["shop"] *= att
        m["rest"] *= att
    
    # after building m[...] and before the mobility section:
    if WAVE_SLUG == "delta":
        prm = apply_event_pulses(date, city, prm)
    
    # --- Mobility scalers (Google) multiply AFTER policy schedule ---
    if USE_MOBILITY != "off":
        mob = MOBILITY_BY_CITY.get(city)
        if mob is not None:
            if date in mob.index:
                mm = mob.loc[date]
            else:
                mm = mob.reindex(mob.index.union([date])).sort_index().ffill().loc[date]
            m["ofc"]  *= float(mm["m_ofc"])
            m["shop"] *= float(mm["m_shop"])
            m["rest"] *= float(mm["m_rest"])

    return prm

def _scale(v, tf):
    """Scale dict values by tf, leave scalars untouched."""
    return {k: x * tf for k, x in v.items()} if isinstance(v, dict) else v

def force_seed_infectious(sim, k, rng):
    """Flip k susceptibles straight to I so they can infect immediately."""
    if k <= 0:
        return []
    sus = [n for n, d in sim.G.nodes(data=True) if d['state'] == 'S']
    if not sus:
        return []
    k = min(k, len(sus))
    pick = rng.choice(sus, size=k, replace=False)
    for n in pick:
        sim.G.nodes[n].update(state='I', days_in_state=0)  # infectious now
    return list(pick)


def trans_factor(date: pd.Timestamp, city: str) -> float:
    # Residual global transmission scalar.
    # Raw base_tf values are not directly comparable across waves:
    # in Delta, this term is multiplied later by variant uplift and vaccination
    # damping, so a higher base_tf can still imply a similar or lower effective
    # transmission level than in Alpha.

    if WAVE_SLUG == "alpha":
        if date <= pd.Timestamp("2020-12-31"):
            x = (date - pd.Timestamp("2020-12-01")).days / 30
            base_tf = 0.55 + 0.45 * (3 * x**2 - 2 * x**3)
        elif date < pd.Timestamp("2021-01-15"):
            base_tf = 1.00
        elif date < pd.Timestamp("2021-02-05"):
            base_tf = _ramp(pd.Timestamp("2021-01-15"), pd.Timestamp("2021-02-05"), date, 1.00, 0.85)
        else:
            base_tf = 0.75
        holiday_mul = 0.98 if pd.Timestamp("2020-12-24") <= date <= pd.Timestamp("2020-12-26") else 1.0

    elif WAVE_SLUG == "delta":
        # Higher than Alpha in raw value because Delta is additionally modulated
        # by variant_mult_today, sus_mult, and inf_mult downstream.
        if date < pd.Timestamp("2021-06-15"):
            base_tf = 0.95
        elif date < pd.Timestamp("2021-07-19"):
            base_tf = 1.00
        elif date < pd.Timestamp("2021-08-15"):
            base_tf = 1.04
        else:
            base_tf = 1.02
        holiday_mul = 1.0

    tier = tier_for(city, date)
    if WAVE_SLUG == "delta":
        tier_mul = {1: 0.95, 2: 0.85}.get(tier, 1.0)
    else:
        tier_mul = {1: 0.95, 2: 0.85, 3: 0.70}.get(tier, 1.0)

    return base_tf * holiday_mul * tier_mul

def bucket(label: str) -> str:
    l = label.lower()
    if "household" in l or "home" in l:
        return "hh"
    if l.startswith(("office", "employees")):
        return "ofc"
    if "school" in l or "education" in l:
        return "sc"
    if l.startswith("shop_"):
        return "shop"
    if l.startswith("rest_") or "restaurant" in l:
        return "rest"
    return "ot"

def _interp_share(date: pd.Timestamp, knots) -> float:
    pts = [(pd.Timestamp(d), s) for d, s in knots]
    pts.sort()
    if date <= pts[0][0]: return pts[0][1]
    for i in range(1, len(pts)):
        if date < pts[i][0]:
            (d0, s0), (d1, s1) = pts[i-1], pts[i]
            t = (date - d0).days / max((d1 - d0).days, 1)
            return s0 + t * (s1 - s0)
    return pts[-1][1]

def _seed_imports_uniform(sim, k, rng):
    if k <= 0: return
    S = [n for n, d in sim.G.nodes(data=True) if d.get("state") == "S"]
    if not S: return
    k = min(k, len(S))
    pick = rng.choice(S, size=k, replace=False)
    for n in pick:
        sim.G.nodes[n].update(state="E", days_in_state=0, quarantine=False)
        
def imports_k_sim(
    date: pd.Timestamp,
    city: str,
    det_full_today: float,
    ascertain: float,
    pop_scale_factor: float,
    ext_det_today: Optional[float] = None,
    ext_asc_today: Optional[float] = None,  
) -> int:
    if IMPORT_METHOD == "external":
        knots = IMPORT_EXTERNAL_SHARE_KNOTS["delta" if WAVE_SLUG == "delta" else "alpha"]
        share = _interp_share(date, knots)
        travel = _travel_volume_mult(date, city=city)
        asc_ext = float(ext_asc_today) if (ext_asc_today is not None) else float(ascertain)
        ext_inf_full = (ext_det_today or 0.0) / max(asc_ext, 1e-9)
        imp_inf_full = int(round(share * travel * ext_inf_full))
        return int(round(imp_inf_full / max(pop_scale_factor, 1e-9)))
    else:
        knots = IMPORT_SHARE_SCHEDULE["delta" if WAVE_SLUG == "delta" else "alpha"]
        import_share = _interp_share(date, knots)
        inf_full = det_full_today / max(ascertain, 1e-9)
        imp_inf_full = int(round(import_share * inf_full))
        return int(round(imp_inf_full / max(pop_scale_factor, 1e-9)))




def _seed_imports_by_purpose(sim, k, rng, split):
    """Heuristic: VFR→random households (uniform over S),
       business→nodes with an 'office' edge, holiday→rest/shops/community."""
    if k <= 0: return
    S = [n for n, d in sim.G.nodes(data=True) if d.get("state") == "S"]
    if not S: return
    office = []
    for u in S:
        for v in sim._nbr_cache.get(u, ()):
            if str(sim.G[u][v].get("contact_type","")).startswith("office"):
                office.append(u); break
    office = list(set(office))
    comm   = list(set(S) - set(office)) or S

    kv = int(round(k * split.get("vfr", 0.0)))
    kb = int(round(k * split.get("business", 0.0)))
    kh = max(0, k - kv - kb)

    def drop_in(pool, num):
        if num <= 0 or not pool: return
        num = min(num, len(pool))
        for n in rng.choice(pool, size=num, replace=False):
            sim.G.nodes[n].update(state="E", days_in_state=0, quarantine=False)

    drop_in(S, kv)
    drop_in(office, kb)
    drop_in(comm, kh)


# ───────────────────────────────────────────────────────────────────────
# Epsim subclass with immunity waning
# ───────────────────────────────────────────────────────────────────────
class ContactEpsimWaning(ContactEpsimGeneric):
    def simulate_day(self, day, params, disabled, partial):
        pre = {n: d["state"] for n, d in self.G.nodes(data=True)}
        super().simulate_day(day, params, disabled, partial)
        # who became infectious *today*?
        self.new_today = [
            n for n, d in self.G.nodes(data=True)
            if d["state"] == "I" and pre.get(n) != "I"
        ]
        # immunity wanes – recovered become susceptible again after 90 d
        # Surveillance typically uses ≥90 d to define possible reinfection;
        # [Ref: PHE/UKHSA reinfection surveillance; ONS reinfection tech article]
        for _, d in self.G.nodes(data=True):
            if d["state"] == "R" and d["days_in_state"] >= 90:
                d.update(state="S", days_in_state=0)


# ───────────────────────────────────────────────────────────────────────
# Main simulation wrapper
# ───────────────────────────────────────────────────────────────────────

def run(city: str):
    # The public artifact uses exactly one processed month per run.
    print(f"\n{city.title()} context")
    city_file = canonical_city_name(city)
    json_path = os.path.join(PEOPLE, f'{city_file}_{MONTH}.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Missing people_on_builds file: {json_path}')

    # Initialise simulation graph ----------------------------------------------
    sim = ContactEpsimWaning(
        json_path,
        perc_split_classes=25,     
        tier=0,
        tier_start_day=0,
        print_progress=False,
        incubation_period=INC_DAYS,
        infectious_period=14,
        quarantine_duration=14,     # [Ref: UK CMOs statement, 11 Dec 2020]
        mean_household=2.35,
        ramp_days=NH_RAMP_DAYS,
        random_seed=SEED,
    )
    sim.initialize_immunity(0.15, 0.30)

    # Scaling factors -----------------------------------------------------------
    observed = sim.G.number_of_nodes()
    full_pop = CITY_POP_REAL[city]

    print(f"[scale] observed={observed:,}, full_pop={full_pop:,}")

    # --- identify a subset of non-household edges to "hold" until January (Alpha only)
    held_edges = []
    if WAVE_SLUG == "alpha":
        for u, v, d in list(sim.G.edges(data=True)):
            ct = str(d.get("contact_type",""))
            if ct.startswith("household") or "school" in ct:
                continue
            hold_p = BRIDGE_HOLD_FRAC[city]
            if ct.startswith("shop_") or ct.startswith("rest_"):
                if rng.random() < hold_p:
                    held_edges.append((u, v, d))

        if held_edges:
            sim.remove_edges_from_graph([(u, v) for u, v, _ in held_edges])
            print(f"[bridges] holding {len(held_edges):,} non-household edges for Jan release")

        # Bucket them into a simple daily release schedule
        sim._bridge_release = [list() for _ in range(BRIDGE_RELEASE_DAYS)]
        for idx, (u, v, d) in enumerate(held_edges):
            day_bin = idx % BRIDGE_RELEASE_DAYS
            sim._bridge_release[day_bin].append((u, v, d))
    else:
        # No holiday bridge hold/release during delta.
        sim._bridge_release = [list() for _ in range(0)]



    # --- diagnostic: how dense is this city's graph? ----------------------
    deg = np.mean([d for _, d in sim.G.degree()])
    edge_counts = Counter(e.get("contact_type", "unknown") for *_, e in sim.G.edges(data=True))
    print(f"Average degree (all contacts): {deg:.2f}")
    print("Edge counts by contact type:", dict(edge_counts))
    # ----------------------------------------------------------------------


    # After replication, actual simulation pop:
    sim_pop = sim.G.number_of_nodes()
    pop_scale_factor = full_pop / sim_pop  # factor to scale outputs → full pop

    # Transmission parameters base -------
    base = copy.deepcopy(BASE_BETAS)

    # Arrays -------------------------------------------------------------
    dates = pd.date_range(WAVE_START, pd.to_datetime(WAVE_END) - timedelta(days=1), freq="D")
    real  = REAL[city.capitalize()].reindex(dates).fillna(0).values
    # 7-day incidence per 100k for precaution feedback
    inc7d_per100k = (
        REAL[city.capitalize()].rolling(7, min_periods=1).mean()
        .reindex(dates).fillna(0).values / (CITY_POP_REAL[city] / 1e5)
    )

    # external prevalence proxy (only if we use external imports) — compute once
    if IMPORT_METHOD == "external":
        others = [c for c in CITY_POP_REAL if c != city]
        ext_det_series = (
            REAL[[o.capitalize() for o in others]]
            .sum(axis=1).rolling(7, min_periods=1).mean()
            .reindex(dates).fillna(0).values
        )
        ext_asc_series = np.array([
            np.mean([ascertain(dt, oc) for oc in others]) for dt in dates
        ])
    else:
        ext_det_series = None
        ext_asc_series = None


    N     = len(dates)
    pos   = np.zeros(N)
    cols  = ("hh", "sc", "ofc", "shop", "rest", "ot")
    arr   = {c: np.zeros(N, int) for c in cols}
    bins  = np.zeros((LAG, N), int)
    
    real_sim = REAL[city.capitalize()].reindex(dates).fillna(0).values / max(pop_scale_factor, 1e-9)
    boot_det_sim = real_sim[:BOOT_DAYS]

    if NO_DIP_FLOOR == "day0":
        handoff_floor = int(round(boot_det_sim[0]))
    elif NO_DIP_FLOOR == "last_boot":
        handoff_floor = int(round(boot_det_sim[-1]))
    else:  # mean_boot
        handoff_floor = int(round(boot_det_sim.mean()))

    # Bootstrap initialization over the first INC_DAYS
    # Plan cohorts that become I on days 0..INC_DAYS-1 so that detections match.
    boot_newI = np.zeros(INC_DAYS, dtype=int)
    for d in range(INC_DAYS):
        target_det_sim = REAL[city.capitalize()].reindex(dates).fillna(0).values[d] / max(1e-9, pop_scale_factor)
        # contribution arriving today from already-planned cohorts (k>=1)
        carry = 0.0
        for k in range(1, min(LAG, d + 1)):
            carry += boot_newI[d - k] * ascertain(dates[d - k], city) * delay_probs[k]
        # choose today's cohort so that (carry  today's immediate share) hits target
        remain = max(0.0, target_det_sim - carry)
        boot_newI[d] = int(round(remain / (max(1e-9, ascertain(dates[d], city)) * delay_probs[0])))

    # Materialize those cohorts as E with staggered ages so they turn I on day d
    sus = [n for n, nd in sim.G.nodes(data=True) if nd.get("state") == "S"]
    rng.shuffle(sus)
    take_idx = 0
    for d in range(INC_DAYS):
        need = boot_newI[d]
        if need <= 0:
            continue
        need = min(need, len(sus) - take_idx)
        cohort = sus[take_idx:take_idx + need]
        take_idx += need
        # They should flip E->I on day d, so set days_in_state = INC_DAYS-1-d
        age = max(0, INC_DAYS - 1 - d)
        for u in cohort:
            sim.G.nodes[u].update(state="E", days_in_state=age, quarantine=False)
    if take_idx:
        print(f"[bootstrap] preloaded {take_idx:,} E-cases to match first {INC_DAYS} days.")


    # Simulation loop ----------------------------------------------------
    for d, date in enumerate(dates):
        K = delay_kernel_for(date, city)

        # time-varying ascertainment
        ASCERTAIN = ascertain(date, city)

        forced_boot = FORCE_EARLY_MATCH and (d < BOOT_DAYS)

        if forced_boot:
            # Pin this day to the real detections (converted to sim units)
            # Seed enough *infectious* people to match today's real detections
            det_full = real[d]
            det_sim  = int(round(det_full / pop_scale_factor))
            seed_I   = int(round(det_sim / max(ASCERTAIN, 1e-9)))
            if d == BOOT_DAYS - 1:                 # small handoff cushion
                seed_I = int(round(seed_I * HANDOFF_BOOST))
            force_seed_infectious(sim, seed_I, rng)
            
            # schedule detections across delays (no borrowing)
            rem = int(round(seed_I * ASCERTAIN))
            
            for k, p in enumerate(K):
                if rem == 0:
                    break
                take = rng.binomial(rem, p / K[k:].sum())
                rem -= take
                tgt = d + k
                if tgt < N:
                    bins[k, tgt] += take
        else:
            seed_sim = imports_k_sim(
                date, city, real[d], ASCERTAIN, pop_scale_factor,
                ext_det_today=(ext_det_series[d] if ext_det_series is not None else None),
                ext_asc_today=(ext_asc_series[d] if ext_asc_series is not None else None),
            )
            if seed_sim > 0:
                _seed_imports_by_purpose(sim, seed_sim, rng, PURPOSE_SPLIT)

        
        if STUDENT_DOMESTIC_SEED and STUDENT_SEED_WINDOW[0] <= date <= STUDENT_SEED_WINDOW[1]:
            k_full = int(round(STUDENT_SEEDS_PER_100K_DAY * CITY_POP_REAL[city] / 1e5))
            k_sim  = int(round(k_full / max(pop_scale_factor, 1e-9)))
            if k_sim > 0:
                pool = _school_nodes(sim) or [n for n,d in sim.G.nodes(data=True) if d.get("state")=="S"]
                k_sim = min(k_sim, len(pool))
                for n in rng.choice(pool, size=k_sim, replace=False):
                    # seed as Exposed so detections show ~INC_DAYS + lag later
                    sim.G.nodes[n].update(state="E", days_in_state=0, quarantine=False)

        
        # Timed release of held bridges across early January (Alpha only)
        if WAVE_SLUG == "alpha":
            if BRIDGE_RELEASE_START <= date < (BRIDGE_RELEASE_START + pd.Timedelta(days=BRIDGE_RELEASE_DAYS)):
                bin_idx = (date - BRIDGE_RELEASE_START).days
                todays = sim._bridge_release[bin_idx]
                if todays:
                    sim.add_edges_to_graph(todays)

        
        disabled_today = set()
        partial = {}

        tf = trans_factor(date, city)

        # build daily parameter dict
        variant_mult_today = 1.0
        if WAVE_SLUG == "delta":
            variant_mult_today = 1.0 + (DELTA_VARIANT_MULT - 1.0) * _delta_share(date, city)
            sus_mult, inf_mult = _vax_multipliers(date, city)
        else:
            sus_mult, inf_mult = 1.0, 1.0
        vax_tf = tf * variant_mult_today * sus_mult * inf_mult
        prm = {k: _scale(v, vax_tf) for k, v in base.items()}

        nh_today = int(round(_piecewise_cap(date, NH_CAP_SCHEDULE[city]) * bridge_open_frac(date)))

        # baseline
        prm.update(
            detect_school      = base["detect_school"],
            detect_adult       = base["detect_adult"],
            location           = base["location"] * vax_tf,
            interhousehold     = 0.0,      
            avg_visit_times    = {},
            need_minutes       = {},
            contact_mult       = {c: 1.0 for c in cols},
            household_contacts = HH_CAP,
            nonhousehold_contacts = int(nh_today),  
            activity_frac      = 1.0,          
        )
        prm = apply_calendar_multipliers(date, prm, city)
        
        sim.quarantine_duration = quarantine_days_for(date)
        
        # (Optional) also pass in prm for future compatibility with backends that read it there:
        prm["quarantine_duration"] = sim.quarantine_duration
        
        # run the day, then clean up transient edges
        sim.simulate_day(d, prm, disabled=disabled_today, partial=partial)
        sim.prev_I = sim.new_today

        # tally infections by bucket
        for c in cols:
            arr[c][d] = 0

        for n in sim.new_today:
            for neigh in sim.G.neighbors(n):
                if sim.G.nodes[neigh]["state"] == "I":
                    arr[bucket(sim.G[n][neigh]["contact_type"])][d] += 1
                    break

        # reporting delays
        rem = int(round(len(sim.new_today) * ASCERTAIN))   # only detected ones
        for k, p in enumerate(K):
            if rem == 0:
                break
            send = rng.binomial(rem, p / K[k:].sum())
            rem -= send
            tgt = d + k
            if tgt < N:
                bins[k, tgt] += send
        
        if NO_DIP_GUARD and (BOOT_DAYS <= d < BOOT_DAYS + NO_DIP_WINDOW):
            need = handoff_floor - bins[0, d]
            if need > 0:
                # number of infectious seeds needed so that ASCERTAIN * p0 ≈ 'need'
                p0 = K[0]
                add_I = int(np.ceil(need / max(ASCERTAIN * p0, 1e-9)))
                force_seed_infectious(sim, add_I, rng)
                # deterministically allocate EXACTLY 'need' to today's bin,
                # and spread the remainder to future days using the kernel.
                bins[0, d] += need
                rem = int(round(add_I * ASCERTAIN)) - need
                for k in range(1, len(K)):
                    if rem <= 0: break
                    send = int(round(rem * K[k] / K[1:].sum()))
                    tgt = d + k
                    if tgt < N: bins[k, tgt] += send
                    rem -= send
                
        # Exact visual match for first INC_DAYS 
        if forced_boot:
            
            # Show exactly the real detections in the plot for the boot window
            pos[d] = det_sim
        else:
            pos[d] = bins[0, d]


        # advance the queue
        bins[:-1, d:] = bins[1:, d:]
        bins[-1,  d:] = 0

    print(city.title(), "Total pos (scaled):", int((pos * pop_scale_factor).sum()))

    # summary printout --------------------------------------------------
    print("Total infections by category:")
    for c in cols:
        print(f"  {c}: {arr[c].sum()}")
    print(f"Documented pop       : {CITY_POP_REAL[city]:,}")
    print(f"Represented (nodes)  : {observed:,}")
    print(f"Scaled sim pop       : {sim_pop:,}")

    total_real = int(real.sum())
    total_sim  = int((pos * pop_scale_factor).sum())
    print(f"Total pos (real)     : {total_real:,}")
    print(f"Total pos (scaled)   : {total_sim:,}")

    # plot --------------------------------------------------------------
    combined = pos * pop_scale_factor

    # rolling-mean
    s = pd.Series(combined)
    smooth = s.rolling(SMOOTH, min_periods=1, center=False).mean()
    freeze_upto = BOOT_DAYS + (SMOOTH - 1)  # don't average across the hand-off
    smooth.iloc[:freeze_upto] = s.iloc[:freeze_upto]
    
        # --- Quantitative evaluation (matches plotted series) -------------
    obs_s = pd.Series(real, index=dates).rolling(SMOOTH, min_periods=1, center=False).mean()
    obs_s.iloc[:freeze_upto] = real[:freeze_upto]

    row = compute_quant_eval(
        city=city,
        wave=WAVE_SLUG,
        dates=dates,
        sim_plot=smooth.values,
        obs_plot=obs_s.values,
        sim_raw=combined,   # raw daily scaled-to-full-pop
        obs_raw=real,       # raw daily observed
        mape_min_real=5.0,
    )
    METRICS.append(row)

    print(
        f"[eval] MAE={row['mae']:.1f}  RMSE={row['rmse']:.1f}  "
        f"MAPE(>=5)={row['mape_ge5']:.1f}%  r={row['pearson_r']:.3f}  "
        f"peak_shift={row['peak_shift_days']:+d}d  total_err={row['total_pct_err']:+.1f}%"
    )
    # ---------------------------------------------------------------


    plt.figure(figsize=(10, 5))
    plt.plot(dates, smooth, label="Simulation", lw=3.6)
    plt.plot(dates, real, "--", label="Reported positives", lw=3.6)
    plt.title(f"{city.title()} {WAVE_NAME}", fontsize=24)
    plt.ylabel("Positive tests per day", fontsize=20)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.xticks(rotation=0, ha="center", fontsize=12)
    plt.yticks(fontsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{city}_{WAVE_SLUG}_onewave.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = ARGS
    IMPORT_METHOD = args.imports
    
    # Set globals according to wave
    if args.wave == "delta":
        WAVE_SLUG = "delta"
        WAVE_NAME = "Delta wave"
        WAVE_START = args.start or "2021-05-01"
        WAVE_END   = args.end   or "2021-09-01"
        TIER_SCHEDULE = TIER_SCHEDULE_DELTA
        Q_COVER_KNOTS = Q_COVER_KNOTS_DELTA
        NH_CAP_SCHEDULE = NH_CAP_SCHEDULE_DELTA
        ASC_MULT_KNOTS = ASC_MULT_KNOTS_DELTA
    else:
        WAVE_SLUG = "alpha"
        WAVE_NAME = "Alpha wave"
        WAVE_START = args.start or "2020-12-01"
        WAVE_END   = args.end   or "2021-03-01"
        TIER_SCHEDULE = TIER_SCHEDULE_ALPHA
        Q_COVER_KNOTS = Q_COVER_KNOTS_ALPHA
        NH_CAP_SCHEDULE = NH_CAP_SCHEDULE_ALPHA
        ASC_MULT_KNOTS = ASC_MULT_KNOTS_ALPHA
        
    # Apply mobility choice and load if not off
    USE_MOBILITY = args.mobility
    init_mobility()

    requested_cities = [c.lower() for c in get_cities(args.cities, CFG)]
    maybe_build_people_inputs(requested_cities)

    for city in requested_cities:
        run(city)

    save_quant_eval_tables(METRICS, out_root=OUTPUT_DIR, wave_slug=WAVE_SLUG)
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if METRICS:
        dfm = pd.DataFrame(METRICS)
        out_path = os.path.join(out_dir, f"quant_eval_{WAVE_SLUG}.csv")
        dfm.to_csv(out_path, index=False)
        print(f"[eval] wrote {out_path}")
