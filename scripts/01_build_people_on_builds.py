#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from _common import (
    add_common_args,
    get_cities,
    get_single_month,
    load_config,
    resolve_optional_path,
    resolve_path,
    resolved_gurobi_options,
)


PARSER = add_common_args(
    argparse.ArgumentParser(description='Build people_on_builds from filtered_pre_model using Gurobi.'),
    needs_months=True,
)
PARSER.add_argument('--force', action='store_true', help='Rebuild outputs even if they already exist.')
ARGS = PARSER.parse_args()
CFG = load_config(ARGS.config)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.append(SRC_DIR)

from people_on_builds_builder import ensure_people_on_builds_inputs  # noqa: E402


def main() -> None:
    filtered_pre_model = resolve_path(CFG, 'filtered_pre_model')
    region_dicts = resolve_path(CFG, 'region_dicts')
    processed_people = resolve_path(CFG, 'processed_people_on_builds')
    model_output_root = resolve_optional_path(CFG, 'model_output_root')

    cities = get_cities(ARGS.cities, CFG)
    month = get_single_month(ARGS.months, CFG)

    built = ensure_people_on_builds_inputs(
        cities=cities,
        months=[month],
        processed_people_dir=processed_people,
        filtered_pre_model_dir=filtered_pre_model,
        region_dicts_dir=region_dicts,
        model_output_dir=model_output_root,
        build_mode='on',
        force=ARGS.force,
        solver_options=resolved_gurobi_options(CFG),
        verbose=True,
    )
    print('\nBuilt files:')
    for path in built:
        print(f'  - {path}')


if __name__ == '__main__':
    main()
