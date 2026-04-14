#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_common_args, canonical_city_name, get_cities, get_single_month, load_config, resolve_optional_path, resolve_path


def summarize_people_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path} is not a JSON object.')

    n_regions = len(data)
    n_home = 0
    n_visit = 0
    building_counts = {'education': 0, 'office': 0, 'shop': 0, 'restaurant': 0}
    first_key = next(iter(data.keys()), None)

    for info in data.values():
        if not isinstance(info, dict):
            continue
        n_home += len(info.get('home_devs', []))
        n_visit += sum(len(v) for v in info.get('visiting_devs', {}).values())
        buildings = info.get('buildings', {})
        if isinstance(buildings, dict):
            for btype in building_counts:
                block = buildings.get(btype, [])
                if isinstance(block, list):
                    building_counts[btype] += len(block)

    return {
        'regions': n_regions,
        'home_devices': n_home,
        'visiting_assignments': n_visit,
        'building_counts': building_counts,
        'first_region_key': first_key,
    }


def summarize_filtered_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path} is not a JSON object.')

    region_keys = [k for k, v in data.items() if isinstance(v, dict)]
    n_staying = 0
    n_visiting = 0
    n_regions_with_buildings = 0
    for key in region_keys:
        info = data[key]
        n_staying += len(info.get('staying_devs', []))
        n_visiting += sum(len(v) for v in info.get('visiting_devs', {}).values())
        if info.get('buildings'):
            n_regions_with_buildings += 1
    return {
        'regions': len(region_keys),
        'staying_devices': n_staying,
        'visiting_assignments': n_visiting,
        'regions_with_buildings': n_regions_with_buildings,
        'first_region_key': region_keys[0] if region_keys else None,
    }


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(
            description='Check that the required simulation inputs are present and readable. '
                        'This public repo validates one month at a time.'
        ),
        needs_months=True,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    people_root = resolve_path(cfg, 'processed_people_on_builds')
    real_cases = resolve_path(cfg, 'real_cases_csv')
    results_root = resolve_path(cfg, 'results_root')
    mobility_csv = resolve_optional_path(cfg, 'mobility_csv')
    filtered_root = resolve_optional_path(cfg, 'filtered_pre_model')
    region_dicts = resolve_optional_path(cfg, 'region_dicts')
    model_output_root = resolve_optional_path(cfg, 'model_output_root')

    repo_root = Path(__file__).resolve().parents[1]
    sim_file = repo_root / 'src' / 'simulation_generic.py'
    builder_file = repo_root / 'src' / 'people_on_builds_builder.py'

    print('Core repository files:')
    for label, path in [
        ('bundled simulation_generic.py', sim_file),
        ('bundled people_on_builds_builder.py', builder_file),
        ('real_cases_csv', real_cases),
        ('results_root', results_root),
        ('processed_people_on_builds', people_root),
    ]:
        status = 'OK' if path.exists() else 'MISSING'
        print(f'  [{status}] {label}: {path}')

    if filtered_root is None:
        print('  [SKIP] filtered_pre_model: not configured')
    else:
        status = 'OK' if filtered_root.exists() else 'MISSING'
        print(f'  [{status}] filtered_pre_model: {filtered_root}')

    if region_dicts is None:
        print('  [SKIP] region_dicts: not configured')
    else:
        status = 'OK' if region_dicts.exists() else 'MISSING'
        print(f'  [{status}] region_dicts: {region_dicts}')

    if model_output_root is None:
        print('  [SKIP] model_output_root: not configured (will default next to processed outputs)')
    else:
        status = 'OK' if model_output_root.exists() else 'MISSING'
        print(f'  [{status}] model_output_root: {model_output_root}')

    if mobility_csv is None:
        print('  [SKIP] mobility_csv: not configured (fine when running with --mobility off)')
    else:
        status = 'OK' if mobility_csv.exists() else 'MISSING'
        print(f'  [{status}] mobility_csv: {mobility_csv}')

    cities = get_cities(args.cities, cfg)
    month = get_single_month(args.months, cfg)

    print(f'\nChecking derived people_on_builds files for month {month}:')
    for city in cities:
        city_name = canonical_city_name(city)
        path = people_root / f'{city_name}_{month}.json'
        if not path.exists():
            print(f'  [MISSING] {path.name}')
            continue
        stats = summarize_people_file(path)
        print(
            f"  [OK] {path.name}: regions={stats['regions']}, home_devices={stats['home_devices']}, "
            f"visiting_assignments={stats['visiting_assignments']}, first_region={stats['first_region_key']}"
        )
        print(f"       buildings={stats['building_counts']}")

    if filtered_root is not None:
        print(f'\nChecking source filtered_pre_model files for month {month}:')
        for city in cities:
            city_name = canonical_city_name(city)
            path = filtered_root / f'{city_name}_{month}.json'
            if not path.exists():
                print(f'  [MISSING] {path.name}')
                continue
            stats = summarize_filtered_file(path)
            print(
                f"  [OK] {path.name}: regions={stats['regions']}, staying_devices={stats['staying_devices']}, "
                f"visiting_assignments={stats['visiting_assignments']}, regions_with_buildings={stats['regions_with_buildings']}"
            )
            print(f"       first_region={stats['first_region_key']}")


if __name__ == '__main__':
    main()
