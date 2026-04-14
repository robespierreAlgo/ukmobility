#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCENARIOS = {
    'alpha_nomob': {'wave': 'alpha', 'mobility': 'off', 'run_tag': 'alpha_nomob'},
    'alpha_mob': {'wave': 'alpha', 'mobility': 'on',  'run_tag': 'alpha_mob'},
    'delta_nomob': {'wave': 'delta', 'mobility': 'off', 'run_tag': 'delta_nomob'},
    'delta_mob': {'wave': 'delta', 'mobility': 'on',  'run_tag': 'delta_mob'},
}


def build_command(
    repo_root: Path,
    python_exe: str,
    config: str,
    scenario: str,
    cities: str | None,
    months: str | None,
    force_rebuild_people: bool,
) -> list[str]:
    spec = SCENARIOS[scenario]
    cmd = [
        python_exe,
        str(repo_root / 'scripts' / '01_run_simulation.py'),
        '--config', config,
        '--wave', spec['wave'],
        '--mobility', spec['mobility'],
        '--build-people', 'auto',
        '--run-tag', spec['run_tag'],
    ]
    if cities:
        cmd.extend(['--cities', cities])
    if months:
        cmd.extend(['--months', months])
    if force_rebuild_people:
        cmd.append('--force-rebuild-people')
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Paper-facing entry point for the four standard scenarios. '
                    'This public runner supports one month per run.'
    )
    parser.add_argument(
        '--scenario',
        choices=['alpha_nomob', 'alpha_mob', 'delta_nomob', 'delta_mob', 'all'],
        default='alpha_nomob',
        help='Scenario to run. Use all to execute the four standard paper scenarios.',
    )
    parser.add_argument(
        '--config',
        default='configs/local_config.json',
        help='Config file path. Defaults to configs/local_config.json.',
    )
    parser.add_argument('--cities', help='Comma-separated city names. Defaults to the config file.')
    parser.add_argument('--months', help='Single month code. Defaults to the config file.')
    parser.add_argument('--force-rebuild-people', action='store_true', help='Rebuild derived simulation inputs even if they already exist.')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    if not config_path.exists():
        raise SystemExit(f'Config file not found: {config_path}')

    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]

    print(f'Using config: {config_path}')
    print('Automatic preprocessing is enabled. Missing intermediate inputs will be built locally when needed.')

    for i, scenario in enumerate(scenarios, start=1):
        print(f'\n[{i}/{len(scenarios)}] Running scenario: {scenario}')
        cmd = build_command(
            repo_root=repo_root,
            python_exe=sys.executable,
            config=str(config_path),
            scenario=scenario,
            cities=args.cities,
            months=args.months,
            force_rebuild_people=args.force_rebuild_people,
        )
        print('Command:', ' '.join(cmd))
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    print('\nDone.')
    print('Results are written to the results_root configured in your config file, in scenario-specific subfolders.')


if __name__ == '__main__':
    main()
