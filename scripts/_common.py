#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in {'.yaml', '.yml'}:
        if yaml is None:
            raise RuntimeError('PyYAML is required for YAML config files. Use JSON or install PyYAML.')
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError('Config root must be a JSON/YAML object.')
    cfg['_config_path'] = str(path)
    cfg['_config_dir'] = str(path.parent)
    cfg.setdefault('paths', {})
    cfg.setdefault('defaults', {})
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    value = cfg.get('paths', {}).get(key)
    if value is None:
        raise KeyError(f'Missing config path: {key}')
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(cfg['_config_dir']) / path).resolve()
    return path


def resolve_optional_path(cfg: dict[str, Any], key: str) -> Path | None:
    value = cfg.get('paths', {}).get(key)
    if value in (None, '', False):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(cfg['_config_dir']) / path).resolve()
    return path


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def get_cities(args_cities: str | None, cfg: dict[str, Any]) -> list[str]:
    if args_cities:
        return split_csv_arg(args_cities)
    cfg_cities = cfg.get('defaults', {}).get('cities')
    if isinstance(cfg_cities, list) and cfg_cities:
        return [str(c) for c in cfg_cities]
    raise ValueError('No cities provided and none found in config defaults.')


def get_months(args_months: str | None, cfg: dict[str, Any]) -> list[str]:
    if args_months:
        return split_csv_arg(args_months)
    cfg_months = cfg.get('defaults', {}).get('months', ['09'])
    return [str(m) for m in cfg_months]


def get_single_month(args_months: str | None, cfg: dict[str, Any]) -> str:
    months = get_months(args_months, cfg)
    if len(months) != 1:
        raise ValueError(
            'This repository currently supports exactly one month per run. '
            f'Got months={months!r}. Set a single month in the config or via --months.'
        )
    return months[0]


def canonical_city_name(city: str) -> str:
    city = city.strip()
    low = city.lower()
    mapping = {
        'reading': 'Reading',
        'cambridge': 'Cambridge',
        'coventry': 'Coventry',
    }
    return mapping.get(low, city[:1].upper() + city[1:])


def add_common_args(parser: argparse.ArgumentParser, *, needs_months: bool = False) -> argparse.ArgumentParser:
    parser.add_argument('--config', required=True, help='Path to JSON/YAML config file.')
    parser.add_argument('--cities', help='Comma-separated city names. Defaults to config.')
    if needs_months:
        parser.add_argument('--months', help='Single month code, e.g. 09. Defaults to config.')
    return parser


def resolved_gurobi_options(cfg: dict[str, Any]) -> dict[str, Any]:
    gurobi_cfg = dict(cfg.get('gurobi', {}) or {})
    license_cfg = dict(gurobi_cfg.get('license', {}) or {})
    options_file = license_cfg.get('options_file')
    if options_file not in (None, '', False):
        path = Path(str(options_file)).expanduser()
        if not path.is_absolute():
            path = (Path(cfg['_config_dir']) / path).resolve()
        license_cfg['options_file'] = str(path)
    gurobi_cfg['license'] = license_cfg
    return gurobi_cfg
