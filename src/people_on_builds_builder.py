from __future__ import annotations

import ast
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CITY_TO_EPS = {"Cambridge": 0.75, "Reading": 0.85, "Coventry": 0.88}

CITY_OVERRIDES = {
    "Reading": {
        "skip_regions": {34},
        "use_special_quotient": True,
    },
    "Coventry": {
        "skip_regions": {63},
        "use_special_quotient": True,
    },
    "Cambridge": {
        "skip_regions": set(),
        "use_special_quotient": True,
    },
}

DEFAULT_SOLVER_OPTIONS: dict[str, Any] = {
    "time_limit_sec": 20,
    "stall_time_sec": 5,
    "stall_rel_improvement": 0.005,
    "stall_abs_improvement": 1.0,
    "no_incumbent_time_sec": 8,
    "mip_gap": 0.03,
    "mip_gap_abs": 1.0,
    "threads": 1,
    "output_flag": 1,
    "mip_focus": 1,
    "heuristics": 0.2,
    "seed": 42,
    "license": {},
}


@dataclass
class StopController:
    stall_time_sec: float
    stall_rel_improvement: float
    stall_abs_improvement: float
    no_incumbent_time_sec: float
    time_limit_sec: float
    best_obj: float | None = None
    last_improve_runtime: float | None = None
    has_incumbent: bool = False
    stop_reason: str = ""

    def register_solution(self, runtime: float, obj: float) -> None:
        if self.best_obj is None:
            self.best_obj = obj
            self.last_improve_runtime = runtime
            self.has_incumbent = True
            return
        abs_improve = self.best_obj - obj
        rel_improve = abs_improve / max(abs(self.best_obj), 1.0)
        if abs_improve >= self.stall_abs_improvement or rel_improve >= self.stall_rel_improvement:
            self.best_obj = obj
            self.last_improve_runtime = runtime
        self.has_incumbent = True

    def should_stop(self, runtime: float) -> bool:
        if runtime >= self.time_limit_sec:
            self.stop_reason = f"time limit reached ({self.time_limit_sec:.1f}s)"
            return True
        if not self.has_incumbent and runtime >= self.no_incumbent_time_sec:
            self.stop_reason = f"no incumbent found after {self.no_incumbent_time_sec:.1f}s"
            return True
        if self.has_incumbent and self.last_improve_runtime is not None:
            if runtime - self.last_improve_runtime >= self.stall_time_sec:
                self.stop_reason = f"no material improvement for {self.stall_time_sec:.1f}s"
                return True
        return False


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _canonical_city_name(city: str) -> str:
    low = city.strip().lower()
    mapping = {
        "reading": "Reading",
        "cambridge": "Cambridge",
        "coventry": "Coventry",
    }
    return mapping.get(low, city[:1].upper() + city[1:])


def _merged_solver_options(solver_options: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_SOLVER_OPTIONS)
    if solver_options:
        out.update({k: v for k, v in solver_options.items() if k != "license"})
        lic = dict(DEFAULT_SOLVER_OPTIONS.get("license", {}))
        lic.update(solver_options.get("license", {}))
        out["license"] = lic
    return out


def _load_wls_options_file(options_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(options_path).expanduser()
    text = path.read_text(encoding="utf-8")
    try:
        node = ast.parse(text, filename=str(path), mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse Gurobi WLS options file: {path}") from exc

    value_node = None
    if len(node.body) == 1:
        stmt = node.body[0]
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "options":
                value_node = stmt.value
        elif isinstance(stmt, ast.Expr):
            value_node = stmt.value
    if value_node is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Expected a Python dict or options = {{...}} in {path}")
        return _normalize_wls_options(ast.literal_eval(text[start:end + 1]))

    return _normalize_wls_options(ast.literal_eval(value_node))


def _normalize_wls_options(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Gurobi WLS options file must evaluate to a dictionary.")
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        normalized[str(key)] = value
    return normalized


def _gurobi_env_params(solver_options: dict[str, Any]) -> dict[str, Any]:
    license_cfg = solver_options.get("license", {}) or {}
    params: dict[str, Any] = {}
    file_params: dict[str, Any] = {}
    options_file = license_cfg.get("options_file")
    if options_file not in (None, "", False):
        file_params = _load_wls_options_file(str(options_file))
    mappings = [
        ("WLSACCESSID", "wls_access_id", "wls_access_id_env"),
        ("WLSSECRET", "wls_secret", "wls_secret_env"),
        ("LICENSEID", "license_id", "license_id_env"),
    ]
    for gp_key, direct_key, env_key in mappings:
        value = license_cfg.get(direct_key)
        if value in (None, ""):
            value = file_params.get(gp_key)
        if value in (None, ""):
            env_name = license_cfg.get(env_key)
            if env_name:
                value = os.environ.get(str(env_name))
        if gp_key == "LICENSEID" and isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if value not in (None, ""):
            params[gp_key] = value
    return params


def _configure_model(model: Any, solver_options: dict[str, Any]) -> None:
    model.Params.OutputFlag = int(solver_options.get("output_flag", 1))
    threads = int(solver_options.get("threads", 1))
    if threads > 0:
        model.Params.Threads = threads
    model.Params.TimeLimit = float(solver_options["time_limit_sec"])
    model.Params.MIPGap = float(solver_options["mip_gap"])
    model.Params.MIPGapAbs = float(solver_options["mip_gap_abs"])
    model.Params.MIPFocus = int(solver_options.get("mip_focus", 1))
    model.Params.Heuristics = float(solver_options.get("heuristics", 0.2))
    model.Params.Seed = int(solver_options.get("seed", 42))


def _make_callback(gp: Any, stop_ctrl: StopController):
    def _callback(model: Any, where: int) -> None:
        if where == gp.GRB.Callback.MIPSOL:
            runtime = float(model.cbGet(gp.GRB.Callback.RUNTIME))
            incumbent = float(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
            stop_ctrl.register_solution(runtime, incumbent)
            if stop_ctrl.should_stop(runtime):
                model.terminate()
        elif where == gp.GRB.Callback.MIP:
            runtime = float(model.cbGet(gp.GRB.Callback.RUNTIME))
            if stop_ctrl.should_stop(runtime):
                model.terminate()
    return _callback


def _get_tot_sqm_and_cap_of_builds(
    data: dict[str, Any],
    region_ind: int,
    build_type: str,
    region_dict: dict[str, int],
) -> tuple[int, int]:
    total_modeled_cap = 0
    total_sqm = 0
    inv_dict = {v: k for k, v in region_dict.items()}
    region_coord = inv_dict[region_ind]
    buildings = data.get(region_coord, {}).get("buildings", {})
    if build_type in buildings:
        for build in buildings[build_type]:
            total_sqm += int(build.get("sqm", 0))
            total_modeled_cap += int(build.get("model_capacity", 0))
    return total_modeled_cap, total_sqm


def _summarize_input_devices(file_data: dict[str, Any], region_keys: list[str]) -> int:
    unique_devices: set[str] = set()
    for region_key in region_keys:
        region_info = file_data.get(region_key, {})
        for dev_list in region_info.get("visiting_devs", {}).values():
            unique_devices.update(dev_list)
        unique_devices.update(region_info.get("staying_devs", []))
    return len(unique_devices)


def _solution_maps(model: Any) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    x_vals: dict[str, float] = {}
    y_vals: dict[str, float] = {}
    z_vals: dict[str, float] = {}
    u_vals: dict[str, float] = {}
    for v in model.getVars():
        name = v.VarName
        val = float(v.X)
        if not name:
            continue
        head = name[0]
        if head == "x":
            x_vals[name] = val
        elif head == "y":
            y_vals[name] = val
        elif head == "z":
            z_vals[name] = val
        elif head == "u":
            u_vals[name] = val
    return x_vals, y_vals, z_vals, u_vals


def _group_by_destination(model_vals: dict[str, float], prefix: str) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, value in model_vals.items():
        if not name.startswith(prefix):
            continue
        dest = name.split("_", 1)[0]
        grouped[dest] = grouped.get(dest, 0.0) + value
    return grouped


def _write_model_summary(
    output_path: Path,
    x_vals: dict[str, float],
    y_vals: dict[str, float],
    z_vals: dict[str, float],
    u_vals: dict[str, float],
) -> None:
    x_grouped = _group_by_destination(x_vals, "x")
    y_grouped = _group_by_destination(y_vals, "y")
    z_grouped = _group_by_destination(z_vals, "z")
    u_grouped = _group_by_destination(u_vals, "u")

    all_prefixes = sorted(
        set(x_grouped.keys()).union(y_grouped.keys()).union(z_grouped.keys()).union(u_grouped.keys()),
        key=lambda x: int(x[1:]),
    )

    total_z_u_global = 0.0
    total_x_y_global = 0.0
    with output_path.open("w", encoding="utf-8") as file:
        for prefix in all_prefixes:
            if not prefix.startswith("x"):
                continue
            x_value = x_grouped.get(prefix, 0.0)
            suffix = prefix[1:]
            y_value = y_grouped.get(f"y{suffix}", 0.0)
            z_value = z_grouped.get(f"z{suffix}", 0.0)
            u_value = u_grouped.get(f"u{suffix}", 0.0)
            file.write(
                f"{prefix:<3}: {x_value:<10.0f} y{suffix}: {y_value:<10.0f} z{suffix}: {z_value:<10.0f} u{suffix}: {u_value:<10.0f}\n"
            )
            if y_value + u_value == 0:
                file.write("WARNING! No employees!\n")
            elif (x_value + z_value) / (y_value + u_value) > 10:
                file.write(f"WARNING! (x+z)/(y+u)= {(x_value + z_value) / (y_value + u_value):.3f}\n")
            if (z_value + u_value) == 0:
                file.write(f"WARNING! Region {suffix}: no non-telefonica!\n")
            elif (x_value + y_value) == 0 or (z_value + u_value) == 0:
                file.write("WARNING! x+y == 0 or z+u == 0\n")
            file.write("\n")
            total_z_u_global += z_value + u_value
            total_x_y_global += x_value + y_value
        file.write(f"nontel: {total_z_u_global:.0f}\n")
        file.write(f"tel: {total_x_y_global:.0f}\n")


def _take_one(pool: list[str], rng: random.Random) -> str | None:
    if not pool:
        return None
    idx = rng.randrange(len(pool))
    return pool.pop(idx)


def _mod_and_save(
    input_path: Path,
    output_path: Path,
    added_vars: dict[str, Any],
    region_dict: dict[str, int],
    model_x_vals: dict[str, float],
    model_y_vals: dict[str, float],
    model_z_vals: dict[str, float],
    model_u_vals: dict[str, float],
    skipped_dest_regions: list[int],
    allocation_seed: int,
) -> None:
    data = _load_json(input_path)
    modified_data: dict[str, Any] = {}
    alloc_rng = random.Random(allocation_seed)

    inv_region_dict = {str(v): k for k, v in region_dict.items()}
    artif_dev_counter = 0
    region_keys = [k for k in sorted(region_dict, key=region_dict.get) if k in data]

    for key in region_keys:
        region_index = region_dict[key]
        if region_index in skipped_dest_regions:
            continue
        new_key = f"({key}, {region_index})"
        value = json.loads(json.dumps(data[key]))
        modified_data[new_key] = value
        modified_data[new_key]["home_devs"] = []
        modified_data[new_key]["leaving_devs"] = {}
        modified_data[new_key]["visiting_devs"] = {}
        modified_data[new_key].pop("staying_devs", None)

    for key in region_keys:
        region_index = region_dict[key]
        if region_index in skipped_dest_regions:
            continue
        new_key = f"({key}, {region_index})"

        this_region_vars: dict[str, float] = {}
        for var_name, var in added_vars.items():
            parts = var_name[1:].split("_", 1)
            if len(parts) != 2:
                continue
            dest_idx, _orig_idx = parts
            if dest_idx == str(region_index):
                this_region_vars[var_name] = float(var.X)

        dev_pool: list[str] = []
        for var_name, val in sorted(this_region_vars.items()):
            if val <= 0:
                continue
            dest_index, origin_index = var_name[1:].split("_")
            origin_region_key = f"({inv_region_dict[origin_index]}, {origin_index})"
            for _ in range(int(round(val))):
                new_dev = f"dev_{artif_dev_counter}"
                artif_dev_counter += 1
                modified_data[new_key].setdefault("visiting_devs", {}).setdefault(inv_region_dict[origin_index], []).append(new_dev)
                dev_pool.append(new_dev)
                if int(origin_index) in skipped_dest_regions:
                    continue
                modified_data[origin_region_key]["home_devs"].append(new_dev)
                modified_data[origin_region_key]["leaving_devs"].setdefault(inv_region_dict[dest_index], []).append(new_dev)

        x_counter = 0
        y_counter = 0
        if not dev_pool:
            print(f"Warning 1: No devices to distribute for region {region_index}")
            continue
        buildings = modified_data[new_key].get("buildings", {})
        if not any(k in buildings for k in ("shop", "restaurant", "education", "office")):
            print(f"Warning 2: No buildings found in region {region_index} for device allocation.")
            continue

        if "education" in buildings:
            education_buildings = buildings["education"]
            total_sqm = sum(int(school.get("sqm", 0)) for school in education_buildings)
            total_pupils = int(round(model_x_vals.get(f"x{region_index}", 0.0) + model_z_vals.get(f"z{region_index}", 0.0)))
            min_pupils_per_school = 8
            total_min_pupils_assigned = 0
            valid_schools = []

            for school in education_buildings:
                school["pupils_devs"] = []
                school["teachers_devs"] = []
                if total_pupils < min_pupils_per_school:
                    print(f"Skipping school with tag {school.get('tag')} due to insufficient pupils in region {region_index}")
                    continue
                num_pupils_to_assign = min_pupils_per_school
                for _ in range(num_pupils_to_assign):
                    dev = _take_one(dev_pool, alloc_rng)
                    if dev is None:
                        break
                    school["pupils_devs"].append(dev)
                    x_counter += 1
                total_min_pupils_assigned += len(school["pupils_devs"])
                valid_schools.append(school)

            if valid_schools:
                remaining_pupils = total_pupils - total_min_pupils_assigned
                if remaining_pupils > 0:
                    for school in valid_schools:
                        proportion = (int(school.get("sqm", 0)) / total_sqm) if total_sqm else 0.0
                        num_pupils = math.floor(proportion * remaining_pupils)
                        for _ in range(num_pupils):
                            if len(school["pupils_devs"]) >= int(school.get("model_capacity", 0)):
                                break
                            dev = _take_one(dev_pool, alloc_rng)
                            if dev is None:
                                break
                            school["pupils_devs"].append(dev)
                            x_counter += 1
                    remaining_pups = int(total_pupils - x_counter)
                    for _ in range(remaining_pups):
                        non_full = [s for s in valid_schools if len(s["pupils_devs"]) < int(s.get("model_capacity", 0))]
                        if not non_full:
                            break
                        random_school = alloc_rng.choice(non_full)
                        dev = _take_one(dev_pool, alloc_rng)
                        if dev is None:
                            break
                        random_school["pupils_devs"].append(dev)
                        x_counter += 1

                for school in valid_schools:
                    teacher_ratio = 8 if school.get("tag") == "kindergarten" else 17
                    num_teachers = math.floor(len(school["pupils_devs"]) / teacher_ratio) + 1
                    for _ in range(num_teachers):
                        dev = _take_one(dev_pool, alloc_rng)
                        if dev is None:
                            break
                        school["teachers_devs"].append(dev)
                        y_counter += 1
            else:
                print(f"No valid schools with enough pupils in region {region_index}.")

            remaining_pups = int(total_pupils - x_counter)
            while remaining_pups > 0 and dev_pool:
                non_full_schools = [
                    school for school in education_buildings
                    if len(school.get("pupils_devs", [])) < int(school.get("model_capacity", 0))
                ]
                if not non_full_schools:
                    break
                chosen_school = alloc_rng.choice(non_full_schools)
                dev = _take_one(dev_pool, alloc_rng)
                if dev is None:
                    break
                chosen_school["pupils_devs"].append(dev)
                x_counter += 1
                remaining_pups -= 1
            if x_counter < total_pupils and dev_pool:
                print(f"Region {region_index}: Not enough school capacity to fit all pupils. Remaining pups: {total_pupils - x_counter}")

        if "office" in buildings:
            offices = buildings["office"]
            total_employees = int(round(model_y_vals.get(f"y{region_index}", 0.0) + model_u_vals.get(f"u{region_index}", 0.0)))
            for office in offices:
                office["employees_devs"] = []
                if int(office.get("model_capacity", 0)) == 0:
                    office["model_capacity"] = 1
                min_allocation = min(math.ceil(0.25 * int(office.get("model_capacity", 1))), total_employees)
                for _ in range(min_allocation):
                    dev = _take_one(dev_pool, alloc_rng)
                    if dev is None:
                        break
                    office["employees_devs"].append(dev)
                    y_counter += 1

            remaining_emps = total_employees - y_counter
            while remaining_emps > 0 and dev_pool:
                non_full_offices = [office for office in offices if len(office.get("employees_devs", [])) < int(office.get("model_capacity", 0))]
                if not non_full_offices:
                    break
                chosen_office = alloc_rng.choice(non_full_offices)
                dev = _take_one(dev_pool, alloc_rng)
                if dev is None:
                    break
                chosen_office["employees_devs"].append(dev)
                y_counter += 1
                remaining_emps -= 1

            if total_employees and y_counter < total_employees:
                buildings_to_distribute: list[dict[str, Any]] = []
                if "shop" in buildings:
                    buildings_to_distribute += buildings["shop"]
                if "restaurant" in buildings:
                    buildings_to_distribute += buildings["restaurant"]
                if not buildings_to_distribute:
                    print(f"No shops or restaurants found in region {region_index}. Remaining employees: {len(dev_pool)}")
                else:
                    total_sqm_remaining = sum(int(b.get("sqm", 0)) for b in buildings_to_distribute)
                    remaining_employees = total_employees - y_counter
                    for building in buildings_to_distribute:
                        dev = _take_one(dev_pool, alloc_rng)
                        if dev is None:
                            break
                        building.setdefault("employees_devs", []).append(dev)
                        y_counter += 1
                    if dev_pool:
                        for building in buildings_to_distribute:
                            cap_div = 10 if building.get("building_type") == "shop" else 15
                            max_employees = int(building.get("sqm", 0)) // cap_div
                            proportion = (int(building.get("sqm", 0)) / total_sqm_remaining) if total_sqm_remaining else 0.0
                            num_employees = math.floor(proportion * remaining_employees)
                            additional = min(num_employees, max(0, max_employees - len(building.get("employees_devs", []))))
                            for _ in range(max(0, additional)):
                                dev = _take_one(dev_pool, alloc_rng)
                                if dev is None:
                                    break
                                building["employees_devs"].append(dev)
                                y_counter += 1
                    while dev_pool:
                        any_alloc = False
                        for building in buildings_to_distribute:
                            cap_div = 10 if building.get("building_type") == "shop" else 15
                            cap = int(building.get("sqm", 0)) // cap_div
                            if len(building.get("employees_devs", [])) < cap:
                                dev = _take_one(dev_pool, alloc_rng)
                                if dev is None:
                                    break
                                building["employees_devs"].append(dev)
                                y_counter += 1
                                any_alloc = True
                                break
                        if not any_alloc:
                            break
                    if y_counter < total_employees and dev_pool:
                        print(f"Region {region_index}: Not enough capacity in shops/restaurants. Remaining employees: {total_employees - y_counter}")
        if dev_pool:
            print(f"Region {region_index}: Unallocated devices: {len(dev_pool)}")

    print(f"Total devices after model: {artif_dev_counter}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=4)


def _build_single_city_month(
    *,
    input_path: Path,
    output_path: Path,
    model_output_path: Path,
    region_dict_path: Path,
    city: str,
    solver_options: dict[str, Any],
    verbose: bool = True,
) -> Path:
    try:
        import gurobipy as gp
    except Exception as exc:
        raise RuntimeError(
            "gurobipy is required to build people_on_builds from filtered_pre_model. "
            "Install Gurobi and ensure your license is available."
        ) from exc

    solver_options = _merged_solver_options(solver_options)
    file_data = _load_json(input_path)
    region_dict = _load_json(region_dict_path)
    region_keys = [k for k in sorted(region_dict, key=region_dict.get) if k in file_data]
    if not region_keys:
        raise ValueError(f"No overlapping region keys between {input_path.name} and {region_dict_path.name}")

    if verbose:
        print(f"\nProcessing {input_path.name}")
        print(region_dict)
        print(f"Total devices before model: {_summarize_input_devices(file_data, region_keys)}")

    env_params = _gurobi_env_params(solver_options)
    env = gp.Env(params=env_params) if env_params else gp.Env()
    with env:
        with gp.Model(env=env) as model:
            _configure_model(model, solver_options)
            added_vars: dict[str, Any] = {}

            def add_variable(name: str):
                if name not in added_vars:
                    var = model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=name)
                    added_vars[name] = var
                return added_vars[name]

            eps = float(CITY_TO_EPS[city])
            overrides = CITY_OVERRIDES.get(city, {})
            origin_regions: set[str] = set()
            skipped_dest_regions: list[int] = []

            for dest_region in region_keys:
                origin_regions.update(file_data.get(dest_region, {}).get("visiting_devs", {}).keys())

            for dest_region in region_keys:
                i = int(region_dict[dest_region])
                if verbose:
                    print(dest_region)
                    print(i)

                total_modeled_cap_office = total_sqm_office = 0
                if "office" in file_data[dest_region].get("buildings", {}):
                    total_modeled_cap_office, total_sqm_office = _get_tot_sqm_and_cap_of_builds(file_data, i, "office", region_dict)

                total_modeled_cap_school = total_sqm_school = 0
                if "education" in file_data[dest_region].get("buildings", {}):
                    total_modeled_cap_school, total_sqm_school = _get_tot_sqm_and_cap_of_builds(file_data, i, "education", region_dict)

                total_modeled_cap = total_modeled_cap_office + total_modeled_cap_school
                total_model_cap_shop = total_sqm_shop = 0
                if "shop" in file_data[dest_region].get("buildings", {}):
                    total_model_cap_shop, total_sqm_shop = _get_tot_sqm_and_cap_of_builds(file_data, i, "shop", region_dict)

                total_model_cap_restaurant = total_sqm_restaurant = 0
                if "restaurant" in file_data[dest_region].get("buildings", {}):
                    total_model_cap_restaurant, total_sqm_restaurant = _get_tot_sqm_and_cap_of_builds(file_data, i, "restaurant", region_dict)

                total_modeled_cap += total_model_cap_shop + total_model_cap_restaurant
                if total_modeled_cap == 0:
                    print(f"Skipping region {i} because of no capacity")
                    skipped_dest_regions.append(i)
                    continue
                if i in overrides.get("skip_regions", set()):
                    print(f"Skipping region {i} due to override")
                    skipped_dest_regions.append(i)
                    continue

                visiting_tel_people = int(file_data[dest_region].get("total_devs_visiting", 0))

                sum_visitor_pup = model.addVar(name=f"sum_v_pup_{i}")
                sum_visitor_emp = model.addVar(name=f"sum_v_emp_{i}")
                sum_nt_vis_pup = model.addVar(name=f"sum_nt_v_pup_{i}")
                sum_nt_vis_emp = model.addVar(name=f"sum_nt_v_emp_{i}")

                visiting_pup: dict[str, Any] = {}
                visiting_emp: dict[str, Any] = {}
                nt_visiting_pup: dict[str, Any] = {}
                nt_visiting_emp: dict[str, Any] = {}

                ratios = file_data[dest_region].get("ratios", {})

                def comp_ratio(key: str, sqm: int, cap: int) -> float:
                    r = ratios.get(key)
                    if r:
                        q = sqm / r
                        if overrides.get("use_special_quotient"):
                            q = cap
                        return q
                    return 0.0

                school_quotient = comp_ratio("school_sqm_per_pupil", total_sqm_school, total_modeled_cap_school)
                office_quotient = comp_ratio("office_sqm_per_employee", total_sqm_office, total_modeled_cap_office)
                rest_quotient = comp_ratio("restaurant_sqm_per_employee", total_sqm_restaurant, total_model_cap_restaurant)
                shop_quotient = comp_ratio("shop_sqm_per_employee", total_sqm_shop, total_model_cap_shop)
                sqm_comb = total_sqm_office + total_sqm_restaurant + total_sqm_shop
                if sqm_comb:
                    combined_emp_quotient = (
                        office_quotient * total_sqm_office + rest_quotient * total_sqm_restaurant + shop_quotient * total_sqm_shop
                    ) / sqm_comb
                else:
                    combined_emp_quotient = 0.0
                if overrides.get("use_special_quotient"):
                    combined_emp_quotient = office_quotient + rest_quotient + shop_quotient
                if verbose:
                    print(
                        f"School Q: {school_quotient}, Office Q: {office_quotient}, Rest Q: {rest_quotient}, Shop Q: {shop_quotient}"
                    )
                    print(f"Combined Emp Q: {combined_emp_quotient}")

                for orig_region in file_data[dest_region].get("visiting_devs", {}):
                    if orig_region not in region_dict:
                        continue
                    j = int(region_dict[orig_region])
                    key = f"{i}_{j}"
                    if key not in visiting_pup:
                        visiting_pup[key] = add_variable(name=f"x{key}")
                        visiting_emp[key] = add_variable(name=f"y{key}")
                        nt_visiting_pup[key] = add_variable(name=f"z{key}")
                        nt_visiting_emp[key] = add_variable(name=f"u{key}")

                sd_count = len(file_data[dest_region].get("staying_devs", []))
                stay_pup = math.floor(sd_count * 0.083)
                total_tel_day = visiting_tel_people + stay_pup

                model.addConstr(sum_visitor_emp == gp.quicksum(visiting_emp.values()))
                model.addConstr(sum_visitor_pup == gp.quicksum(visiting_pup.values()))
                model.addConstr(sum_nt_vis_emp == gp.quicksum(nt_visiting_emp.values()))
                model.addConstr(sum_nt_vis_pup == gp.quicksum(nt_visiting_pup.values()))

                total_visiting_pup = model.addVar(name=f"total_vis_pup_{i}")
                total_visiting_emp = model.addVar(name=f"total_vis_emp_{i}")
                model.addConstr(total_visiting_pup == sum_visitor_pup + sum_nt_vis_pup)
                model.addConstr(total_visiting_emp == sum_visitor_emp + sum_nt_vis_emp)
                model.addConstr(total_modeled_cap * (1 - eps) <= total_visiting_pup + total_visiting_emp)
                model.addConstr(total_visiting_pup + total_visiting_emp <= total_modeled_cap * (1 + eps))
                model.addConstr(total_tel_day * (1 - eps) <= sum_visitor_pup + sum_visitor_emp)
                model.addConstr(sum_visitor_pup + sum_visitor_emp <= total_tel_day * (1 + eps))
                if combined_emp_quotient and school_quotient:
                    ratio = combined_emp_quotient / school_quotient
                    inv = 1 / ratio
                    model.addConstr(inv * total_visiting_emp * (1 - eps) <= total_visiting_pup)
                    model.addConstr(total_visiting_pup <= inv * total_visiting_emp * (1 + eps))

            c_emp_pup = 20 / 49
            print(f"Skipping dest regions: {skipped_dest_regions}")
            for orig_region in sorted(origin_regions, key=lambda r: region_dict.get(r, 10**9)):
                if orig_region not in file_data or orig_region not in region_dict:
                    continue
                residents = int(file_data[orig_region].get("tel_residents", 0)) + int(file_data[orig_region].get("non_tel_residents", 0))
                leave_all = []
                leave_tel = []
                leave_pup = []
                leave_emp = []
                for dest_region in file_data[orig_region].get("leaving_devs", {}):
                    if dest_region not in region_dict:
                        continue
                    di = int(region_dict[dest_region])
                    if di in skipped_dest_regions:
                        continue
                    key = f"{di}_{region_dict[orig_region]}"
                    if f"x{key}" in added_vars:
                        x = added_vars[f"x{key}"]
                        y = added_vars[f"y{key}"]
                        z = added_vars[f"z{key}"]
                        u = added_vars[f"u{key}"]
                        leave_all += [x, y, z, u]
                        leave_tel += [x, y]
                        leave_pup += [x, z]
                        leave_emp += [y, u]
                if leave_all:
                    model.addConstr(residents * (1 - eps) <= gp.quicksum(leave_all))
                    model.addConstr(gp.quicksum(leave_all) <= residents * (1 + eps))
                    model.addConstr(int(file_data[orig_region].get("tel_residents", 0)) * (1 - eps) <= gp.quicksum(leave_tel))
                    model.addConstr(gp.quicksum(leave_tel) <= int(file_data[orig_region].get("tel_residents", 0)) * (1 + eps))
                    model.addConstr(c_emp_pup * (1 - eps) * gp.quicksum(leave_emp) <= gp.quicksum(leave_pup))
                    model.addConstr(gp.quicksum(leave_pup) <= c_emp_pup * (1 + eps) * gp.quicksum(leave_emp))

            c_tn = 41 / 59
            zu = [v for k, v in added_vars.items() if k[0] in ("z", "u")]
            xy = [v for k, v in added_vars.items() if k[0] in ("x", "y")]
            if zu and xy:
                model.addConstr(c_tn * (1 - eps) * gp.quicksum(zu) <= gp.quicksum(xy))
                model.addConstr(gp.quicksum(xy) <= c_tn * (1 + eps) * gp.quicksum(zu))

            model.setObjective(gp.quicksum(v * v for v in added_vars.values()), gp.GRB.MINIMIZE)
            stop_ctrl = StopController(
                stall_time_sec=float(solver_options["stall_time_sec"]),
                stall_rel_improvement=float(solver_options["stall_rel_improvement"]),
                stall_abs_improvement=float(solver_options["stall_abs_improvement"]),
                no_incumbent_time_sec=float(solver_options["no_incumbent_time_sec"]),
                time_limit_sec=float(solver_options["time_limit_sec"]),
            )
            model.optimize(_make_callback(gp, stop_ctrl))
            if stop_ctrl.stop_reason:
                print(f"[gurobi] stopped early: {stop_ctrl.stop_reason}")
            print(f"[gurobi] status={model.Status} runtime={model.Runtime:.2f}s obj={getattr(model, 'ObjVal', float('nan'))}")

            if model.SolCount == 0:
                if model.Status in (gp.GRB.Status.INFEASIBLE, gp.GRB.Status.INF_OR_UNBD):
                    iis_path = model_output_path.with_suffix(".ilp")
                    model.computeIIS()
                    model.write(str(iis_path))
                    raise RuntimeError(f"Model infeasible for {input_path.name}. IIS written to {iis_path}")
                raise RuntimeError(f"Gurobi finished without an incumbent solution for {input_path.name}")

            x_vals, y_vals, z_vals, u_vals = _solution_maps(model)
            _write_model_summary(model_output_path, x_vals, y_vals, z_vals, u_vals)
            _mod_and_save(
                input_path=input_path,
                output_path=output_path,
                added_vars=added_vars,
                region_dict=region_dict,
                model_x_vals=_group_by_destination(x_vals, "x"),
                model_y_vals=_group_by_destination(y_vals, "y"),
                model_z_vals=_group_by_destination(z_vals, "z"),
                model_u_vals=_group_by_destination(u_vals, "u"),
                skipped_dest_regions=skipped_dest_regions,
                allocation_seed=int(solver_options.get("seed", 42)),
            )
            return output_path


def ensure_people_on_builds_inputs(
    *,
    cities: list[str],
    months: list[str],
    processed_people_dir: str | Path,
    filtered_pre_model_dir: str | Path,
    region_dicts_dir: str | Path,
    model_output_dir: str | Path | None = None,
    build_mode: str = "auto",
    force: bool = False,
    solver_options: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[Path]:
    processed_people_dir = Path(processed_people_dir)
    filtered_pre_model_dir = Path(filtered_pre_model_dir)
    region_dicts_dir = Path(region_dicts_dir)
    model_output_dir = Path(model_output_dir) if model_output_dir is not None else processed_people_dir.parent / "model_output"
    processed_people_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if not months:
        raise ValueError("At least one month is required")
    built: list[Path] = []
    for month in months:
        for city_in in cities:
            city = _canonical_city_name(city_in)
            out_path = processed_people_dir / f"{city}_{month}.json"
            need_build = force or build_mode == "on" or (build_mode == "auto" and not out_path.exists())
            if build_mode == "off":
                need_build = False
            if not need_build:
                if not out_path.exists():
                    raise FileNotFoundError(
                        f"Missing people_on_builds file {out_path}. Use build_mode='auto' or 'on', or provide the file directly."
                    )
                built.append(out_path)
                continue

            input_path = filtered_pre_model_dir / f"{city}_{month}.json"
            if not input_path.exists():
                raise FileNotFoundError(f"Missing filtered_pre_model input: {input_path}")
            region_matches = sorted(region_dicts_dir.glob(f"*{city}*.json"))
            if not region_matches:
                raise FileNotFoundError(f"Missing region_dict for {city} under {region_dicts_dir}")
            region_dict_path = region_matches[0]
            model_output_path = model_output_dir / f"{city}_{month}_model_output.txt"
            built.append(
                _build_single_city_month(
                    input_path=input_path,
                    output_path=out_path,
                    model_output_path=model_output_path,
                    region_dict_path=region_dict_path,
                    city=city,
                    solver_options=solver_options or {},
                    verbose=verbose,
                )
            )
    return built
