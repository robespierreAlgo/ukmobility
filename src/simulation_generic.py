import json
import random
from math import exp
from typing import Iterable

import networkx as nx
from numpy.random import default_rng


def get_day_based_probability(prob_dict, current_day):
    """
    If prob_dict is a dict, return the value for the largest key <= current_day.
    If prob_dict is a float, return it directly.
    """
    if not isinstance(prob_dict, dict):
        return float(prob_dict)
    keys = sorted(prob_dict.keys())
    val = prob_dict[keys[0]]
    for k in keys:
        if current_day >= k:
            val = prob_dict[k]
        else:
            break
    return val


class ContactEpsimGeneric:
    """Simple SEIR network simulator with deterministic RNG plumbing.

    The simulator keeps lightweight neighbour caches because the daily update
    repeatedly needs to sample household versus non-household contacts. Any
    structural change to the graph must therefore update or rebuild the caches.
    """

    def __init__(
        self,
        json_file_path,
        perc_split_classes=0.0,
        tier=3,
        tier_start_day=0,
        print_progress=False,
        incubation_period=7,
        infectious_period=14,
        quarantine_duration=14,
        mean_household=2.86,
        ramp_days=14,
        random_seed=None,
        rng=None,
    ):
        self.perc_split_classes = perc_split_classes
        self.tier = tier
        self.tier_start_day = tier_start_day
        self.print_progress = print_progress
        self.incubation_period = incubation_period
        self.infectious_period = infectious_period
        self.quarantine_duration = quarantine_duration
        self.mean_household = mean_household
        self.ramp_days = ramp_days
        self.random_seed = random_seed
        self.rng = rng if rng is not None else default_rng(random_seed)
        self.py_random = random.Random(random_seed)

        with open(json_file_path, encoding='utf-8') as f:
            data = json.load(f)

        self.G = nx.Graph()
        self._build_network(data)
        self._create_households(data)

        for n in self.G.nodes:
            self.G.nodes[n].update(state='S', days_in_state=0, quarantine=False)

        self.refresh_neighbor_caches()
    
    def _add_clique(self, members, contact_type):
        members = list(dict.fromkeys(members))  # dedupe, preserve order
        for i, u in enumerate(members):
            for v in members[i + 1:]:
                self.G.add_edge(u, v, contact_type=contact_type)

    def _add_school_classes_for_building(self, pupils, teachers, class_size=25):
        pupils = list(pupils)
        teachers = list(teachers)

        if not pupils:
            return

        self.py_random.shuffle(pupils)
        self.py_random.shuffle(teachers)

        classes = [
            pupils[i:i + class_size]
            for i in range(0, len(pupils), class_size)
            if pupils[i:i + class_size]
        ]

        # assign teachers as evenly as possible across classes
        teacher_groups = [[] for _ in classes]
        if teachers and classes:
            for idx, t in enumerate(teachers):
                teacher_groups[idx % len(classes)].append(t)

        # build one clique per class, including its assigned teachers
        for cls, cls_teachers in zip(classes, teacher_groups):
            members = cls + cls_teachers
            self._add_clique(members, contact_type='school')
    
    def refresh_neighbor_caches(self):
        """Rebuild all neighbour caches from the current graph."""
        self._nbr_hh = {}
        self._nbr_nh = {}
        self._nbr_cache = {}
        for u in self.G.nodes:
            hh, nh = [], []
            all_nei = list(self.G.neighbors(u))
            for v in all_nei:
                (hh if self.G[u][v]['contact_type'] == 'household' else nh).append(v)
            self._nbr_hh[u] = hh
            self._nbr_nh[u] = nh
            self._nbr_cache[u] = all_nei

    def add_edges_to_graph(self, edges: Iterable[tuple]):
        """Add edges and keep caches in sync."""
        edges = list(edges)
        if not edges:
            return
        self.G.add_edges_from(edges)
        self.refresh_neighbor_caches()

    def remove_edges_from_graph(self, edges: Iterable[tuple]):
        """Remove edges and keep caches in sync."""
        edges = list(edges)
        if not edges:
            return
        self.G.remove_edges_from(edges)
        self.refresh_neighbor_caches()

    def _build_network(self, data):
        """Create the contact graph from the processed people_on_builds JSON.

        Household edges are added later in ``_create_households``.
        Schools, offices, shops, and restaurants are represented by employee/
        pupil/teacher cliques as encoded in the processed JSON.
        """
        for _region, rd in data.items():
            for dev in rd.get('home_devs', []):
                self.G.add_node(dev, role='resident')

            for cat, blist in rd.get('buildings', {}).items():
                if cat == 'education':
                    for b in blist:
                        pupils = b.get('pupils_devs', [])
                        teachers = b.get('teachers_devs', [])

                        for u in pupils:
                            self.G.add_node(u, role='pupil')
                        for u in teachers:
                            self.G.add_node(u, role='teacher')

                        if self.perc_split_classes > 0:
                            class_size = int(self.perc_split_classes) if self.perc_split_classes >= 1 else 25
                            self._add_school_classes_for_building(
                                pupils,
                                teachers,
                                class_size=class_size,
                            )
                        else:
                            self._add_clique(pupils + teachers, contact_type='school')

                elif cat == 'office':
                    for b in blist:
                        emps = b.get('employees_devs', [])
                        for u in emps:
                            self.G.add_node(u, role='employee')
                        for i, u in enumerate(emps):
                            for v in emps[i + 1:]:
                                self.G.add_edge(u, v, contact_type='office')

                elif cat == 'shop':
                    for b in blist:
                        emps = b.get('employees_devs', [])
                        for u in emps:
                            self.G.add_node(u, role='shop_employee')
                        for i, u in enumerate(emps):
                            for v in emps[i + 1:]:
                                self.G.add_edge(u, v, contact_type='shop_employees')

                elif cat == 'restaurant':
                    for b in blist:
                        emps = b.get('employees_devs', [])
                        for u in emps:
                            self.G.add_node(u, role='rest_employee')
                        for i, u in enumerate(emps):
                            for v in emps[i + 1:]:
                                self.G.add_edge(u, v, contact_type='rest_employees')

                else:
                    for b in blist:
                        for key, devs in b.items():
                            if key.endswith('_devs'):
                                role = key[:-5]
                                for u in devs:
                                    self.G.add_node(u, role=role)
                                for i, u in enumerate(devs):
                                    for v in devs[i + 1:]:
                                        self.G.add_edge(u, v, contact_type=role)

    def _create_households(self, data):
        for _region, rd in data.items():
            devs = rd.get('home_devs', [])[:]
            self.py_random.shuffle(devs)
            i = 0
            while i < len(devs):
                size = max(1, int(self.rng.poisson(self.mean_household)))
                hh = devs[i:i + size]
                i += size
                for u in hh:
                    for v in hh:
                        if u != v:
                            self.G.add_edge(u, v, contact_type='household')

    def _create_school_bubbles(self, _data, bubble_size=25):
        to_rm = [(u, v) for u, v, d in self.G.edges(data=True) if d['contact_type'] == 'school']
        self.G.remove_edges_from(to_rm)
        pupils = [n for n, d in self.G.nodes(data=True) if d.get('role') == 'pupil']
        self.py_random.shuffle(pupils)
        for i in range(0, len(pupils), bubble_size):
            bubble = pupils[i:i + bubble_size]
            for u in bubble:
                for v in bubble:
                    if u != v:
                        self.G.add_edge(u, v, contact_type='school_bubble')

    def initialize_immunity(self, perc_child=0.21, perc_adult=0.36):
        for _n, d in self.G.nodes(data=True):
            if d['state'] == 'S':
                pr = perc_child if d['role'] in ('pupil', 'teacher') else perc_adult
                if self.py_random.random() < pr:
                    d['state'] = 'R'

    def seed_infections(self, num):
        sus = [n for n, d in self.G.nodes(data=True) if d['state'] == 'S']
        self.py_random.shuffle(sus)
        for n in sus[:num]:
            self.G.nodes[n].update(state='E', days_in_state=0)

    def simulate_day(self, day, params, disabled, partial):
        """Budgeted daily contacts with SEIR state updates and quarantine."""
        self.budget_mode = True

        def _bucket(ct):
            if ct == 'household':
                return 'hh'
            if ct in ('school', 'school_bubble'):
                return 'sc'
            if ct == 'office':
                return 'ofc'
            if ct.startswith('shop'):
                return 'shop'
            if ct.startswith('rest'):
                return 'rest'
            return 'ot'

        inf = {'hh': 0, 'sc': 0, 'ofc': 0, 'loc': 0, 'inter': 0}
        ramp = min(day / self.ramp_days, 1.0)
        expose = []

        k_hh = max(0, int(params.get('household_contacts', 2)))
        k_nh = max(0, int(params.get('nonhousehold_contacts', 4)))
        cmult = params.get('contact_mult', {}) or {}
        disabled_set = set(disabled) if disabled else set()

        for u, d in self.G.nodes(data=True):
            if d['state'] != 'S':
                continue

            hh_all = self._nbr_hh.get(u, [])
            nh_all = self._nbr_nh.get(u, [])
            hh_nei = [
                (v, 'household')
                for v in hh_all
                if self.G.has_edge(u, v) and (u, v) not in disabled_set and (v, u) not in disabled_set
            ]
            nh_nei = []
            for v in nh_all:
                if self.G.has_edge(u, v) and (u, v) not in disabled_set and (v, u) not in disabled_set:
                    edata = self.G.get_edge_data(u, v)
                    if edata is not None:
                        nh_nei.append((v, edata.get('contact_type', 'other')))

            if len(hh_nei) > k_hh:
                hh_nei = self.py_random.sample(hh_nei, k_hh)
            if len(nh_nei) > k_nh:
                nh_nei = self.py_random.sample(nh_nei, k_nh)

            lam = 0.0
            for v, ct in hh_nei + nh_nei:
                dv = self.G.nodes[v]
                if dv['state'] == 'I' and not dv.get('quarantine', False):
                    if ct == 'household':
                        pdict = params['household_dict']
                    elif ct in ('school', 'school_bubble'):
                        pdict = params['school_dict']
                    elif ct == 'office':
                        pdict = params['office_dict']
                    else:
                        pdict = params['location']
                    p0 = get_day_based_probability(pdict, day)
                    p0 *= cmult.get(_bucket(ct), 1.0)
                    lam += p0 * ramp

            lam += params.get('interhousehold', 0.0) * ramp
            if lam > 0.0 and self.py_random.random() < 1.0 - exp(-lam):
                expose.append(u)

        for u in expose:
            self.G.nodes[u].update(state='E', days_in_state=0)

        for _n, d in self.G.nodes(data=True):
            if d['state'] in ('E', 'I'):
                d['days_in_state'] = d.get('days_in_state', 0) + 1
                if d['state'] == 'E' and d['days_in_state'] >= self.incubation_period:
                    d.update(state='I', days_in_state=0)
                elif d['state'] == 'I' and d['days_in_state'] >= self.infectious_period:
                    d['state'] = 'R'

        for _n, d in self.G.nodes(data=True):
            if d.get('state') == 'I' and not d.get('quarantine', False):
                p_det = params['detect_school'] if d.get('role') in ('pupil', 'teacher') else params['detect_adult']
                if self.py_random.random() < p_det:
                    d['quarantine'] = True
                    d['days_quarantined'] = 0

        for _n, d in self.G.nodes(data=True):
            if d.get('quarantine', False):
                d['days_quarantined'] = d.get('days_quarantined', 0) + 1
                if d['days_quarantined'] >= self.quarantine_duration:
                    d['quarantine'] = False

        return inf
