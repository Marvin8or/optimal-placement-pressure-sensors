# -*- coding: utf-8 -*-
"""
PSO-based Leak Localization Evaluation for Water Distribution Networks.

For a given sensor configuration, this module runs a Particle Swarm
Optimisation (PSO) to locate the source of a leak in a water distribution
network. It reads pre-simulated leak scenarios, uses the pressure readings
at sensor nodes as the target, and optimises the leak location parameters
(x, y coordinates, Gaussian spread std, and discharge coefficient height h)
to minimise the MSE between simulated and measured pressures.

@author: gabri
"""

import os
import copy
import json
import logging
import warnings
import wntr
import hashlib
import numpy as np
import pandas as pd

logging.getLogger("wntr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Not all curves were used")

from indago import PSO


def generate_initial_candidates(
    inp_file, possible_leak_junctions, min_std, max_std, min_h, max_h
):
    wn = wntr.network.WaterNetworkModel(inp_file)
    initial_candidates = []
    for junction_id in possible_leak_junctions:
        junction = wn.get_node(junction_id)
        jx, jy = junction.coordinates
        initial_candidates.append(
            [
                jx,
                jy,
                np.mean([min_std, max_std]),
                np.mean([min_h, max_h]),
            ]
        )
    return np.vstack(initial_candidates)


def read_simulated_scenarios(read_file_path):
    simulated_values = np.load(read_file_path, allow_pickle=True)
    return simulated_values


def get_sensor_node_name_mapping(sensor_combination, node_name_mappings):
    sensor_node_name_mapping = {
        snode_id: node_name_mappings[snode_id]
        for snode_id in sensor_combination
    }
    return sensor_node_name_mapping


def get_sensor_node_indices_sorted(sensor_node_name_mapping):
    sensor_node_indices = np.sort(
        np.array(
            [snode_idx for snode_idx in sensor_node_name_mapping.values()]
        )
    )
    return sensor_node_indices


def read_sensor_node_values(available_node_values, sensor_indice_sorted):
    sensor_node_values = available_node_values[sensor_indice_sorted, :]
    return sensor_node_values


def custom_bivariate_gaussian_simplified(x, y, eta_x, eta_y, std, h):
    return h * np.exp(
        -((x - eta_x) ** 2) / (2 * std**2)
        - (y - eta_y) ** 2 / (2 * std**2)
    )


def define_simulation_model():
    """Build the base WN model once and cache it in memory (no disk I/O)."""
    global BASE_WN_MODEL
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    wn.options.time.duration = SIM_DUR
    wn.options.time.hydraulic_timestep = TS
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.time.report_timestep = TS
    wn.options.time.pattern_timestep = TS
    BASE_WN_MODEL = wn


# Pre-compute junction coordinates as a numpy array (avoids repeated lookups)
_JUNC_COORDS = None


def _init_junction_coords():
    global _JUNC_COORDS
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    coords = np.array(
        [wn.get_node(jid).coordinates for jid in POSSIBLE_LEAK_JUNCTIONS]
    )
    _JUNC_COORDS = coords  # shape (n_junctions, 2)


def generate_sample_simulation(eta_x, eta_y, std, h):
    wn = copy.deepcopy(BASE_WN_MODEL)

    # Vectorized nearest-junction lookup
    dists = np.sqrt(
        (_JUNC_COORDS[:, 0] - eta_x) ** 2
        + (_JUNC_COORDS[:, 1] - eta_y) ** 2
    )
    min_idx = np.argmin(dists)

    lj_x, lj_y = _JUNC_COORDS[min_idx]
    disch_coeff = custom_bivariate_gaussian_simplified(
        lj_x, lj_y, eta_x, eta_y, std, h
    )

    leak_junction = wn.get_node(POSSIBLE_LEAK_JUNCTIONS[min_idx])
    leak_junction.add_leak(
        wn,
        area=LEAK_AREA,
        discharge_coeff=disch_coeff,
        start_time=LEAK_START,
        end_time=LEAK_END,
    )

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    # Only extract sensor node pressures (not all nodes)
    sensor_pressures = np.empty(
        (len(SENSOR_INDICES), int(SIM_DUR // TS) + 1)
    )
    sensor_node_ids = SENSOR_INDEX_TO_ID  # pre-built mapping
    for i, node_id in enumerate(sensor_node_ids):
        sensor_pressures[i, :] = results.node["pressure"][node_id].values

    return sensor_pressures


def objective_function_mse(X) -> float:
    eta_x, eta_y, std, h = X[0], X[1], X[2], X[3]

    simulated_pressures = generate_sample_simulation(eta_x, eta_y, std, h)
    measured_pressures = LEAK_MEASURED_SENSOR_VALUES

    diff = simulated_pressures - measured_pressures
    mse = np.mean(diff * diff)
    return mse


def euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def criteria(x1, x2, y1, y2, r=10):
    if euclidean_distance(x1, x2, y1, y2) <= r:
        return True
    else:
        return False


def optimization_information(**kwargs):
    optim_info = {}
    optim_info["Random seed"] = kwargs["random_seed"]
    optim_info["Scenarios file name"] = kwargs["scenario_file_name"]
    optim_info["Sensor combination"] = kwargs["sensor_combination"]
    optim_info["Leak node"] = kwargs["leak_node"]
    optim_info["Leak area"] = kwargs["leak_area"]
    optim_info["Leak disch. coeff."] = kwargs["leak_disch_coeff"]
    optim_info["Leak node X coord."] = kwargs["leak_node_x_coord"]
    optim_info["Leak node Y coord."] = kwargs["leak_node_y_coord"]
    optim_info["Predicted X coord."] = kwargs["optim_results"].X[0]
    optim_info["Predicted Y coord."] = kwargs["optim_results"].X[1]
    optim_info["Predicted h"] = kwargs["optim_results"].X[2]
    optim_info["Predicted std"] = kwargs["optim_results"].X[3]

    optim_info["Achieved fitness"] = kwargs["optim_results"].f
    optim_info["Correctly located leak"] = criteria(
        kwargs["leak_node_x_coord"],
        kwargs["optim_results"].X[0],
        kwargs["leak_node_y_coord"],
        kwargs["optim_results"].X[1],
    )
    optim_info["Optimization elapsed time"] = kwargs["elapsed_time"]

    return optim_info


def combine_dicts2df(list_of_dicts):
    columns = [k for k in list_of_dicts[0].keys()]
    df_dict = {col: [] for col in columns}
    for col in columns:
        for single_dict in list_of_dicts:
            df_dict[col].append(single_dict[col])

    df = pd.DataFrame(df_dict)
    return df


class CVInputHasher:
    def __init__(self, **kwargs):
        params = kwargs
        self._params = params
        self._combo_hash = self._generate_hash()

    def _generate_hash(self):
        params_str = json.dumps(self._params, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()

    def details(self):
        return self._combo_hash, self._params


#### CONSTANTS
SCENARIOS_NPZ_FILE_PATH = os.path.join(
    os.getcwd(),
    "Network_1_evaluation",
    "scenarios",
    "minimal_scenarios.npz",
)

INP_FILE = "networks/Network_1.inp"
TS = 3600
SIM_DUR = 4 * 3600
LEAK_START = 0
LEAK_END = SIM_DUR

MIN_H = 0.01
MAX_H = 0.1
MIN_X = 8000
MIN_Y = 5000
MAX_X = 20000
MAX_Y = 13000
MIN_STD = 100
MAX_STD = 500
LB = [MIN_X, MIN_Y, MIN_STD, MIN_H]
UB = [MAX_X, MAX_Y, MAX_STD, MAX_H]

POSSIBLE_LEAK_JUNCTIONS = [
    "JUNCTION-82",
    "JUNCTION-93",
]


INITIAL_CANDIDATES = generate_initial_candidates(
    INP_FILE, POSSIBLE_LEAK_JUNCTIONS, MIN_STD, MAX_STD, MIN_H, MAX_H
)
RANDOM_SEEDS = [32]
NUM_REPEATS = len(RANDOM_SEEDS)
DEBUG = False


def main(process_id, sensor_config):
    # Globals
    global PROCESS_ID
    global LEAK_AREA
    global LEAK_MEASURED_SENSOR_VALUES
    global SENSOR_INDICES
    global SENSOR_INDEX_TO_ID
    global AVAILABLE_NODE_NAME_MAPPINGS

    PROCESS_ID = process_id
    SENSORS = sensor_config
    define_simulation_model()
    if _JUNC_COORDS is None:
        _init_junction_coords()
    hasher = CVInputHasher(process_id=PROCESS_ID, sensors=SENSORS)
    folder_name, _ = hasher.details()
    sensor_comb_results_folder_path = os.path.join(
        os.getcwd(),
        "Network_1_evaluation",
        "sensor_configurations",
        str(len(SENSORS)),
        folder_name,
    )
    os.makedirs(sensor_comb_results_folder_path, exist_ok=True)

    simulated_scenarios = read_simulated_scenarios(SCENARIOS_NPZ_FILE_PATH)
    # node_name_mapping, scenarios_params, scenarios
    AVAILABLE_NODE_NAME_MAPPINGS = simulated_scenarios[
        "node_name_mapping"
    ].item()

    AVAILABLE_NODE_NAME_MAPPINGS_INV = {
        node_idx: node_id
        for node_id, node_idx in AVAILABLE_NODE_NAME_MAPPINGS.items()
    }
    SCENARIO_PARAMS = simulated_scenarios["scenarios_params"]
    SCENARIOS = simulated_scenarios["scenarios"]

    SENSOR_NODE_NAME_MAPPINGS = get_sensor_node_name_mapping(
        SENSORS, AVAILABLE_NODE_NAME_MAPPINGS
    )

    SENSOR_INDICES = get_sensor_node_indices_sorted(SENSOR_NODE_NAME_MAPPINGS)

    # Pre-build sorted sensor index → node ID mapping for fast pressure extraction
    SENSOR_INDEX_TO_ID = [
        AVAILABLE_NODE_NAME_MAPPINGS_INV[idx] for idx in SENSOR_INDICES
    ]

    assert np.all(
        [
            AVAILABLE_NODE_NAME_MAPPINGS[snode_id]
            == SENSOR_NODE_NAME_MAPPINGS[snode_id]
            for snode_id in SENSOR_NODE_NAME_MAPPINGS.keys()
        ]
    ), "Sensor node mappings don't match with available node mappings"

    num_scenarios = SCENARIOS.shape[0]

    optim_results = []
    for RANDOM_SEED in RANDOM_SEEDS:
        for sidx in range(num_scenarios):
            full_scenario = SCENARIOS[sidx]

            assert (
                full_scenario.ndim == 2
                and full_scenario.shape[0]
                == len(AVAILABLE_NODE_NAME_MAPPINGS.keys())
                and full_scenario.shape[1] == (int(SIM_DUR // TS) + 1)
            ), "Dimensions don't match"

            # Pre-slice sensor pressures for the objective function
            LEAK_MEASURED_SENSOR_VALUES = full_scenario[SENSOR_INDICES, :]

            leak_node_idx, LEAK_AREA, leak_disch_coeff = SCENARIO_PARAMS[sidx]
            leak_node_id = AVAILABLE_NODE_NAME_MAPPINGS_INV[leak_node_idx]
            leak_node_obj = BASE_WN_MODEL.get_node(leak_node_id)

            #### OPTIMIZATION ALGORITHM
            optimizer = PSO()
            # Default params
            # swarm_size 10
            # inertia 0.75
            # cognitive_rate 1
            # social_rate 1
            # max_iterations 50 * 4 ^2 = 800
            # max_evaluations 50 * 4 ^2 = 800

            optimizer.params["swarm_size"] = len(
                POSSIBLE_LEAK_JUNCTIONS
            )  # Default 10
            optimizer.max_iterations = 10
            optimizer.max_evaluations = 5
            optimizer.X0 = np.array(INITIAL_CANDIDATES)

            optimizer.lb = LB
            optimizer.ub = UB
            optimizer.evaluation_function = objective_function_mse
            optimizer.objectives = 1
            result = optimizer.optimize(seed=RANDOM_SEED)

            located = criteria(
                leak_node_obj.coordinates[0],
                result.X[0],
                leak_node_obj.coordinates[1],
                result.X[1],
            )
            print(
                f"[PID {PROCESS_ID}] Seed {RANDOM_SEED} | "
                f"Scenario {sidx + 1}/{num_scenarios} | "
                f"Leak @ {leak_node_id} | "
                f"MSE={result.f:.6f} | "
                f"Located={'YES' if located else 'NO'} | "
                f"Time={optimizer.elapsed_time:.1f}s"
            )

            optimization_result_info = optimization_information(
                random_seed=RANDOM_SEED,
                scenario_file_name=SCENARIOS_NPZ_FILE_PATH,
                sensor_combination=SENSORS,
                leak_node=leak_node_id,
                leak_area=LEAK_AREA,
                leak_disch_coeff=leak_disch_coeff,
                leak_node_x_coord=leak_node_obj.coordinates[0],
                leak_node_y_coord=leak_node_obj.coordinates[1],
                optim_results=result,
                elapsed_time=optimizer.elapsed_time,
            )
            optim_results.append(optimization_result_info)

            if DEBUG:
                print("\n==== Sensor combination results ====")
                for k, v in optimization_result_info.items():
                    if isinstance(v, float):
                        print(f"{k}: {v:.3f}")
                    else:
                        print(f"{k}: {v}")
                print("========\n")

    #### POST-PROCESSING
    optim_results_df = combine_dicts2df(optim_results)
    optim_results_df.to_csv(
        os.path.join(
            sensor_comb_results_folder_path, "combo_individual_results.csv"
        )
    )
    if DEBUG:
        located_leaks = 0.0
        elapsed_time = []
        for opt_res in optim_results:
            located_leaks += opt_res["Correctly located leak"]
            elapsed_time.append(opt_res["Optimization elapsed time"])

        #### FINAL EVALUATION
        print("\n==== Final Sensor Combination Evaluation ====")
        print(
            f"Total correctly located leaks: {int(located_leaks)}|{(NUM_REPEATS*num_scenarios)}"
        )
        print(
            f"Localization rate: {100*(located_leaks/(NUM_REPEATS*num_scenarios)):.3f}%"
        )
        print(f"Total elapsed time: {np.sum(elapsed_time):.3f}")
        print(
            f"Mean elapsed time per scenario evaluation: {np.mean(elapsed_time):.3f}"
        )
