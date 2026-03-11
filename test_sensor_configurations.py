# -*- coding: utf-8 -*-
"""
Exhaustive Sensor Configuration Tester.

Generates all possible sensor placement combinations for a given
water distribution network and evaluates each one in parallel using
the PSO-based leak localization in ``multi_scenario_eval.py``.

@author: gabri
"""
import os
import sys
import wntr
from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import multi_scenario_eval as eval_script

INP_FILE = "networks/Network_1.inp"
# Use a representative subset of junctions as candidate sensor positions.
# Using all ~130 nodes would produce an intractable number of combinations.
NODES = [
    "JUNCTION-42",
    "JUNCTION-58",
    "JUNCTION-68",
    "JUNCTION-82",
    "JUNCTION-93",
    "JUNCTION-124",
    "RESERVOIR-129",
    "JUNCTION-120",
    "JUNCTION-89",
    "JUNCTION-46",
    "JUNCTION-86",
    "JUNCTION-101",
    "JUNCTION-79",
    "JUNCTION-45",
    "JUNCTION-85"
]


def run_eval_script(input_sensor_combo):
    pid = os.getpid()
    eval_script.main(pid, input_sensor_combo)
    return pid


def get_sensor_combinations(
    possible_sensor_positions, num_sensors_configuration
):
    possible_configurations = combinations(
        possible_sensor_positions, r=num_sensors_configuration
    )
    return possible_configurations


if __name__ == "__main__":
    for NUM_SENSORS in [3, 5]:
        sensor_num_path = os.path.join(
            os.getcwd(),
            "Net1_evaluation",
            "sensor_configurations",
            str(NUM_SENSORS),
        )
        try:
            os.makedirs(sensor_num_path)
        except:
            print(sensor_num_path, " already exists!!")

        sensor_combinations = list(
            get_sensor_combinations(
                NODES,
                num_sensors_configuration=NUM_SENSORS,
            )
        )
        print(
            f"Number of total {NUM_SENSORS}-sensor combinations {len(sensor_combinations)}\n"
        )
        num_processes = os.cpu_count()

        with Pool(processes=num_processes) as pool:
            r = list(
                tqdm(
                    pool.imap(run_eval_script, sensor_combinations),
                    total=len(sensor_combinations),
                    desc=f"Evaluating {NUM_SENSORS}-sensor combinations",
                )
            )
