# -*- coding: utf-8 -*-
"""
Leak Simulation Generator for Water Distribution Networks.

Generates hydraulic simulations of leak scenarios at specified junctions
with configurable leak areas and discharge coefficients. Simulations use
WNTR (Water Network Tool for Resilience) and save pressure time-series
as NumPy archives (.npz).

@author: gabri
"""
import os
import sys
import wntr
import itertools
import numpy as np


def modify_baseline_pattern(
    base_daily_pattern,
    sim_duration_seconds,
    monthly_multiplier,
    morning_multiplier=1.3,
    afternoon_multiplier=1.2,
    evening_multiplier=1.5,
    night_multiplier=1.0,
    weekend_multiplier=0.8,
    noise_std=0.0,
):
    baseline_daily_pattern = np.copy(base_daily_pattern)

    # Add daily variation multipliers
    baseline_daily_pattern[0:7] = (
        baseline_daily_pattern[0:7] * night_multiplier
    )
    baseline_daily_pattern[7:14] = (
        baseline_daily_pattern[7:14] * morning_multiplier
    )
    baseline_daily_pattern[14:21] = (
        baseline_daily_pattern[14:21] * afternoon_multiplier
    )
    baseline_daily_pattern[21:] = (
        baseline_daily_pattern[21:] * evening_multiplier
    )

    baseline_monthly_pattern = []
    for day in range(int(sim_duration_seconds / 24 / 3600)):
        if day % 7 < 5:  # Weekday
            baseline_monthly_pattern.extend(baseline_daily_pattern)
        else:  # Weekend 20% less demand
            baseline_monthly_pattern.extend(
                [p * 0.8 for p in baseline_daily_pattern]
            )

    baseline_monthly_pattern = (
        np.array(baseline_monthly_pattern) * monthly_multiplier
    )

    baseline_monthly_pattern = baseline_monthly_pattern + np.random.normal(
        loc=0.0, scale=noise_std, size=len(baseline_monthly_pattern)
    )

    baseline_monthly_pattern[baseline_monthly_pattern < 0] = 0

    return baseline_monthly_pattern


def generate_single_simulation(
    inp_file,
    sim_duration_seconds,
    timestep,
    pattern_timestep,
    monthly_multiplier,  # Depends on month and season
    node_mappings,
    tank_controls,  # dict
    leak_node,
    leak_area,
    leak_discharge_coeff,
    noise_std,
):
    sim_wn = wntr.network.WaterNetworkModel(inp_file)

    # Extract the first tank and pump from the network model
    tank_name = sim_wn.tank_name_list[0]
    pump_name = sim_wn.pump_name_list[0]

    # Setup simulation options
    sim_wn.options.time.duration = sim_duration_seconds
    sim_wn.options.time.hydraulic_timestep = timestep
    sim_wn.options.time.pattern_timestep = pattern_timestep
    sim_wn.options.time.report_timestep = timestep
    sim_wn.options.hydraulic.demand_model = "PDD"

    # For each demand pattern in the inp file, modify it and replace it
    for pattern_name in list(sim_wn.pattern_name_list):
        base_pattern = np.array(
            list(sim_wn.get_pattern(pattern_name).multipliers)
        )
        modified_pattern = modify_baseline_pattern(
            base_pattern,
            sim_duration_seconds,
            monthly_multiplier,
            morning_multiplier=1.0,
            afternoon_multiplier=1.0,
            evening_multiplier=1.0,
            night_multiplier=1.0,
            weekend_multiplier=1.0,
            noise_std=noise_std,
        )
        modified_pat_name = pattern_name + "_modified"
        sim_wn.add_pattern(modified_pat_name, modified_pattern)
        for junc_name in sim_wn.junction_name_list:
            junc = sim_wn.get_node(junc_name)
            updated_demands = []
            for demand in junc.demand_timeseries_list:
                pat = demand.pattern
                if pat is not None and pat.name == pattern_name:
                    updated_demands.append(
                        (demand.base_value, modified_pat_name, demand.category)
                    )
                else:
                    pat_name = pat.name if pat is not None else None
                    updated_demands.append(
                        (demand.base_value, pat_name, demand.category)
                    )
            junc.demand_timeseries_list.clear()
            for base_val, p_name, cat in updated_demands:
                junc.add_demand(base_val, p_name, cat)

    leak_node_obj = sim_wn.get_node(leak_node)
    leak_node_obj.add_leak(
        sim_wn,
        area=leak_area,
        discharge_coeff=leak_discharge_coeff,
        start_time=0,
        end_time=sim_duration_seconds,
    )

    # Run the simulation
    sim = wntr.sim.WNTRSimulator(sim_wn)
    sim_results = sim.run_sim()

    pressure_values = np.empty(
        shape=(
            len(node_mappings.keys()),
            int((sim_duration_seconds + timestep) / timestep),
        )
    )

    for node_id, node_idx in node_mappings.items():
        pressure_values[node_idx, :] = sim_results.node["pressure"][
            node_id
        ].values

    return pressure_values


def generate_multiple_scenarios(
    combinations, simulation_options, save_file_path
):
    scenarios = []
    scenarios_params = []
    for comb in combinations:
        leak_node, leak_area, leak_discharge_coeff = comb
        single_simulation_pressure_values = generate_single_simulation(
            inp_file=simulation_options["inp_file"],
            sim_duration_seconds=simulation_options["simulation_duration"],
            timestep=simulation_options["timestep"],
            pattern_timestep=simulation_options["timestep"],
            monthly_multiplier=simulation_options[
                "season_multiplier"
            ],  # Depends on month and season
            node_mappings=simulation_options["node_name_mappings"],
            tank_controls=simulation_options["tank_controls"],  # dict
            leak_node=leak_node,
            leak_area=leak_area,
            leak_discharge_coeff=leak_discharge_coeff,
            noise_std=simulation_options["noise_level"],
        )
        scenarios.append(single_simulation_pressure_values)
        scenarios_params.append(
            np.array(
                [
                    simulation_options["node_name_mappings"][leak_node],
                    leak_area,
                    leak_discharge_coeff,
                ]
            )
        )
    scenarios = np.array(scenarios)
    scenarios_params = np.array(scenarios_params)
    np.savez(
        file=save_file_path,
        node_name_mapping=simulation_options["node_name_mappings"],
        scenarios_params=scenarios_params,
        scenarios=scenarios,
    )


if __name__ == "__main__":
    INP_FILE = "networks/Network_1.inp"
    SAVE_FILE_PATH = os.path.join(
        os.getcwd(),
        "Network_1_evaluation",
        "scenarios",
        "minimal_scenarios",
    )
    simulation_duration = 24 * 3600
    leak_areas = [0.005]        # m²: small (5 cm²), medium (50 cm²), large (500 cm²)
    leak_discharge_coeffs = [0.75]       # Cd: sharp-edged orifice, rounded orifice
    leak_nodes = [
        "JUNCTION-82",
        "JUNCTION-93",
    ]
    WN = wntr.network.WaterNetworkModel(INP_FILE)

    AVAILABLE_NODE_MAPPINGS = {
        node_id: node_idx for node_idx, node_id in enumerate(WN.node_name_list)
    }

    simulation_parameter_combinations = list(
        itertools.product(leak_nodes, leak_areas, leak_discharge_coeffs)
    )
    print(
        "Different simulation parameter combinations",
        len(simulation_parameter_combinations),
    )
    tank_controls = {
        "low": {
            "initial_level_frac": 1.0,
            "low_level_frac": 0.9,
            "high_level_frac": 0.95,
        }
    }
    num_sim_params = len(simulation_parameter_combinations[0])

    simulation_options = dict(
        inp_file=INP_FILE,
        simulation_duration=4*3600,
        timestep=3600,
        season_multiplier=1.0,
        node_name_mappings=AVAILABLE_NODE_MAPPINGS,
        tank_controls=tank_controls,
        noise_level=0.0,
    )
    generate_multiple_scenarios(
        simulation_parameter_combinations, simulation_options, SAVE_FILE_PATH
    )