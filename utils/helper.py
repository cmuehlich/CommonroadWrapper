"""
Helper functions for convenience.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.scenario.trajectory import State


def read_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.full_load(file)
    return config


def plot_scenario(scenario_name: str) -> None:
    """
    For visualization of defined scenario.
    """
    # read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader("../scenario/" + scenario_name + ".xml").open()
    plt.figure(figsize=(20, 10))
    draw_object(scenario)
    draw_object(planning_problem_set)
    plt.gca().set_aspect('equal')
    plt.margins(0.05, 0.05)
    plt.show()


def get_arr_from_state(state: State) -> np.ndarray:
    x, y = state.position[0], state.position[1]
    delta = state.steering_angle
    velocity = state.velocity
    yaw = state.orientation
    return np.array([x, y, delta, velocity, yaw])


def get_state_from_arr(arr: np.ndarray, time_step: float) -> State:
    return State(**{"position": np.array([arr[0], arr[1]]),
                    "steering_angle": arr[2],
                    "velocity": arr[3],
                    "orientation": arr[4],
                    "time_step": time_step})
