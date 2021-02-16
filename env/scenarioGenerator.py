from typing import Dict
import xml.etree.ElementTree as ET
import numpy as np

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_writer import OverwriteExistingFile
from commonroad.scenario.scenario import Tag
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.trajectory import State
from commonroad.common.util import AngleInterval, Interval

from utils.helper import read_config


TEMPLATE = "scenario/Scenario_Template.xml"
CONFIG_PATH = "env/config/scenarioConfig.yaml"
PB_PATH = "env/config/planningProblem.yaml"


class ScenarioGenerator:
    def __init__(self, config_path: str, template_path: str, planning_problem_path: str):
        self.config = read_config(filepath=config_path)
        self.planning_problem_info = read_config(filepath=planning_problem_path)
        self.tree = ET.parse(template_path)
        self.adjacent_lanes: Dict = dict()
        self.info = {"author": 'CM', "affiliation": 'TUD', "source": "", "tags": {Tag.CRITICAL, Tag.INTERSTATE}}

    def create_scenario_xml(self, scenario_name: str) -> None:
        """
        Main function for creating a specified scenario.
        """
        root_node = self.tree.getroot()
        for lane_id in range(self.config["lane_count"]):
            self.create_straight_lanelet(root=root_node, lane_id=lane_id)

        self.tree.write("scenario/" + scenario_name + ".xml")

    def create_straight_lanelet(self, root, lane_id: int) -> None:
        """
        Creates straight lane as specified in config file
        """
        max_lane_count = self.config["lane_count"]
        # Check if lane has neighbor lanes
        if lane_id == 0 and lane_id < max_lane_count - 1:
            adj_left = None
            adj_right = lane_id + 2
        elif lane_id == 0 and lane_id == max_lane_count - 1:
            adj_left = None
            adj_right = None
        elif lane_id == max_lane_count -1:
            adj_left = lane_id
            adj_right = None
        else:
            adj_left = lane_id
            adj_right = lane_id + 2

        self.adjacent_lanes[str(lane_id + 1)] = {"left": adj_left, "right": adj_right}

        lane_element = ET.SubElement(root, "lanelet", {"id": str(lane_id + 1)})
        lane_left_bound = ET.SubElement(lane_element, "leftBound")
        lane_right_bound = ET.SubElement(lane_element, "rightBound")
        x_coord = np.arange(0, self.config["lane_length"])
        lane_width = self.config["lane_width"]
        lower_bound = self.config["lane_offset"] + lane_id * lane_width
        y_coord = np.array([lower_bound, lower_bound + lane_width])

        # Create points which describe one straight line
        for y_c, lane_bound in zip(y_coord, [lane_left_bound, lane_right_bound]):
            for x_c in x_coord:
                lane_point = ET.SubElement(lane_bound, "point")
                lane_point_x = ET.SubElement(lane_point, "x")
                lane_point_x.text = str(x_c)
                lane_point_y = ET.SubElement(lane_point, "y")
                lane_point_y.text = str(y_c)

        if self.adjacent_lanes[str(lane_id+1)]["left"] is not None:
            adjacent_left = ET.SubElement(lane_element, "adjacentLeft",
                                          {"ref": str(self.adjacent_lanes[str(lane_id+1)]["left"]),
                                           "drivingDir": "same"})

        if self.adjacent_lanes[str(lane_id+1)]["right"] is not None:
            adjacent_right = ET.SubElement(lane_element, "adjacentRight",
                                           {"ref": str(self.adjacent_lanes[str(lane_id+1)]["right"]),
                                            "drivingDir": "opposite"})

        lanelet_type = ET.SubElement(lane_element, "laneletType")
        lanelet_type.text = "unknown"

    def add_planning_problem(self, scenario_name: str, replace: bool = False) -> None:
        scenario, pb_set = CommonRoadFileReader("../scenario/" + scenario_name + ".xml").open()

        for pb_id in list(self.planning_problem_info.keys()):
            if int(pb_id) in list(pb_set.planning_problem_dict.keys()) and replace is True:
                pb_set = PlanningProblemSet()
            elif int(pb_id) in list(pb_set.planning_problem_dict.keys()) and replace is False:
                continue

            init_state = State(**{"position": [self.planning_problem_info[pb_id]["start_state"]["x"],
                                               self.planning_problem_info[pb_id]["start_state"]["y"]],
                                  "orientation": 0,
                                  "velocity": 0,
                                  "steering_angle": 0,
                                  "yaw_rate": 0,
                                  "slip_angle": 0,
                                  "time_step": 0})
            goal_state_list = list()
            for target in self.planning_problem_info[pb_id]["goals"]:
                goal_position = Rectangle(length=target["goal_region"]["length"],
                                          width=target["goal_region"]["width"],
                                          center=np.array(target["goal_region"]["center"]))

                goal_state = State(**{"position": goal_position,
                                      "orientation": AngleInterval(start=0, end=np.pi),
                                      "velocity": Interval(start=0, end=10),
                                      "time_step": Interval(start=0, end=10)})
                goal_state_list.append(goal_state)

            pb = PlanningProblem(planning_problem_id=int(pb_id), initial_state=init_state,
                                 goal_region=GoalRegion(state_list=goal_state_list))
            pb_set.add_planning_problem(planning_problem=pb)

        # modify scenario
        fw = CommonRoadFileWriter(scenario, pb_set, self.info["author"],
                                  self.info["affiliation"], self.info["source"], self.info["tags"])

        filename = "scenario/" + scenario_name + ".xml"
        fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)


def create_env(scenario_name: str):
    sceneGen = ScenarioGenerator(config_path=CONFIG_PATH, template_path=TEMPLATE, planning_problem_path=PB_PATH)
    sceneGen.create_scenario_xml(scenario_name=scenario_name)






