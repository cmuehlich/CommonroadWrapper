from typing import Dict, Union, List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils.helper import read_config
from agents.vehicle import Vehicle
from agents.behaviorModels import IDMVehicle

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.trajectory import State
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.visualization.plot_helper import set_non_blocking, redraw_obstacles
matplotlib.use('Qt5Agg')


class DrivingEnv:
    """
    Main Class for storing a time dependent traffic scenario.
    """
    def __init__(self, configpath: str):
        self.config = read_config(filepath=configpath)
        self.lane_offset = self.config["lane_offset"]
        self.lane_x_bound = (0, self.config["lane_length"])
        self.lane_y_bound = (self.lane_offset, self.lane_offset + self.config["lane_count"]*self.config["lane_width"])
        self.time_step = 0.1
        self.time: int = 0

        self.scenario: Union[Scenario, None] = None
        self.planning_problem_set: Union[PlanningProblemSet, None] = None
        self.obstacles: Dict[int, Vehicle] = dict()
        self.off_obstacles = list()
        # Storing for each lane id the order of vehicles in an ordered list
        self.occupancies: Dict[int, List[Tuple[int, float]]] = dict()
        self.remove_ids: List[int] = list()

        # Prepare plot settings
        set_non_blocking()
        plt.style.use("classic")
        inch_in_cm = 2.54
        fig_size = [30, 8]
        self.fig = plt.figure(figsize=(fig_size[0] / inch_in_cm, fig_size[1] / inch_in_cm))
        self.fig.gca().axis('equal')
        self.handles = {}

    def reset(self):
        """
        Resets the environment
        """
        self.time = 0

    def make(self, scenario_name: str):
        """
        Make function for creating the environment according to its configuration files.
        """
        self.scenario, self.planning_problem_set = CommonRoadFileReader("scenario/" + scenario_name + ".xml").open()

        for obstacle_config in self.config["obstacles"]:
            self.create_vehicle(object_id=self.scenario.generate_object_id(),
                                lane_id=obstacle_config["lane"]-1,
                                x_pos=obstacle_config["x_pos"],
                                speed=obstacle_config["speed"])

        self.update_neighbors()

    def step(self):
        """
        Environment step function. Iterates over all vehicles and computes their future state according to their
        control commands.
        """
        self.remove_ids.clear()
        for object_id, vehicle in self.obstacles.items():
            vehicle.act(target_velocity=self.config["speed_limit"]/3.6)

            # if state is out of bounds remove vehicle from update list and add it to history
            if not vehicle.validate_state(x_bound=self.lane_x_bound, y_bound=self.lane_y_bound):
                self.off_obstacles.append(vehicle)
                self.remove_ids.append(object_id)

        for object_id in self.remove_ids:
            del self.obstacles[object_id]

        self.time += 1

        # Update Occupancies of objects
        self.update_occupancies()

        # Create follow up traffic randomly
        if self.time % 25 == 0 and self.time > 0:
            self.create_follow_up_traffic()

        # Update Environment information for each vehicle
        self.update_neighbors()

    def render(self):
        """
        Render function for visualizing the current scenario time step.
        """
        if self.time == 0:
            # Generate object_list
            object_list = list()
            for obj_id, obj in self.obstacles.items():
                object_list.append(obj.generate_dynamic_obstacle(object_id=obj_id))

            self.scenario.add_objects(scenario_object=object_list)
            draw_object(self.scenario, handles=self.handles, draw_params={'time_begin': self.time})
            draw_object(self.planning_problem_set)
            self.fig.canvas.draw()
            plt.gca().autoscale()
        else:
            # Update object positions
            object_list = list()
            for obj_id, obj in self.obstacles.items():
                found_id = False
                for check_id in range(len(self.scenario.obstacles)):
                    if self.scenario.obstacles[check_id].obstacle_id == obj_id:
                        self.scenario.obstacles[check_id].prediction = self.obstacles[obj_id].update_prediction()
                        found_id = True
                        break
                if found_id is False:
                    object_list.append(obj.generate_dynamic_obstacle(object_id=obj_id))

            if len(object_list) > 0:
                self.scenario.add_objects(scenario_object=object_list)
            redraw_obstacles(scenario=self.scenario, handles=self.handles, figure_handle=self.fig,
                             plot_limits=None, draw_params={'time_begin': self.time})
            draw_object(self.planning_problem_set)

    def create_follow_up_traffic(self):
        """
        Creates additional vehicles for continuous traffic flow.
        """
        speed1 = np.random.choice(np.arange(0.5, 0.8, 0.1))
        self.create_vehicle(object_id=self.scenario.generate_object_id(),
                            lane_id=1,
                            x_pos=5,
                            speed=speed1)

        speed2 = np.random.choice(np.arange(0.5, 0.9, 0.1))
        self.create_vehicle(object_id=self.scenario.generate_object_id(),
                            lane_id=2,
                            x_pos=5,
                            speed=speed2)

    def create_vehicle(self, object_id: int, lane_id: int, x_pos: int, speed: float) -> None:
        """
        Creates one Vehicle according to the Intelligent Driver Model and places it at the specified lane position.
        """
        idm_vehicle = IDMVehicle()

        # get random x coordinate on first 50m on lane
        y = lane_id * 3.5
        delta, orientation = 0, 0
        init_state = State(**{"position": np.array([x_pos, y]),
                              "steering_angle": delta,
                              "velocity": speed * (self.config["speed_limit"]/3.6),
                              "orientation": orientation,
                              "time_step": self.time})
        idm_vehicle.init_vehicle(state=init_state)

        # Add vehicle to dictionary
        self.obstacles[object_id] = idm_vehicle
        self.longitudinal_occupancy_check(lane_id=lane_id, object_id=object_id, x_coord=x_pos)

    def longitudinal_occupancy_check(self, lane_id: int, object_id: int, x_coord: float) -> None:
        """
        Checks the position of a specified car on a specified lane and inserts its information at the list index
        corresponding to the vehicle's lane order position.
        """
        try:
            occupancy_list = self.occupancies[lane_id]
            counter = 0
            inserted = False
            for check_id, check_pos in occupancy_list:
                if x_coord < check_pos:
                    occupancy_list.insert(counter, (object_id, x_coord))
                    inserted = True
                    break
                counter += 1
            if inserted is False:
                occupancy_list.append((object_id, x_coord))

            self.occupancies[lane_id] = occupancy_list

        except KeyError:
            self.occupancies[lane_id] = [(object_id, x_coord)]

    def update_occupancies(self) -> None:
        """
        Updating the occupancies of each car on each lane. Only the x position will be stored as we are preliminary
        dealing with straight lanes only. Additionally obstacles which are off bounds will be removed from list.
        """
        occupancy_dict = dict()

        for lane_id in list(self.occupancies.keys()):
            occupancies = list()
            for object_id, _ in self.occupancies[lane_id]:
                if object_id not in self.remove_ids:
                    occupancies.append((object_id, self.obstacles[object_id].current_state.position[0]))
            occupancy_dict[lane_id] = occupancies

        self.occupancies = occupancy_dict

    def update_neighbors(self) -> None:
        """
        Updates neighbor information for each car. Each car has one dictionary containing information about the
        neighbor's state information.
        """
        for lane_id in list(self.occupancies.keys()):
            counter = 0
            for object_id, _ in self.occupancies[lane_id]:
                # check position of vehicle
                if counter == 0 and len(self.occupancies[lane_id]) == 1:
                    # Only one vehicle on lane
                    follower = None
                    leader = None
                elif counter == 0 and counter < len(self.occupancies[lane_id]):
                    # first vehicle
                    follower = None
                    leader = self.occupancies[lane_id][counter + 1]
                elif counter == len(self.occupancies[lane_id]) - 1:
                    # last vehicle
                    follower = self.occupancies[lane_id][counter - 1]
                    leader = None
                else:
                    follower = self.occupancies[lane_id][counter - 1]
                    leader = self.occupancies[lane_id][counter + 1]

                counter += 1

                if follower is None:
                    follow_state = None
                else:
                    follow_state = self.obstacles[follower[0]].current_state

                if leader is None:
                    lead_state = None
                else:
                    lead_state = self.obstacles[leader[0]].current_state

                self.obstacles[object_id].update_neighbors(left=None, right=None,
                                                           behind=follow_state, lead=lead_state)
