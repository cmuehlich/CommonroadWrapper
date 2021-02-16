import abc
from enum import Enum
from typing import List, Union, Tuple
import numpy as np
from scipy.integrate import odeint

from utils.helper import get_state_from_arr, get_arr_from_state

from commonroad.scenario.trajectory import State, Trajectory
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels.vehicle_parameters import VehicleParameters
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1 as carParameters
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks as bicycleModel
from vehiclemodels.init_ks import init_ks

# Proportional Gains for controller
K_P_LONGITUDINAL = 0.1
K_P_LATERAL = 0.1


class VehicleControlTypes(Enum):
    CONTROLLED = 0
    IDM = 1


class Control:
    def __init__(self, accel: float, steering_angle_rate: float):
        self.accel = accel
        self.steering_angle_rate = steering_angle_rate

    def get_arr(self) -> np.ndarray:
        return np.array([self.steering_angle_rate, self.accel])


class Vehicle(abc.ABC):
    def __init__(self, control_type: VehicleControlTypes, vehicle_type: ObstacleType):
        self.parameters = carParameters()
        self.control_type = control_type
        self.vehicle_type = vehicle_type
        self.shape = Rectangle(length=self.parameters.l, width=self.parameters.w)

        self.current_state: Union[State, None] = None
        self.time_step: int = 0
        self.current_control: Union[Control, None] = None
        self.trajectory: List[State] = list()
        self.control_history: List[Control] = list()
        self.in_road_bounds: bool = False

        self.neighbors = {"behind": Union[None, DynamicObstacle], "leading": Union[None, DynamicObstacle],
                          "left": Union[None, DynamicObstacle], "right": Union[None, DynamicObstacle]}

    def init_vehicle(self, state: State) -> None:
        """
        Initializing vehicle state.
        """
        self.time_step = state.time_step

        self.current_state = state
        self.trajectory.append(state)

    def update_neighbors(self, left: Union[None, State], right: Union[None, State],
                         behind: Union[None, State], lead: Union[None, State]):
        self.neighbors["behind"] = behind
        self.neighbors["leading"] = lead
        self.neighbors["left"] = left
        self.neighbors["right"] = right

    def act(self, target_velocity: float) -> State:
        """
        Computes state transition according to Single Track Kinematic Model (Commonroad-Vehicles)
        """
        # Init time steps with scenario base time step
        delta_t = 0.1
        t = np.array([0, delta_t])

        # Get Control Commands
        ctrl_set = self.set_control_input(target_velocity=target_velocity)
        ctrl = ctrl_set.get_arr()

        # Init State
        init_state = get_arr_from_state(state=self.current_state)
        x0 = init_ks(init_state.tolist())

        # Integrate transition function
        x1 = odeint(self._state_transition, x0, t, args=(ctrl.tolist(), self.parameters))

        # Create state
        self.time_step += 1
        new_state = get_state_from_arr(arr=x1[1, :], time_step=self.time_step)

        # Update current state
        self.current_state = new_state
        self.trajectory.append(new_state)
        self.control_history.append(ctrl_set)

        return new_state

    def validate_state(self, x_bound: Tuple[float, float], y_bound: Tuple[float, float]) -> bool:
        """
        Check if vehicle state is within road bounds.
        """
        x, y = self.current_state.position
        if x_bound[0] <= x <= x_bound[1] and y_bound[0] <= y <= y_bound[1]:
            return True
        else:
            return False


    @staticmethod
    def _state_transition(x: np.ndarray, t: np.ndarray, u: np.ndarray, p: VehicleParameters) -> np.ndarray:
        new_state = bicycleModel(x, u, p)
        return new_state

    @staticmethod
    def _longitudinal_control(vel_set: float, vel_act: float) -> float:
        error = vel_set - vel_act
        return K_P_LONGITUDINAL * error

    @staticmethod
    def _lateral_control(delta_set: float, delta_act: float) -> float:
        error = delta_set - delta_act
        return K_P_LATERAL * error

    def plan_motion(self):
        pass

    @abc.abstractmethod
    def set_control_input(self, target_velocity: float) -> Control:
        raise NotImplementedError()

    @abc.abstractmethod
    def _compute_acceleration(self, target_velocity: float) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def _compute_steering_angle(self) -> float:
        raise NotImplementedError()

    def update_prediction(self) -> TrajectoryPrediction:
        prediction = TrajectoryPrediction(trajectory=Trajectory(initial_time_step=self.trajectory[0].time_step,
                                                                state_list=self.trajectory),
                                          shape=self.shape)
        return prediction

    def generate_dynamic_obstacle(self, object_id: int) -> DynamicObstacle:
        prediction = TrajectoryPrediction(trajectory=Trajectory(initial_time_step=self.trajectory[0].time_step,
                                                                state_list=self.trajectory),
                                          shape=self.shape)
        return DynamicObstacle(obstacle_id=object_id,
                               obstacle_type=self.vehicle_type,
                               obstacle_shape=self.shape,
                               initial_state=self.trajectory[0],
                               prediction=prediction)
