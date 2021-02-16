from typing import Tuple, Union
import numpy as np

from agents.vehicle import Vehicle, VehicleControlTypes, Control

from commonroad.scenario.obstacle import ObstacleType


class IDMVehicle(Vehicle):
    def __init__(self):
        super().__init__(control_type=VehicleControlTypes.IDM,
                         vehicle_type=ObstacleType.CAR)

        # Default driver setting
        self.driver_attitude = {"accel_exp": 4,
                                "min_lead_dist": 2,
                                "time_gap": 1.6}

    def set_control_input(self, target_velocity: float) -> Control:
        accel = self._compute_acceleration(target_velocity=target_velocity)
        delta_rate = self._compute_steering_angle()
        return Control(accel=accel, steering_angle_rate=delta_rate)

    def _compute_acceleration(self, target_velocity: float) -> float:
        # term for acceleration strategy
        accel_term = (1 - np.power((self.current_state.velocity / target_velocity), self.driver_attitude["accel_exp"]))

        # term for braking strategy
        # get info of leading vehicle
        lead_distance, v_rel = self.get_info_of_lead_vehicle()

        if lead_distance is not None:
            dyn_term = (self.current_state.velocity * v_rel) / (np.sqrt(2) * self.parameters.longitudinal.a_max)
            safe_gap = self.driver_attitude["min_lead_dist"] + \
                self.current_state.velocity * self.driver_attitude["time_gap"] + dyn_term

            braking_term = np.power((safe_gap / lead_distance), 2)
            if braking_term < 0:
                braking_term = 0
        else:
            braking_term = 0

        acceleration = self.parameters.longitudinal.a_max * (accel_term - braking_term)
        return acceleration

    def _compute_steering_angle(self) -> float:
        # drive straight
        return 0.0

    def get_info_of_lead_vehicle(self) -> Union[Tuple[float, float], Tuple[None, None]]:
        if self.neighbors["leading"] is None:
            return None, None

        lead_vehicle = self.neighbors["leading"]

        # Compute distance in x direction
        lead_dist = lead_vehicle.position[0] - self.current_state.position[0]
        assert lead_dist > 0

        # Compute relative velocity
        vel_rel = self.current_state.velocity - lead_vehicle.velocity

        return lead_dist, vel_rel
