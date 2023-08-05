from abc import abstractmethod
import math
from typing import Optional

from gymnasium import Env
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class ParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1,
            "vehicles_count": 0,
            "add_walls": True,
            "start_location": [0,0]
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        num_points = self.config["quantized_line_points"]
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))
            quantized_line_1 = self._quantize_straight_line_positions([x, y_offset], [x, y_offset+length], num_points=num_points, horizontal_offset=-width/2)
            quantized_line_2 = self._quantize_straight_line_positions([x, -y_offset], [x, -y_offset-length], num_points=num_points, horizontal_offset=-width/2)
            if k == 0:
                self.discretized_line_positions = np.concatenate((quantized_line_1, quantized_line_2))
            else:
                self.discretized_line_positions = np.concatenate((self.discretized_line_positions, quantized_line_1, quantized_line_2))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _quantize_straight_line_positions(self, init_line_position: list, end_line_position: list, num_points: int, horizontal_offset: float=0) -> np.ndarray:
        assert num_points > 0, "num_points must be strictly greater than zero'"
        quantized_line_positions = []
        # Initial positioning of the quantized point on the line
        x_pos = init_line_position[0] + horizontal_offset # This is usually relative to the width of the lane
        y_pos = init_line_position[1]
        for _ in range(num_points):
            quantized_line_positions.append([x_pos, y_pos])
            longitudinal_diff = (init_line_position[0]-end_line_position[0])/num_points
            lateral_diff = (init_line_position[1]-end_line_position[1])/num_points
            x_pos -= longitudinal_diff
            y_pos -= lateral_diff
        return np.asarray(quantized_line_positions)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            starting_location = [i*20 + self.config['start_location'][0], self.config['start_location'][1]]
            vehicle = self.action_type.vehicle_class(self.road, starting_location, 2*np.pi*self.np_random.uniform(), 0)
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        # Goal
        lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
        lane = self.road.network.get_lane(lane_index)
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)
        empty_spots.remove(lane_index)

        # Other vehicles
        for i in range(self.config["vehicles_count"]):
            if not empty_spots:
                continue
            lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            v = Vehicle.make_on_lane(self.road, lane_index, 4, speed=0)
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        # Walls
        if self.config['add_walls']:
            width, height = 70, 42
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        computed_reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
        return computed_reward

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def compute_cost(self, achieved_goal: np.ndarray) -> float:
        """Determine costs. The vehicle should stay out of a certain range of the parking lines."""
        # Calculate cost on constraint violations
        # TODO: Only works for one controlled vehicle
        cost = {}
        get_quantized_line_dist = lambda x: np.linalg.norm(self.controlled_vehicles[0].position-np.array([x[0], x[1]]))
        quantized_line_dist = [get_quantized_line_dist(position) for position in self.discretized_line_positions]
        cost["cost_sum"] = self.config['cost_delta_distance'] - min(quantized_line_dist) if min(quantized_line_dist) < self.config['cost_delta_distance'] else 0
        return cost

    #TODO: Refactor this
    def _cost(self) -> float:
        cost = {}
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        cost["cost"] = sum(self.compute_cost(agent_obs['achieved_goal'])["cost_sum"] for agent_obs in obs)
        # cost["cost"] = 
        # Added additional cost functions here
        return cost

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached or time is over."""
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return bool(crashed or success)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time is over."""
        return self.time >= self.config["duration"]


class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__({"vehicles_count": 10})
