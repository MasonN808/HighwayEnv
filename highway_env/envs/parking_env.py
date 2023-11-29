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
            # The higher the reward weights, the harder it is to achieve the goal
            "reward_weights": [1, 0.3, .02, .02, 0.02, 0.02],
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
            "start_location": [0,0],
            "start_angle": 0, # This is radians

            # # Costs
            # "constraint_type": ["lines", "speed"],
            # # Cost-speed
            # "speed_limit": 3,
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
            if "constraint_type" in self.config:
                cost = self._cost()
                info["cost"] = cost

        info["is_success"] = success
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()
        if "constraint_type" in self.config and "lines" in self.config["constraint_type"]:
            # Remove points around the goal
            self.road.objects = self._remove_boundaries_near_dest()

    def _create_road(self, spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10 # default is 10
        length = 8

        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

        if "constraint_type" in self.config and "lines" in self.config["constraint_type"]:
            # Add in the lane polynomial boundaries
            for k in range(spots+1): # Add one since we care about lines not parking spots
                x = (k + 1 - spots // 2) * (width + x_offset) - width # removed /2 since we care about lines not center of spots
                self._create_constraint_boundaries(x, y_offset, x, y_offset+length, line_width=1)
                self._create_constraint_boundaries(x, -y_offset, x, -y_offset-length, line_width=1)

    def _remove_boundaries_near_dest(self):
        """Removes bounaries within one lane of the vehicle."""
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        # Normalize the desired goal since it is not on the same scale
        norm_desired_goal = np.asarray([obs[0]['desired_goal'][i]*self.config['observation']['scales'][i] for i in range(0,2)])
        # Get the distance from the desired_goal to the quantized line points
        get_distance = lambda position: np.linalg.norm(np.array([norm_desired_goal[0], norm_desired_goal[1]]) - np.array([position[0], position[1]]))
        distances = [get_distance(road_object.position) for road_object in self.road.objects]

        # Sort both lists wrt to the distances
        zipped_lists = zip(self.road.objects, distances)
        sorted_pairs = sorted(zipped_lists, key=lambda x: x[1])

        # Unzip them
        sorted_road_objects, _ = zip(*sorted_pairs)

        # Only print the lane lines that are labelled for line constraints
        nearest_line_constraint_objs = []
        for road_object in sorted_road_objects:
            if len(nearest_line_constraint_objs) == 2:
                break
            
            if road_object.label == "line_constraint":
                nearest_line_constraint_objs.append(road_object)

        # Remove the nearest_line_constraint_objs from the sorted road objects but keep all other lines
        return [road_object for road_object in sorted_road_objects if road_object not in nearest_line_constraint_objs] 

    def _create_constraint_boundaries(self, x1: float, y1: float, x2: float, y2: float, line_width: float = 1):
        """
        Generate the 4 corners of a box with padding around a line.
        """
        # Calculate the min and max y values
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        mid = ((x1 + x2) / 2, (y1 + y2) / 2)

        obstacle = Obstacle(self.road, mid, heading=(np.pi / 2))
        obstacle.LENGTH, obstacle.WIDTH = (np.linalg.norm(np.array(max_y - min_y)), line_width)
        # Get the diagonal via the distance between two symmetrically opposite points
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        # Label it for future querying
        obstacle.label = "line_constraint"
        # Disables physical collision so cars can drive over it
        obstacle.collidable = False
        self.road.objects.append(obstacle)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        empty_spots = list(self.road.network.lanes_dict().keys())

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            starting_location = [i*20 + self.config['start_location'][0], self.config['start_location'][1]]
            starting_angle = 2*np.pi*self.np_random.uniform() if self.config['start_angle'] == "random" else self.config['start_angle']
            vehicle = self.action_type.vehicle_class(self.road, starting_location, starting_angle, 0)
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
            width, height = 100, 80
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

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        pseudo_reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
        if pseudo_reward > -self.config["success_goal_reward"]:
            return 100
        else:
            return 0
    
    # def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
    #     """
    #     Proximity to the goal is rewarded

    #     We use a weighted p-norm

    #     :param achieved_goal: the goal that was achieved
    #     :param desired_goal: the goal that was desired
    #     :param dict info: any supplementary information
    #     :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
    #     :return: the corresponding reward
    #     """
    #     computed_reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
    #     return computed_reward

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {}) for agent_obs in obs)
        # Add extra reward for finding goal
        # for agent_obs in obs:
        #     if self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']):
        #         reward += 1000
        reward += self.config['collision_reward'] * sum(v.crashed for v in self.controlled_vehicles)
        return reward

    def compute_cost_lines(self) -> float:
        """Determine line distance costs. The vehicle should stay out of a certain range of the parking lines."""
        for road_object in self.road.objects:
            if road_object.label == "line_constraint":
                # Check whether the polygons intersect between the car and the line polygons
                intersecting, _, _ = self.controlled_vehicles[0]._is_colliding(road_object, dt=1/self.config["simulation_frequency"])
                if intersecting:
                    return 1
        return 0

    def compute_cost_speed(self, achieved_goal: np.ndarray, absolute_cost=True) -> float:
        """Determine speed costs. The vehicle should stay within a certain speed."""
        speed = np.linalg.norm(np.array([achieved_goal[2], achieved_goal[3]]))
        # print(speed)
        if absolute_cost:
            return 1 if speed - self.config['speed_limit'] > 0 else 0

        return max(0, speed - self.config['speed_limit'])

    def _cost(self) -> float:
        cost = {}
        if len(self.config['constraint_type']) > 0:
            cost["cost"] = [0]*len(self.config['constraint_type'])
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        traversed = [False]*len(self.config['constraint_type'])
        # Append the costs in the order recevived in constraint_type
        if cost:
            for i in range(len(cost["cost"])):
                if self.config['constraint_type'][i]=="lines" and not traversed[i]:
                    cost["cost"][i] += sum(self.compute_cost_lines() for _ in obs)
                elif self.config['constraint_type'][i]=="speed" and not traversed[i]:
                    cost["cost"][i] += sum(self.compute_cost_speed(agent_obs['achieved_goal']) for agent_obs in obs)
                traversed[i] = True
                # TODO Add additional cost functions here
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
