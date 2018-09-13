import rospy
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gazebo_connection import GazeboConnection
from controllers_connection import ControllersConnection
#https://bitbucket.org/theconstructcore/theconstruct_msgs/src/master/msg/RLExperimentInfo.msg
from openai_ros.msg import RLExperimentInfo

# https://github.com/openai/gym/blob/master/gym/core.py
class TurtlebotGazeboEnv(gym.Env):

  def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True, reset_world_or_sim="SIMULATION"):
    self.max_x = 5
    self.max_y = 5
    self.goal_position = None

    self.low = np.array([-self.max_x, -self.max_y])
    self.high = np.array([self.max_x, self.max_y])

    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(self.low, self.high)

    self.seed()
    self.reset()

  def _set_init_pose(self):
    """Sets the Robot in its init pose
    """
    raise NotImplementedError()

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    raise NotImplementedError()

  def _get_obs(self):
    """Returns the observation.
    """
    raise NotImplementedError()

  def _init_env_variables(self):
    """Inits variables needed to be initialised each time we reset at the start
    of an episode.
    """
    raise NotImplementedError()

  def _set_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _is_done(self, observations):
    """Indicates whether or not the episode is done ( the robot has fallen for example).
    """
    raise NotImplementedError()

  def _compute_reward(self, observations, done):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _env_setup(self, initial_qpos):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    raise NotImplementedError()

