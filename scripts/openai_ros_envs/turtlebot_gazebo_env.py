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
    rospy.logdebug("START init TurtlebotGazeboEnv")
    self.gazebo = GazeboConnection(start_init_physics_parameters, reset_world_or_sim)
    
    self.max_pos = 5 # x, y
    self.max_vel = 1 # vs, vy
    self.max_tri = 1 # sinyaw, cosyaw
    self.max_w = 1 # yawdot
    self.init_position = None
    self.goal_position = None

    self.low = np.array(
      [
        -self.max_pos, # min_x
        -self.max_pos, # min_y
        -self.max_vel, # min_v_x
        -self.max_vel, # min_v_y
        -self.max_tri, # min_cos_yaw
        -self.max_tri, # min_sin_yaw
        -self.max_w, # min_yaw_dot
      ]
    )
    self.high = -self.low

    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(self.low, self.high)

    self.seed()
    self.reset()
    rospy.logdebug("END init TurtlebotGazeboEnv")

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    self.gazebo.unpauseSim()
    self._take_action(action)
    self.gazebo.pauseSim()
    self.state = self._get_obs() # continuous
    self.state = np.clip(self.state, -self.low, self.high)
    done = self._is_done(self.state, self.goal_position)
    reward = self._compute_reward(self.state, self.init_position, self.goal_position, done)

    return np.array(self.state), reward, done, {}

  def reset(self):
    self.gazebo.pauseSim()
    self.gazebo.resetSim()
    self.gazebo.unpauseSim()
    self._check_all_systems_ready()
    self.gazebo.pauseSim()
    self.init_position, self.goal_position = self._set_init_cond()
    self.gazebo.unpauseSim()
    self.gazebo.pauseSim()
    self.state = self._get_obs()
    
    return np.array(self.state)

  def close(self):
    """
    Function executed when closing the environment.
    Use it for closing GUIS and other systems that need closing.
    :return:
    """
    rospy.logdebug("Closing RobotGazeboEnvironment")
    rospy.signal_shutdown("Closing RobotGazeboEnvironment")
  
  def _set_init_cond(self):
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

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _is_done(self, obs, goal):
    """Indicates whether or not the episode is done ( the robot has fallen for example).
    """
    raise NotImplementedError()

  def _compute_reward(self, obs, goal, done):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _env_setup(self, initial_qpos):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    raise NotImplementedError()

