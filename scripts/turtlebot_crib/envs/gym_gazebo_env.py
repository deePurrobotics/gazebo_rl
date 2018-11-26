import numpy as np
import rospy
import gym # https://github.com/openai/gym/blob/master/gym/core.py
from gym.utils import seeding
from .gazebo_connection import GazeboConnection

class GymGazeboEnv(gym.Env):
  """
  Gazebo env converts standard openai gym methods into Gazebo commands
  
  To check any topic we need to have the simulations running, we need to do two things:
    1)Unpause the simulation: without that the stream of data doesnt flow. This is for simulations
      that are pause for whatever the reason
    2)If the simulation was running already for some reason, we need to reset the controlers.
      This has to do with the fact that some plugins with tf, dont understand the reset of the simulation and need to be reseted to work properly.
  """
  def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="WORLD"):
    # To reset Simulations
    rospy.logdebug("START init RobotGazeboEnv")
    self.gazebo = GazeboConnection(
      start_init_physics_parameters=True,
      reset_world_or_sim="WORLD"
    )
    self.seed()
    rospy.logdebug("END init RobotGazeboEnv")

  # Env methods
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    """
    Gives env an action to enter the next state,
    obs, reward, done, info = env.step(action)
    """
    # Convert the action num to movement action
    self.gazebo.unpauseSim()
    self._take_action(action)
    self.gazebo.pauseSim()
    obs = self._get_observation()
    done = self._is_done()
    reward = self._compute_reward()
    info = self._post_information()

    return obs, reward, done, info

  def reset(self):
    """ 
    obs, info = env.reset() 
    """
    # self.gazebo.pauseSim()
    rospy.logdebug("Reseting RobotGazeboEnvironment")
    self._reset_sim()
    self.gazebo.unpauseSim()
    self._set_init()
    self.gazebo.pauseSim()
    obs = self._get_observation()
    info = self._post_information()
    rospy.logdebug("END Reseting RobotGazeboEnvironment")
    return obs, info
  
  def close(self):
    """
    Function executed when closing the environment.
    Use it for closing GUIS and other systems that need closing.
    :return:
    """
    rospy.logwarn("Closing RobotGazeboEnvironment")
    rospy.signal_shutdown("Closing RobotGazeboEnvironment")

  def _reset_sim(self):
    """Resets a simulation
    """
    rospy.logdebug("START robot gazebo _reset_sim")
    self.gazebo.pauseSim()
    self.gazebo.resetSim()
    self.gazebo.unpauseSim()
    self._check_all_systems_ready()
    self.gazebo.pauseSim()    
    rospy.logdebug("END robot gazebo _reset_sim")
    
    return True

  def _set_init(self):
    """Sets the Robot in its init pose
    """
    raise NotImplementedError()

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    raise NotImplementedError()

  def _get_observation(self):
    """Returns the observation.
    """
    raise NotImplementedError()

  def _post_information(self):
    """Returns the info.
    """
    raise NotImplementedError()

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _is_done(self):
    """Indicates whether or not the episode is done ( the robot has fallen for example).
    """
    raise NotImplementedError()

  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _env_setup(self, initial_qpos):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    raise NotImplementedError()

