from __future__ import print_function

import rospy
import numpy as np
import time
import math
import random
import tf
from gym import spaces
import turtlebot_robot_env
from gym.envs.registration import register

from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Twist, Point

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
  id='TurtleBotCrib-v0',
  entry_point='openai_ros_envs.crib_task_env:TurtleBotCribEnv',
  timestep_limit=timestep_limit_per_episode,
  max_episode_steps = 1000
)

class TurtleBotCribEnv(turtlebot_env.TurtleBotEnv):
  def __init__(self):
    """
    This Task Env is designed for having the TurtleBot2 in a walled world.
    It will learn how to move toward designated point.
    """
        
    # Only variable needed to be set here
    # number_actions = rospy.get_param('/turtlebot2/n_actions')
    # self.action_space = spaces.Discrete(number_actions)
        
    # # We set the reward range, which is not compulsory but here we do it.
    # self.reward_range = (-np.inf, np.inf)
        
        
    #number_observations = rospy.get_param('/turtlebot2/n_observations')
        
    # Actions and Observations
    self.linear_speed = 0.4 # rospy.get_param('/turtlebot2/linear_speed')
    self.angular_speed = 0.4 # rospy.get_param('/turtlebot2/angular_speed')
    # self.obs_high = rospy.get_param('/turtlebot2/obs_high')
    # self.obs_low = rospy.get_param('/turtlebot2/obs_low')
    self.min_range = -1 # rospy.get_param('/turtlebot2/min_range')

    # We create two arrays based on the binary values that will be assigned
    # In the discretization method.
    # high = np.array(self.obs_high) # upper bound of observations
    # low = np.array(self.obs_low) # lower bound of observations 
    # # We only use two integers
    # self.observation_space = spaces.Box(low, high)
        
    rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
    rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
    # Rewards
    self.cumulated_steps = 0.0

    # Set model state service
    self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    self.init_position = None
    # Here we will add any init functions prior to starting the MyRobotEnv
    super(TurtleBotCribEnv, self).__init__()

  def _set_init_cond(self):
    """ Initial condition of simulation
        1. Call /gazebo/set_model_state service to set turtlebot at a random pose inside crib
        2. Set a goal for turtlebot to navigate
    """
    # Set turtlebot inside crib, away from crib edges
    x = random.uniform(-4, 4)
    y = random.uniform(-4, 4)
    w = random.uniform(-math.pi, math.pi)    
    model_state = ModelState()
    model_state.model_name = "mobile_base"
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = 0
    model_state.pose.orientation.x = 0
    model_state.pose.orientation.y = 0
    model_state.pose.orientation.z = 1
    model_state.pose.orientation.w = w
    model_state.twist.linear.x = 0.0
    model_state.twist.linear.y = 0
    model_state.twist.linear.z = 0
    model_state.twist.angular.x = 0.0
    model_state.twist.angular.y = 0
    model_state.twist.angular.z = 0.0
    model_state.reference_frame = "world"
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
      self.set_model_state(model_state)
    except rospy.ServiceException as e:
      print ("/gazebo/pause_physics service call failed")
    rospy.logwarn("Turtlebot was set @ {}".format(model_state))

    # Set goal point
    states = self.get_model_states()
    goal_x = random.uniform(-5, 5)
    goal_y = random.uniform(-5, 5)
    bot_x = states.pose[-1].position.x
    bot_y = states.pose[-1].position.y
    while math.floor(goal_x)==math.floor(bot_x) and math.floor(goal_y)==math.floor(bot_y): # goal and bot should not in the same grid
      goal_x = random.uniform(-5, 5)
      goal_y = random.uniform(-5, 5)
    goal_position = [goal_x, goal_y]
    rospy.logwarn("Goal point was set @ {}".format(goal_position))
    init_position = [bot_x, bot_y]

    self._episode_done = False
    time.sleep(0.2)
    
    return init_position, goal_position

  def _init_env_variables(self):
    """
    Inits variables needed to be initialised each time we reset at the start
    of an episode.
    :return:
    """
    # For Info Purposes
    self.cumulated_reward = 0.0
    # Set to false Done, because its calculated asyncronously
    self._episode_done = False
        
    # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
    # and sometimes still have values from the prior position that triguered the done.
    time.sleep(0.2)


  def _take_action(self, action):
    """
    This set action will Set the linear and angular speed of the turtlebot2
    based on the action number given.
    :param action: The action integer that set s what movement to do next.
    """
    rospy.logdebug("Start Set Action ==>"+str(action))
    # We convert the actions to speed movements to send to the parent class turtlebot_env
    lin_spd_pool = [-self.linear_speed, self.linear_speed]
    ang_spd_pool = [-self.angular_speed, self.angular_speed]
    i_l = action/len(ang_spd_pool) # index of speed in linear speed pool
    i_a = action%len(ang_spd_pool) 
    # We tell TurtleBot2 the linear and angular speed to set to execute
    self.move_base(
      linear_speed=lin_spd_pool[i_l],
      angular_speed=ang_spd_pool[i_a],
      epsilon=0.05,
      update_rate=10,
      min_laser_distance=self.min_range
    )
    rospy.logdebug("END Set Action ==>"+str(action))

  def _get_obs(self):
    """
    Here we define states as gazebo model_states
    :return:
    7 observations
    [
      x,
      y,
      v_x,
      v_y,
      cos(yaw),
      sin(yaw),
      yaw_dot
    ]
    """
    rospy.logdebug("Start Get Observation ==>")
    # We get the model states data
    model_states = self.get_model_states()
    x = model_states.pose[-1].position.x # turtlebot was the last model in model_states
    y = model_states.pose[-1].position.y
    v_x = model_states.twist[-1].linear.x
    v_y = model_states.twist[-1].linear.y
    quat = (
      model_states.pose[-1].orientation.x,
      model_states.pose[-1].orientation.y,
      model_states.pose[-1].orientation.z,
      model_states.pose[-1].orientation.w
    )
    euler = tf.transformations.euler_from_quaternion(quat)
    cos_yaw = math.cos(euler[2])
    sin_yaw = math.sin(euler[2])
    yaw_dot = model_states.twist[-1].angular.z
        
    observations = [x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot]
    rospy.logdebug("Observations==>"+str(observations))
    rospy.logdebug("END Get Observation ==>")
    return observations

  def _is_done(self, obs, goal):
    if math.floor(obs[0])==math.floor(goal[0]) and \
    math.floor(obs[1])==math.floor(goal[1]): # goal reached
      self._episode_done = True
      rospy.logwarn("Turtlebot reached destination")
    else:
      self._episode_done = False
      rospy.loginfo("TurtleBot is Ok ==>")

    return self._episode_done

  def _compute_reward(self, obs, init, goal, done):
    if not done:
      reward = -np.linalg.norm(np.array(obs[:2]) - np.array(goal)) \
               / np.linalg.norm(np.array(init) - np.array(goal))
    else:
      reward = 0
    rospy.logdebug("reward=" + str(reward))
    # self.cumulated_reward += reward
    # rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
    # self.cumulated_steps += 1
    # rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
    
    return reward


