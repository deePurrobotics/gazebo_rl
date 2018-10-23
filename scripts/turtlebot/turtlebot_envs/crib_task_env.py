from __future__ import print_function

import rospy
import numpy as np
import time
import math
import random
import tf
from gym import spaces
from .turtlebot_robot_env import TurtlebotRobotEnv
from gym.envs.registration import register

from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Twist, Point

# Register crib env 
register(
  id='TurtlebotCrib-v0',
  entry_point='turtlebot_envs.crib_task_env:CribTaskEnv',
  timestep_limit=1000000,
)


class CribTaskEnv(TurtlebotRobotEnv):
  def __init__(self):
    """
    This Task Env is designed for having the TurtleBot in a walled world.
    It will learn how to move toward designated point.
    """
    # Crib env
    self.max_x = 5
    self.max_y = 5
    self.max_vx = 2
    self.max_vy = 2
    self.max_cosyaw = 1
    self.max_sinyaw = 1
    self.max_yawdot = math.pi
    self.high = np.array(
      [
        self.max_x,
        self.max_y,
        self.max_vx,
        self.max_vy,
        self.max_cosyaw,
        self.max_sinyaw,
        self.max_yawdot
      ]
    )
    self.low = -self.high
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(self.low, self.high)
    self.info = {}
    # robot initial position and goal position
    self.init_position = np.zeros(2)
    self.current_position = np.zeros(2)
    self.previous_position = np.zeros(2)
    self.goal_position = np.zeros(2)
    # Linear and angular speed for /cmd_vel
    self.linear_speed = 0.4 # rospy.get_param('/turtlebot2/linear_speed')
    self.angular_speed = 1 # rospy.get_param('/turtlebot2/angular_speed')        
    # Set model state
    self.set_model_state_publisher = rospy.Publisher(
      "/gazebo/set_model_state",
      ModelState,
      queue_size=100
    )
    # self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    self._episode_done = False
    # Here we will add any init functions prior to starting the MyRobotEnv
    super(CribTaskEnv, self).__init__()

  def _set_init(self):
    """ 
    Set initial condition for simulation
    1. Set turtlebot at a random pose inside crib by publishing /gazebo/set_model_state topic
    2. Set a goal point inside crib for turtlebot to navigate towards
    
    Returns: 
      init_position: array([x, y]) 
      goal_position: array([x, y])
      
    """
    rospy.logdebug("Start initializing robot...")
    # Set turtlebot inside crib, away from crib edges
    x = random.uniform(self.low[0]+1, self.high[0]-1)
    y = random.uniform(self.low[1]+1, self.high[1]-1)
    self.init_position = np.array([x, y])
    self.previous_position = self.init_position
    self.current_position = self.init_position
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
    model_state.reference_frame = "world"
    rate = rospy.Rate(100)
    for _ in range(10):
      self.set_model_state_publisher.publish(model_state)
      rate.sleep()
    # rospy.wait_for_service('/gazebo/set_model_state')
    # try:
    #   self.set_model_state(model_state)
    # except rospy.ServiceException as e:
    #   rospy.logerr("/gazebo/pause_physics service call failed")
    rospy.logdebug("Robot was initiated as {}".format(model_state.pose))

    # Set goal point
    goal_x = random.uniform(self.low[0]+.5, self.high[0]-.5)
    goal_y = random.uniform(self.low[1]+.5, self.high[1]-.5)
    self.goal_position = np.array([goal_x, goal_y])
    while np.linalg.norm(self.goal_position - self.init_position) <= 0.5:
    # while int(goal_x)==int(x) and int(goal_y)==int(y): # goal and bot should not in the same grid
      rospy.logerr("Goal was set too close to the robot, reset the goal...")
      goal_x = random.uniform(self.low[0]+.5, self.high[0]-.5)
      goal_y = random.uniform(self.low[1]+.5, self.high[1]-.5)
      self.goal_position = np.array([goal_x, goal_y])
    rospy.logdebug("Goal point was set @ {}".format(self.goal_position))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Finish initialize robot.")
    
    return self.init_position, self.goal_position

  def _take_action(self, action):
    """
    This set action will Set the linear and angular speed of the turtlebot
    based on the action number given.

    Args:
      action: The action integer that sets what movement to do next.
    """
    # Update bot previous position
    self.previous_position = self.current_position    
    # We construct 4 possible actions indicated by linear speed and angular speed combinations
    # We send these actions to the parent class turtlebot_robot_env
    lin_spd_pool = [-self.linear_speed, self.linear_speed]
    ang_spd_pool = [-self.angular_speed, self.angular_speed]
    i_l = action/len(ang_spd_pool) # index of speed in linear speed pool
    i_a = action%len(ang_spd_pool) 
    # We tell TurtleBot2 the linear and angular speed to set to execute
    self.move_base(
      linear_speed=lin_spd_pool[i_l],
      angular_speed=ang_spd_pool[i_a],
    )
    rospy.logdebug("END Set Action ==>"+str(action))

  def _observe(self):
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
    rospy.logdebug("Turtlebot is @ state of {}".format(model_states))
    x = model_states.pose[-1].position.x # turtlebot was the last model in model_states
    y = model_states.pose[-1].position.y
    self.current_position = np.array([x, y])
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
        
    observations = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    rospy.logdebug("Observations ==> {}".format(observations))
    return observations

  def _get_info(self):
    """
    Return robot's initial position and goal position 
    """
    self.info = {
      "init_position": self.init_position,
      "goal_position": self.goal_position,
      "current_position": self.current_position,
      "previous_position": self.previous_position
    }

    return self.info

  def _is_done(self, obs, goal):
    if np.linalg.norm(obs[:2]-goal) <= 0.2: # reaching goal position
      self._episode_done = True
      rospy.logdebug("Turtlebot reached destination !!!")
    else:
      self._episode_done = False
      rospy.logdebug("TurtleBot is working on its way to goal @ {}...".format(self.goal_position))

    return self._episode_done

  def _compute_reward(self):
    if not self._episode_done:
      if np.linalg.norm(self.current_position-self.goal_position) \
     < np.linalg.norm(self.previous_position-self.goal_position): # if move closer
        reward = 1
      else:
        reward = 0
    else:
      reward = 1 / np.linalg.norm(self.goal_position-self.init_position)
    rospy.logdebug("reward = {}".format(reward))
    
    return reward


