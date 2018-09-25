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

# Register crib env 
register(
  id='TurtlebotCrib-v0',
  entry_point='openai_ros_envs.crib_task_env:CribTaskEnv',
  timestep_limit=10000,
)


class CribTaskEnv(turtlebot_robot_env.TurtlebotRobotEnv):
  def __init__(self):
    """
    This Task Env is designed for having the TurtleBot in a walled world.
    It will learn how to move toward designated point.
    """
    # Crib env
    self.max_x = 5
    self.max_y = 5
    self.max_vx = 1
    self.max_vy = 1
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
    # Linear and angular speed for /cmd_vel
    self.linear_speed = 0.8 # rospy.get_param('/turtlebot2/linear_speed')
    self.angular_speed = 1 # rospy.get_param('/turtlebot2/angular_speed')        
    # Set model state service
    self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    self._episode_done = False
    # Here we will add any init functions prior to starting the MyRobotEnv
    super(CribTaskEnv, self).__init__()

  def _set_init(self):
    """ 
    Set initial condition for simulation
    1. Set turtlebot at a random pose inside crib by calling /gazebo/set_model_state service
    2. Set a goal point inside crib for turtlebot to navigate towards
    
    Returns: 
      init_position: [x, y] 
      goal_position: [x, y]
      
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
      rospy.logerr("/gazebo/pause_physics service call failed")
    rospy.logdebug("Turtlebot was set @ {}".format(model_state))

    # Set goal point
    goal_x = random.uniform(-5, 5)
    goal_y = random.uniform(-5, 5)
    while np.linalg.norm(np.array([goal_x, goal_y])-np.array([x, y])) <= 0.5:
    # while int(goal_x)==int(x) and int(goal_y)==int(y): # goal and bot should not in the same grid
      goal_x = random.uniform(-5, 5)
      goal_y = random.uniform(-5, 5)
    goal_position = np.array([goal_x, goal_y])
    rospy.logwarn("Goal point was set @ {}".format(goal_position))
    init_position = np.array([x, y])

    self._episode_done = False
    # Give the system a little time to finish initialization
    time.sleep(0.2)
    
    return init_position, goal_position

  def _take_action(self, action):
    """
    This set action will Set the linear and angular speed of the turtlebot
    based on the action number given.

    Args:
      action: The action integer that sets what movement to do next.
    """
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
      epsilon=0.05,
      update_rate=10,
      min_laser_distance=-1
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
    rospy.logdebug("Turtlebot is @ state of {}".format(model_states))
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
        
    observations = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    rospy.loginfo("Observations==>"+str(observations))
    return observations

  def _is_done(self, obs, goal):
    # if int(obs[0])==int(goal[0]) and int(obs[1])==int(goal[1]):
    if np.linalg.norm(obs[:2]-goal) <= 0.2:
      self._episode_done = True
      rospy.loginfo("Turtlebot reached destination !!!")
    else:
      self._episode_done = False
      rospy.loginfo("TurtleBot is working on its way to goal @ {}...".format(self.goal_position))

    return self._episode_done

  def _compute_reward(self, obs, init, goal):
    if not self._episode_done:
      # no reward if 
      reward = 0
    else:
      reward = 1
    rospy.loginfo("reward = {}".format(reward))
    
    return reward


