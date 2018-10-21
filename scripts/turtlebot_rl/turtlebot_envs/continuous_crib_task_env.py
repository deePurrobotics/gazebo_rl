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

from gazebo_msgs.msg import ModelState
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
    This task-env is designed for TurtleBot navigating to a random placed goal
    in a walled world. Action and state space will both be set to continuous. 
    """
    # action limits
    self.max_linear_speed = 2.0
    self.max_angular_speed = math.pi
    # observation limits
    self.max_x = 5
    self.max_y = 5
    self.max_vx = 2
    self.max_vy = y
    self.max_cosyaw = 1
    self.max_sinyaw = 1
    self.max_yawdot = math.pi
    # action space
    self.high_action = np.array([self.max_linear_speed, self.max_angular_speed])
    self.low_action = np.array([-self.max_linear_speed, -self.max_angular_speed])
    # observation space
    self.high_observation = np.array(
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
    self.low_observation = -self.high_observation
    # robot initial position and goal position
    self.init_position = np.zeros(2)
    self.current_position = np.zeros(2)
    self.previous_position = np.zeros(2)
    self.goal_position = np.zeros(2)
    # Set model state
    self.set_model_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
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
    self.current_position = self.init_position
    # set turtlebot inside crib, away from crib edges
    x = random.uniform(-self.max_x+1, self.max_x-1)
    y = random.uniform(-self.max_y+1, self.max_y-1)
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
    # publish model_state to set bot
    self.set_model_state_publisher.publish(model_state)

    self.init_position = np.array([x, y])
    self.previous_position = self.init_position
    rospy.logdebug("Robot was initiated as {}".format(model_state.pose))

    # set goal point
    goal_x = random.uniform(-self.max_x+.5, self.max_x-.5)
    goal_y = random.uniform(-self.max_y+.5, self.max_y-.5)
    self.goal_position = np.array([goal_x, goal_y])
    # reset goal if it too close to bot's original position
    while np.linalg.norm(self.goal_position - self.init_position) <= 0.5:
      rospy.logerr("Goal was set too close to the robot, reset the goal...")
      goal_x = random.uniform(-self.max_x+.5, self.max_x-.5)
      goal_y = random.uniform(-self.max_y+.5, self.max_y-.5)
      self.goal_position = np.array([goal_x, goal_y])
    rospy.logdebug("Goal point was set @ {}".format(self.goal_position))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Finish initialize robot.")
    
    return self.init_position, self.goal_position
    
  def _take_action(self, action):
    """
    Set linear and angular speed for Turtlebot and execute

    Args:
      action: Twist().
    """
    rospy.logdebug("TurtleBot2 Base Twist Cmd>>\nlinear: {}\nangular{}".format(action.linear.x, action.angular.z))
    self._check_publishers_connection()
    rate = rospy.Rate(100)
    for _ in range(10):
      self._cmd_vel_pub.publish(action)
      rospy.logdebug("cmd_vel: \nlinear: {}\nangular{}".format(action.linear.x, action.angular.z))
      rate.sleep()


def _check_publishers_connection(self):
  raise NotImplementedError()
