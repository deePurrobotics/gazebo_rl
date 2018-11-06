#!/usr/bin/env python

from __future__ import print_function

import rospy
import numpy as np
import time
import math
import random
import tf
from gym import spaces
from .cable_joint_robot_env import CableJointRobotEnv
from gym.envs.registration import register

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float32MultiArray

# Register crib env 
register(
  id='CablePoint-v0',
  entry_point='envs.cable_point_task_env:CablePointTaskEnv')


class CablePointTaskEnv(CableJointRobotEnv):
  def __init__(self):
    """
    This task-env is designed for cable-driven joint pointing at the desired goal. 
    Action and state space will be both set to continuous. 
    """
    # action limits
    self.max_force = 20
    self.min_force = 0
    # observation limits
    self.max_roll = math.pi
    self.max_pitch = math.pi
    self.max_yaw = math.pi
    self.max_roll_dot = 10*math.pi
    self.max_pitch_dot = 10*math.pi
    self.max_yaw_dot = 10*math.pi
    # action space
    self.high_action = np.array(4*[self.max_force])
    self.low_action = np.zeros(4)
    self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
    # observation space
    self.high_observation = np.array(
      [
        self.max_roll,
        self.max_pitch,
        self.max_yaw,
        self.max_roll_dot,
        self.max_pitch_dot,
        self.max_yaw_dot
      ]
    )
    self.low_observation = -self.high_observation
    self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation) 
    # action and observation
    self.action = np.zeros(self.action_space.shape[0])
    self.observation = np.zeros(self.observation_space.shape[0])
    # info, initial position and goal position
    self.init_orientation = np.zeros(3)
    self.current_orientation = np.zeros(3)
    self.previous_orientation = np.zeros(3)
    self.goal_orientation = np.zeros(3)
    self.info = {}
    # # set goal marker
    # self.set_pin_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
    self._episode_done = False
    # Here we will add any init functions prior to starting the MyRobotEnv
    super(CablePointTaskEnv, self).__init__()

  def _set_init(self):
    """ 
    Set initial condition for simulation
      Set a goal orientation for cable joint to point to
    Returns: 
      goal_orientation: array([roll, pitch, yaw])      
    """
    rospy.logdebug("Start initializing robot...")
    # set goal orientation
    goal_roll = np.random.uniform(-np.pi/9, np.pi/9) # +/-20 deg
    goal_pitch = np.random.uniform(-np.pi/9, np.pi/9)
    goal_yaw = np.random.uniform(-np.pi/9, np.pi/9)
    self.goal_orientation = np.array([goal_roll, goal_pitch, goal_yaw])
    time.sleep(0.1)

    # # set goal marker's model state
    # pin_state = ModelState()
    # pin_state.model_name = "pin"
    # pin_state.pose.position.x = goal_x
    # pin_state.pose.position.y = goal_y
    # pin_state.pose.position.z = 0
    # pin_state.reference_frame = "world"
    # # publish model_state to set bot
    # rate = rospy.Rate(100)
    # for _ in range(10):
    #   self.set_robot_state_publisher.publish(robot_state)
    #   self.set_pin_state_publisher.publish(pin_state)
    #   rate.sleep()
    
    rospy.logwarn("Goal orientation was set @ {}".format(self.goal_orientation))
    # Episode cannot done at beginning
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Finish initialize robot.")
    
    return self.goal_orientation
    
  def _take_action(self, action):
    """
    Set cable forces for cable-driven joint and execute.
    Args:
      action: (4,) numpy array.
    """
    rospy.logdebug("Cable forces >> {}".format(action))
    cmd_frc = Float32MultiArray()
    cmd_frc.data = action
    self._check_publishers_connection()
    rate = rospy.Rate(100)
    for _ in range(10):
      self._cmd_frc_pub.publish(cmd_frc)
      rospy.logdebug("cmd_frc: {}".format(cmd_frc))
      rate.sleep()

  def _get_observation(self):
    """
    Get observations from env
    Return:
      observation: [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
    """
    rospy.logdebug("Start Getting Observation ==>")
    link_states = self.get_link_states() # refer to cable_joint_robot_env
    # update previous position
    self.previous_orientation = self.current_orientation
    rospy.logdebug("link_states: {}".format(link_states))
    x = link_states.pose[4].orientation.x # cable_joint::link_3
    y = link_states.pose[4].orientation.y
    z = link_states.pose[4].orientation.z
    w = link_states.pose[4].orientation.w
    euler = tf.transformations.euler_from_quaternion([x, y, z, w])
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    roll_dot = link_states.twist[4].angular.x
    pitch_dot = link_states.twist[4].angular.y
    yaw_dot = link_states.twist[4].angular.z
    
    self.observation = np.array([roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot])
    self.current_orientation = np.array(euler)

    rospy.logdebug("Observation ==> {}".format(self.observation))
    
    return self.observation

  def _post_information(self):
    """
    Return:
      info: {"goal_orientation", "current_orientation", "previous_orientation"}
    """
    self.info = {
      "goal_orientation": self.goal_orientation,
      "current_orientation": self.current_orientation,
      "previous_orientation": self.previous_orientation
    }
    
    return self.info

  def _is_done(self):
    """
    If simulation exceeded time limit, return done==True
    Return:
      episode_done
    """
    self._episode_done = False
    rospy.logdebug("Cable joint is working on its way to goal @ {}...".format(self.goal_orientation))

    return self._episode_done

  def _compute_reward(self):
    if not self._episode_done:
    #   if np.linalg.norm(self.current_position-self.goal_position) \
    #  < np.linalg.norm(self.previous_position-self.goal_position): # if move closer
    #     reward = 0
    #   else:
    #     reward = -1
      reward = 0
    else:
      # if bot reached goal, the init distance will be the reward
      # reward = np.linalg.norm(self.goal_position-self.init_position)
      reward = 100
    rospy.logdebug("Compute reward done. \nreward = {}".format(reward))
    
    return reward


def _check_publishers_connection(self):
  raise NotImplementedError()

def _cmd_frc_pub(self):
  raise NotImplementedError()
