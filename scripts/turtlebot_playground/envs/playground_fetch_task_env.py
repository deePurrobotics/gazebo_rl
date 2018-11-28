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
from geometry_msgs.msg import Pose, Twist
from cv_bridge import CvBridge, CvBridgeError

# Register crib env 
register(
  id='PlaygroundFetch-v0',
  entry_point='envs.playground_fetch_task_env:PlaygroundFetchTaskEnv')

class PlaygroundFetchTaskEnv(TurtlebotRobotEnv):
  def __init__(self):
    """
    This task-env is designed for TurtleBot finding a random placed red ball 
    in its walled playground. Action and state space will be both set to continuous. 
    """
    # action limits
    self.max_linear_speed = .8
    self.max_angular_speed = math.pi / 3
    # observation limits
    # action space
    self.high_action = np.array([self.max_linear_speed, self.max_angular_speed])
    self.low_action = -self.high_action
    self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
    # observation space
    self.rgb_space = spaces.Box(low=0, high=255, shape=(480, 640, 3))
    self.depth_space = spaces.Box(low=0, high=np.inf, shape=(480,640))
    self.laser_space = spaces.Box(low=0,high=np.inf, shape=(640,))
    self.angvel_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    self.linacc_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
    self.observation_space = spaces.Tuple((
      self.rgb_space,
      self.depth_space,
      self.laser_space,
      self.angvel_space,
      self.linacc_space
    ))
    # info, initial position and goal position
    self.init_position = np.zeros(2)
    self.current_position = np.zeros(2)
    self.previous_position = np.zeros(2)
    self.goal_position = np.zeros(2)
    self.info = {}
    # Set model state
    self.set_robot_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
    self.set_ball_state_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
    # not done
    self._episode_done = False
    # Here we will add any init functions prior to starting the MyRobotEnv
    super(PlaygroundFetchTaskEnv, self).__init__()

  def _set_init(self):
    """ 
    Set initial condition for simulation
      1. Set turtlebot at a random pose inside playground by publishing /gazebo/set_model_state topic
      2. Set a goal point inside playground for red ball
    Returns: 
      init_position: array([x, y]) 
      goal_position: array([x, y])      
    """
    rospy.logdebug("Start initializing robot...")
    self.current_position = self.init_position
    # set turtlebot inside crib, away from crib edges
    mag = random.uniform(0, 1) # robot vector magnitude
    ang = random.uniform(-math.pi, math.pi) # robot vector orientation
    x = mag * math.cos(ang)
    y = mag * math.sin(ang)
    w = random.uniform(-1.0, 1.0)    
    robot_state = ModelState()
    robot_state.model_name = "mobile_base"
    robot_state.pose.position.x = x
    robot_state.pose.position.y = y
    robot_state.pose.position.z = 0
    robot_state.pose.orientation.x = 0
    robot_state.pose.orientation.y = 0
    robot_state.pose.orientation.z = math.sqrt(1 - w**2)
    robot_state.pose.orientation.w = w
    robot_state.reference_frame = "world"
    self.init_position = np.array([x, y])
    self.previous_position = self.init_position
    # set goal point using pole coordinate
    goal_r = random.uniform(0, 8) # goal vector magnitude
    goal_theta = random.uniform(-math.pi, math.pi) # goal vector orientation
    goal_x = goal_r * math.cos(goal_theta)
    goal_y = goal_r * math.sin(goal_theta)
    self.goal_position = np.array([goal_x, goal_y])
    # reset goal if it too close to bot's original position
    while np.linalg.norm(self.goal_position - self.init_position) <= 0.5:
      rospy.logerr("Goal was set too close to the robot, reset the goal...")
      goal_r = random.uniform(0, 4.8) # goal vector magnitude
      goal_theta = random.uniform(-math.pi, math.pi) # goal vector orientation
      goal_x = goal_r * math.cos(goal_theta)
      goal_y = goal_r * math.sin(goal_theta)
      self.goal_position = np.array([goal_x, goal_y])
    # set pin's model state
    ball_state = ModelState()
    ball_state.model_name = "red_ball"
    ball_state.pose.position.x = goal_x
    ball_state.pose.position.y = goal_y
    ball_state.pose.position.z = 2.5
    ball_state.reference_frame = "world"
    # publish model_state to set bot
    rate = rospy.Rate(100)
    for _ in range(10):
      self.set_robot_state_publisher.publish(robot_state)
      self.set_ball_state_publisher.publish(ball_state)
      rate.sleep()
    
    rospy.logwarn("Robot was initiated as {}".format(self.init_position))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Finish initialize robot.")
    
    return self.init_position, self.goal_position
    
  def _take_action(self, action):
    """
    Set linear and angular speed for Turtlebot and execute.
    Args:
      action: 2-d numpy array.
    """
    rospy.logdebug("TurtleBot2 Base Twist Cmd>>\nlinear: {}\nangular{}".format(action[0], action[1]))
    cmd_vel = Twist()
    cmd_vel.linear.x = action[0]
    cmd_vel.angular.z = action[1]
    self._check_publishers_connection()
    rate = rospy.Rate(100)
    for _ in range(10):
      self._cmd_vel_pub.publish(cmd_vel)
      rospy.logdebug("cmd_vel: \nlinear: {}\nangular{}".format(cmd_vel.linear.x,
                                                               cmd_vel.angular.z))
      rate.sleep()

  def _get_observation(self):
    """
    Get observations from env
    Return:
      observation: (rgb_image: (480,640,3),
                    depth_image: (480,640),
                    laser_scan: (640),
                    angular_velocity: (3,)
                    linear_acceleration: (3,))
    """
    rospy.logdebug("Start Get Observation ==>")
    model_states = self.get_model_states() # refer to turtlebot_robot_env
    # update previous position
    self.previous_position = self.current_position    
    rospy.logdebug("model_states: {}".format(model_states))
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
    
    self.observation = np.array([x, y, v_x, v_y, cos_yaw, sin_yaw, yaw_dot])
    rospy.logdebug("Observation ==> {}".format(self.observation))
    
    return self.observation

  def _post_information(self):
    """
    Return:
      info: {"init_position", "goal_position", "current_position", "previous_position"}
    """
    self.info = {
      "init_position": self.init_position,
      "current_position": self.current_position,
      "previous_position": self.previous_position
    }
    
    return self.info

  def _is_done(self):
    """
    If TurtleBot moves into a small circle around the goal, return done==True
    Args:
      obs: observation
      goal: goal position
    Return:
      episode_done
    """
    
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
      reward = 1
    rospy.logdebug("Compute reward done. \nreward = {}".format(reward))
    
    return reward


def _check_publishers_connection(self):
  raise NotImplementedError()

def _cmd_vel_pub(self):
  raise NotImplementedError()
