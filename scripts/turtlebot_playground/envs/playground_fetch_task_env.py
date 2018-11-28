from __future__ import print_function

import rospy
import numpy as np
import pandas as pd
import time
import math
import random
import tf
from gym import spaces
from .turtlebot_robot_env import TurtlebotRobotEnv
from gym.envs.registration import register

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Point
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
    self.init_pose = Pose()
    self.curr_pose = Pose()
    self.goal_position = Point()
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
    # set turtlebot init pose
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
    # set red_ball init position and velocity
    mag_ball = random.uniform(0 ,5)
    ang_ball = random.uniform(-math.pi, math.pi)
    x_ball = mag_ball * math.cos(ang_ball)
    y_ball = mag_ball * math.sin(ang_ball)
    ball_state = ModelState()
    ball_state.model_name = "red_ball"
    ball_state.pose.position.x = x_ball
    ball_state.pose.position.y = y_ball
    ball_state.pose.position.z = 3
    ball_state.twist.linear.x = random.uniform(-4, 4)
    ball_state.twist.linear.y = random.uniform(-4, 4)
    ball_state.twist.linear.z = random.uniform(-0.01, 0.01)
    ball_state.reference_frame = "world"
    # publish model_state to set bot
    rate = rospy.Rate(100)
    for _ in range(10):
      self.set_robot_state_publisher.publish(robot_state)
      self.set_ball_state_publisher.publish(ball_state)
      rate.sleep()
      
    self.init_pose = robot_state.pose
    self.curr_pose = robot_state.pose
    self.goal_position = ball_state.pose.position
    rospy.logwarn("Robot was initiated as {}".format(self.init_pose))
    # Episode cannot done
    self._episode_done = False
    # Give the system a little time to finish initialization
    rospy.logdebug("Finish initialize robot.")
    
    return self.init_pose, self.goal_position
    
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
    # get observations in observation_space
    bridge = CvBridge()
    rgb_obs = bridge.imgmsg_to_cv2(self.camera_rgb_image_raw, "bgr8")
    # convert depth_image into pandas DataFrame to sub nan with inf
    depth_df = pd.DataFrame(bridge.imgmsg_to_cv2(self.camera_depth_image_raw, "32FC1")).replace(np.nan, np.inf)
    depth_obs = depth_df.values
    # convert laser_scan into pandas DataFrame to sub nan with inf
    laser_df = pd.DataFrame(list(self.laser_scan.ranges)).replace(np.nan, np.inf)
    laser_obs = laser_df.values
    angvel_obs = np.array([self.imu_data.angular_velocity.x,
                           self.imu_data.angular_velocity.y,
                           self.imu_data.angular_velocity.z])
    linacc_obs = np.array([self.imu_data.linear_acceleration.x,
                           self.imu_data.linear_acceleration.y,
                           self.imu_data.linear_acceleration.z])
    self.observation = (rgb_obs, depth_obs, laser_obs, angvel_obs, linacc_obs)
    
    rospy.logdebug("Observation ==> {}".format(self.observation))
    # get info
    model_states = self.get_model_states()
    self.curr_pose = model_states.pose[model_states.name.index("mobile_base")]
    self.goal_position = model_states.pose[model_states.name.index("red_ball")].position
    
    return self.observation

  def _post_information(self):
    """
    Return:
      info: {"init_pose", "goal_position", "current_pose"}
    """
    self.info = {
      "initial_pose": self.init_pose,
      "goal_position": self.goal_position,
      "current_pose": self.curr_pose
    }
    
    return self.info

  def _is_done(self):
    """
    Return True if self._episode_done
    """
    
    return self._episode_done

  def _compute_reward(self):
    distance = np.linalg.norm(
      np.array([
        self.curr_pose.position.x - self.goal_position.x,
        self.curr_pose.position.y - self.goal_position.y,
        self.curr_pose.position.z - self.goal_position.z
      ])
    )
    if distance < 0.1:
      reward = 1
      self._episode_done = True
      rospy.logwarn("\n!!!\nTurtlebot found the ball\n!!!")
    else:
      reward = 0
      self._episode_done = False
      rospy.logwarn("TurtleBot is working very hard to locate the red ball @ {}...".format(self.goal_position))
    rospy.logdebug("Compute reward done. \nreward = {}".format(reward))
    
    return reward


def _check_publishers_connection(self):
  raise NotImplementedError()

def _cmd_vel_pub(self):
  raise NotImplementedError()
