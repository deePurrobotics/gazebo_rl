#!/usr/bin/env python
from __future__ import print_function

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates

# import our training environment
from openai_ros_envs import crib_task_env # need write task env

rospy.init_node('gym_test', anonymous=True, log_level=rospy.WARN)    
env = gym.make('TurtleBotCrib-v0')

#obs = env.reset()

def pose_cb(data):
  po = data.pose[-1].position
  # tw = data.twist[-1]
  print(po)
  # print(tw)
    
# base_pose = rospy.Subscriber("/gazebo/model_states", ModelStates, pose_cb)

# rospy.spin()
