#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros_envs import crib_task_env # need write task env

rospy.init_node('gym_test', anonymous=True, log_level=rospy.WARN)    
env = gym.make('TurtleBotCrib-v0')

obs = env.reset()

