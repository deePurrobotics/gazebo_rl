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

Alpha = rospy.get_param("/turtlebot2/alpha")
Epsilon = rospy.get_param("/turtlebot2/epsilon")
Gamma = rospy.get_param("/turtlebot2/gamma")
epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount")
nepisodes = rospy.get_param("/turtlebot2/nepisodes")
nsteps = rospy.get_param("/turtlebot2/nsteps")
running_step = rospy.get_param("/turtlebot2/running_step")
qlearn = qlearn.QLearn(
    actions=range(env.action_space.n),
    alpha=Alpha,
    gamma=Gamma,
    epsilon=Epsilon
)
initial_epsilon = qlearn.epsilon

observation = env.reset()
state = ''.join(map(str, observation))
action = qlearn.chooseAction(state)
