#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

import envs.crib_nav_task_env
from utils import bcolors, obs_to_state

    
def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

if __name__ == "__main__":
  rospy.init_node("crib_nav_vpg", anonymous=True, log_level=rospy.WARN)
  env_name = 'CribNav-v0'
  env = gym.make(env_name)
  rospy.loginfo("CribNav environment set")

  for ep in range(num_episodes):
    obs, info = env.reset()
    print(
      bcolors.WARNING,
      "Env reset ...",
      "\nEpisode: {}".format(episode),
      "\nRobot init position: {}".format(obs[:2]),
      "\nGoal position: {}".format(info["goal_position"]),
      bcolors.ENDC
    )
    state = obs_to_state(obs, info)
    done = False
    episode_reward = []
    for st in range(num_steps):
      batch_state.append(state.copy())
      action = sess.run(actions, feed_dict={state_ph: state[None,:]})[0]
    
