#! /usr/bin/env python

"""
Model based control with shooting method example in turtlebot crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://bair.berkeley.edu/blog/2017/11/30/model-based-rl/

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import matplotlib.pyplot as plt

import openai_ros_envs.crib_task_env

class Memory:
  def __init__(self, max_memory):
    self._max_memory = max_memory
    self._samples = []

  def add_sample(self, sample):
    self._samples.append(sample)
    if len(self._samples) > self._max_memory:
      self._samples.pop(0)

  def sample(self, num_samples):
    if num_samples > len(self._samples):
      return random.sample(self._samples, len(self._samples))
    else:
      return random.sample(self._samples, num_samples)

if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", anonymous=True, log_level=rospy.INFO)
  env_name = "TurtlebotCrib-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # Set parameters
  num_episodes = 10
  num_steps = 10
  batch_size =  16
  memory = Memory(50000)

  for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    train_samples = memory.sample(num_samples=batch_size)
    nn_model.train()
