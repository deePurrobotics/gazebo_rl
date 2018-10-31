#! /usr/bin/env python

"""

Q-Learning example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import print_function

import gym
from gym import wrappers
import rospy
import numpy as np
import matplotlib.pyplot as plt
import utils

import envs.crib_nav_task_env


def obs_to_state(obs, info):
  """
  This function converts observation into state
  Args: 
    obs: [x, y, v_x, v_y, cos(theta), sin(theta), theta_dot]
        theta= robot orientation, alpha= angle between r->g and x-axis
    info: {"goal_position", ...}
  Returns:
    state: [r_norm, p_norm, alpha, alpha_dot, beta, beta_dot]
      r_norm: distance from map origin to robot
      p_norm: distance from robot to goal
      alpha: angle from map's x to r
      beta: angle from robot's x to p
      *_dot: angular velocity
  """
  # compute states
  r = obs[:2]
  p = info["goal_position"] - obs[:2]
  r_norm = np.linalg.norm(r) # sqrt(x^2+y^2)
  p_norm = np.linalg.norm(p)
  alpha = np.arctan2(obs[1], obs[0])
  alpha_dot = np.arctan2(obs[3], obs[2])
  # comput phi: angle from map's x_axis to p  
  x_axis = np.array([1, 0])
  y_axis = np.array([0, 1])
  cos_phi = np.dot(p, x_axis) / (np.linalg.norm(p)*np.linalg.norm(x_axis))
  sin_phi = np.dot(p, y_axis) / (np.linalg.norm(p)*np.linalg.norm(y_axis))
  phi = np.arctan2(sin_phi, cos_phi)
  # compute beta
  beta = phi - np.arctan2(obs[-2], obs[-3])
  beta_dot = obs[-1]
  state = np.array([r_norm, p_norm, alpha, alpha_dot, beta, beta_dot]).astype(np.float32)

  return state

if __name__ == "__main__":
  rospy.init_node("crib_nav_qlearn", anonymous=True, log_level=rospy.WARN)
  env_name = 'CribNav-v0'
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # Load parameters
  num_states = 100
  num_actions = 4
  Alpha = 1. # learning rate
  Gamma = 0.95 # reward discount
  num_episodes = 2000
  num_steps = 500
  low = env.observation_space.low
  # Initialize Q table
  Q = np.zeros([num_states, num_actions])
  reward_list = []
  for ep in range(num_episodes):
    # Reset env and get first observation
    obs = env.reset()
    state = utils.obs2state(obs, low)
    total_reward = 0
    done = False
    for st in range(num_steps):
      # Choose action greedily
      action = np.argmax(Q[state,:] + np.random.randn(1, num_actions)*(1./(ep+1)))
      # Get new state and reward
      obs, reward, done, _ = env.step(action)
      state1 = utils.obs2state(obs, low)
      # Update Q table
      Q[state, action] = Q[state, action] + Alpha*(reward + Gamma*np.max(Q[state1,:]) - Q[state, action])
      total_reward += reward
      state = state1
      rospy.loginfo("Total reward = {}".format(total_reward))
      if done:
        break

    reward_list.append(total_reward)

  print("Score over time: {}".format(sum(reward_list)/num_episodes))
  print("Final Q-table: {}".format(Q))

  plt.plot(reward_list)
