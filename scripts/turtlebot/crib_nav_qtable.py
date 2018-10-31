#! /usr/bin/env python

"""

Q-Learning example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import absolute_import, division, print_function

import gym
from gym import wrappers
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import bcolors

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
  if beta > math.pi:
    beta -= 2*math.pi
  elif beta < -math.pi:
    beta += 2*math.pi
  beta_dot = obs[-1]
  state = np.array([r_norm, p_norm, alpha, alpha_dot, beta, beta_dot]).astype(np.float32)

  return state

def discretize_state(state, boxes):
  """
  Converts continuous state into discrete states
  Args: 
    state:
    boxes:
  Returns:
    index: state index in Q table, represent in tuple
  """
  # match state into box
  index = []
  for i_s, st in enumerate(state):
    for i_b, box in enumerate(boxes[i_s]):
      if st >= box[0] and st <= box[1]:
        index.append(i_b)
        break
  assert len(index) == 6
  
  return tuple(index)


if __name__ == "__main__":
  # init node
  rospy.init_node("crib_nav_qlearn", anonymous=True, log_level=rospy.WARN)
  env_name = 'CribNav-v0'
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # parameters
  num_actions = 2
  Alpha = 1. # learning rate
  Gamma = 0.95 # reward discount
  num_episodes = 5000
  num_steps = 128
  # define state boxes
  box_1 = np.array([[0, 1.6], [1.6, 3.2], [3.2, 7]])
  box_2 = np.array([[0, 0.1], [0.1, 2], [2, 15]])
  box_3 = np.array([
    [-math.pi, -math.pi/2],
    [-math.pi/2, 0],
    [0, math.pi/2],
    [math.pi/2, math.pi]
  ])
  box_4 = np.array([
    [-math.pi, -math.pi/2],
    [-math.pi/2, -math.pi/12],
    [-math.pi/12, 0],
    [0, math.pi/12],
    [math.pi/12, math.pi/2],
    [math.pi/2, math.pi]
  ])
  box_5 = box_4
  box_6 = box_4
  boxes = [box_1, box_2, box_3, box_4, box_5, box_6]
  # initiate Q-table
  q_axes = []
  for b in boxes:
    q_axes.append(b.shape[0])
  q_axes.append(num_actions) # dont forget num_actions
  Q = np.zeros(q_axes)
  reward_list = []
  for episode in range(num_episodes):
    # Reset env and get first observation
    obs, info = env.reset()
    state = obs_to_state(obs, info)
    state_id = discretize_state(state, boxes)
    total_reward = 0
    done = False
    for step in range(num_steps):
      # Choose action with epsilon-greedy
      action_id = np.argmax(Q[state_id] + np.random.randn(num_actions)*(1./(episode+1)))
      if not action_id:
        action = np.array([env.action_space.high[0], env.action_space.low[1]]) # id=0 => [high_lin, low_ang]
      else:
        action = env.action_space.high # id=1 => [high_lin, high_ang]
      # Get new state and reward
      obs, reward, done, info = env.step(action)
      next_state = obs_to_state(obs, info)
      next_state_id = discretize_state(state, boxes)
      # Update Q table
      Q[state_id][action_id] = Q[state_id][action_id] + Alpha*(reward + Gamma*np.max(Q[next_state_id]) - Q[state_id][action_id])
      total_reward += reward
      # update state
      state = next_state
      state_id = next_state_id
      print(
        bcolors.OKGREEN, "\nEpisode: {}, Step: {}".format(episode, step), bcolors.ENDC,
        "\ncurrent_state: {}".format(state),
        "\nnext_state: {}".format(next_state),
        "\naction: {}".format(action),
        "\ngoal: {}".format(info["goal_position"])
      )
      rospy.loginfo("Total reward = {}".format(total_reward))
      if done:
        break

    reward_list.append(total_reward)

  print("Score over time: {}".format(sum(reward_list)/num_episodes))

  plt.plot(reward_list)
