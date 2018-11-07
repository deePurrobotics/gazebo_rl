#! /usr/bin/env python

"""

Q-Learning example using cable-driven joint pointing environment
Rotate ro the desired orientation

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import absolute_import, division, print_function

import gym
from gym import wrappers
import rospy
import math
import numpy as np
import random
import os
import datetime
import matplotlib.pyplot as plt
import pickle

from utils import bcolors

import envs.crib_nav_task_env

class Boxes:
  """
  Define discrete boxes to segment the state space
  """
  def __init__(self):
    self.boxes = []
  def generate_boxes(self):
    # define state boxes
    box_1 = np.array([
      [-math.pi, -math.pi/15],
      [-math.pi/15, 0],
      [0, math.pi/15],
      [math.pi/15, math.pi],
    ])
    box_2 = box_1
    box_3 = box_1
    box_4 = np.array([
      [-np.inf, -math.pi],
      [-math.pi, 0],
      [0, math.pi],
      [math.pi, np.inf]
    ])
    box_5 = box_4
    box_6 = box_4
    self.boxes = [box_1, box_2, box_3, box_4, box_5, box_6]

    return self.boxes
  
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
  rospy.loginfo("CribNav environment set")
  # parameters
  num_actions = 2
  Alpha = 1. # learning rate
  Gamma = 0.8 # reward discount
  num_episodes = 1000
  num_steps = 128
  # define state boxes
  box_1 = np.array([[0, 1.6], [1.6, 3.2], [3.2, np.inf]])
  box_2 = np.array([[0, 0.3], [0.3, 3], [3, np.inf]])
  box_3 = np.array([
    [-math.pi, -math.pi/2],
    [-math.pi/2, 0],
    [0, math.pi/2],
    [math.pi/2, math.pi]
  ])
  box_4 = np.array([
    [-np.inf, -math.pi],
    [-math.pi, -math.pi/12],
    [-math.pi/12, 0],
    [0, math.pi/12],
    [math.pi/12, math.pi],
    [math.pi, np.inf]
  ])
  box_5 = box_3
  box_6 = box_4
  boxes = [box_1, box_2, box_3, box_4, box_5, box_6]
  # initiate Q-table
  q_axes = []
  for b in boxes:
    q_axes.append(b.shape[0])
  q_axes.append(num_actions) # dont forget num_actions
  Q = np.zeros(q_axes)
  reward_list = []
  # make storage for q tables
  qtable_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/turtlebot/crib_nav/qtables"
  today = today = datetime.datetime.today().strftime("%Y%m%d")
  qtable_dir = os.path.join(qtable_dir, today, "_dense_reward")
  if not os.path.exists(os.path.dirname(qtable_dir)):
    try:
      os.makedirs(qtable_dir)
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
  for episode in range(num_episodes):
    # Reset env and get first observation
    obs, info = env.reset()
    print(
      bcolors.WARNING, "Episode: {}".format(episode),
      "\nRobot init position: {}".format(obs[:2]),
      "\nGoal position: {}".format(info["goal_position"]),
      bcolors.ENDC
    )
    state = obs_to_state(obs, info)
    state_id = discretize_state(state, boxes)
    p_0 = state[1] # initial distance to goal
    epsilon = max(0.01, min(1, 1-math.log10((1+episode)/100.))) # explore rate
    episode_reward = 0
    done = False
    for step in range(num_steps):
      # Choose action with epsilon-greedy
      if random.random() < epsilon:
        action_id = random.randrange(num_actions) # explore
        print(bcolors.FAIL, "Explore")
      else:
        action_id = np.argmax(Q[state_id]) # exploit
      if not action_id:
        action = np.array([env.action_space.high[0], env.action_space.low[1]]) # id=0 => [high_lin, low_ang]
      else:
        action = env.action_space.high # id=1 => [high_lin, high_ang]
      # Get new state and reward
      obs, reward, done, info = env.step(action)
      next_state = obs_to_state(obs, info)
      next_state_id = discretize_state(state, boxes)
      reward = reward + (state[1]-next_state[1])/p_0
      # Update Q table
      Q[state_id][action_id] = Q[state_id][action_id] + Alpha*(reward + Gamma*np.max(Q[next_state_id]) - Q[state_id][action_id])
      episode_reward += reward
      # update state
      state = next_state
      state_id = next_state_id
      print(
        bcolors.OKGREEN, "\nEpisode: {}, Step: {}".format(episode, step), bcolors.ENDC,
        "\nRobot current position: {}".format(obs[:2]),
        "\nGoal: {}".format(info["goal_position"]),
        "\nAction: {}".format(action),
        "\nreward: {}".format(reward)
      )
      rospy.loginfo("Total reward = {}".format(episode_reward))
      if done:
        print(bcolors.WARNING, "\n!!!\nGOAL\n!!!\n", bcolors.ENDC)
        break
    print(bcolors.BOLD, "Episodic reward: {}".format(episode_reward), bcolors.ENDC)
    reward_list.append(episode_reward)
    # save qtable every 100 episode
    if not (episode+1) % 100:
      with open(os.path.join(qtable_dir, "qtable_{}-{}.txt".format(episode+1, num_episodes)), "wb") as pk:
        pickle.dump(Q,pk)

  print("Score over time: {}".format(sum(reward_list)/num_episodes))

  plt.plot(reward_list)
