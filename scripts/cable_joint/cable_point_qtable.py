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

import envs.cable_point_task_env


def generate_boxes():
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
  boxes = [box_1, box_2, box_3, box_4, box_5, box_6]
  
  return boxes
  
def between_pis(angle_array):
  """
  Convert an angle in rad into a value in range: (-pi,pi)
  """
  for angle in angle_array:
    # make angle in range: (-2pi, 2pi)
    if angle > np.pi*2:
      angle %= np.pi*2
    elif angle < -np.pi*2:
      angle %= -np.pi*2
      # make anle in range: (-pi, pi)
    if angle>np.pi:
      angle = angle%(2*np.pi)-2*np.pi
    elif angle<-np.pi:
      angle = angle % (-2*np.pi) + 2*np.pi

  return angle
    
def obs_to_state(obs, info):
  """
  This function converts observation into state
  Args: 
    obs: [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot]
    info: {"goal_orientation", ...}
  Returns:
    state: [d_roll, d_pitch, d_yaw, roll_dot, pitch_dot, yaw_dot]
  """
  # compute states
  state = obs
  delta_orientation = between_pis(info["goal_orientation"] - obs[:3])
  state[:3] = delta_orientation

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
  rospy.init_node("cable_point_qlearn", anonymous=True, log_level=rospy.WARN)
  env_name = 'CablePoint-v0'
  env = gym.make(env_name)
  rospy.loginfo("CablePoint environment set")
  # parameters
  num_actions = 4
  Alpha = 1. # learning rate
  Gamma = 0.9 # reward discount
  num_episodes = 10
  num_steps = 128
  # define state boxes
  boxes = generate_boxes()
  # initiate Q-table
  q_axes = []
  for b in boxes:
    q_axes.append(b.shape[0])
  q_axes.append(num_actions) # dont forget num_actions
  Q = np.zeros(q_axes)
  reward_list = []
  # make storage for q tables
  qtable_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/cable_joint/cable_point/qtables"
  today = today = datetime.datetime.today().strftime("%Y%m%d")
  qtable_dir = os.path.join(qtable_dir, today)
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
      "\nRobot init orientation: {}".format(obs[:3]),
      "\nGoal orientation: {}".format(info["goal_orientation"]),
      bcolors.ENDC
    )
    state = obs_to_state(obs, info)
    state_id = discretize_state(state, boxes)
    p_0 = state[1] # initial distance to goal
    epsilon = max(0.01, min(1, 1-math.log10((1+episode)/95.))) # explore rate
    episode_reward = 0
    done = False
    for step in range(num_steps):
      # Choose action with epsilon-greedy
      if random.random() < epsilon:
        action_id = random.randrange(num_actions) # explore
        print(bcolors.FAIL, "Explore")
      else:
        action_id = np.argmax(Q[state_id]) # exploit
      action = np.zeros(num_actions)
      action[action_id] = 1
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
        "\nRobot current orientation: {}".format(obs[:3]),
        "\nGoal: {}".format(info["goal_orientation"]),
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
