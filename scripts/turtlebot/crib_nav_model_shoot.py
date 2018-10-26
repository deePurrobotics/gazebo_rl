#! /usr/bin/env python

"""
Model based control with random shoot method for crib nav task.
Controller randomly samples a bunch of actions, and pick the most gap closing one. 

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://bair.berkeley.edu/blog/2017/11/30/model-based-rl/

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import pickle
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

import envs.crib_nav_task_env
import utils
from utils import bcolors

import pdb

tf.enable_eager_execution()


def loss(model, x, y):
  y_ = model(x)
  return tf.losses.mean_squared_error(labels=y, predictions=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def obs_to_state(obs, info):
  """
  This function converts observation into state
  Args: 
    obs: [x, y, v_x, v_y, cos(theta), sin(theta), theta_dot]
        theta= robot orientation, alpha= angle between r->g and x-axis
    info: {"goal_position", ...}
  Returns:
    state: [x, y, v_x, v_y, cos(theta), sin(theta), theta_dot, dx, dy, cos(alpha), sin(alpha)]
  """
  # create state based on obs
  state = np.zeros(obs.shape[0]+4)
  state[:-4] = obs
  # compute alpha
  robot_position = obs[:2]
  goal_position = info["goal_position"]
  vec_x = np.array([1, 0])
  vec_y = np.array([0, 1])
  vec_r2g = goal_position - robot_position
  cos_alpha = np.dot(vec_r2g, vec_x) / (np.linalg.norm(vec_r2g)*np.linalg.norm(vec_x))
  sin_alpha = np.dot(vec_r2g, vec_y) / (np.linalg.norm(vec_r2g)*np.linalg.norm(vec_y))
  # append new states
  state[-4:-2] = vec_r2g # dx, dy
  state[-2:] = [cos_alpha, sin_alpha]
  state = state.astype(np.float32)
  
  return state

def random_action_samples(env, num_samples):
  """ 
  Generate an array of random action samples
  Args:
    env: gym environment
    num_action_samples: number of actions to be sampled randomly
  Returns:
    action_samples: in shape of (num_action_samples, action_space_dim)
  """
  action_space_dim = env.action_space.shape[0]
  action_samples = np.zeros([num_samples, action_space_dim])
  for i in range(action_samples.shape[0]):
    action_samples[i] = env.action_space.sample()
  
  return action_samples

def shoot_action(model, action_samples, state, goal):
  """ 
  Pick the action from the action_samples that most gap-closing
  Args:
    model: neural network dynamic model
    action_samples: 
    state: input to model
    goal: goal position
  Returns:
    action: best action picked from action_samples
  """
  max_gap_close = 0
  index = 0 # index of bext action
  stacs = np.zeros((action_samples.shape[0], state.shape[0]+action_samples.shape[1]))
  for i in range(stacs.shape[0]):
    stacs[i] = np.concatenate((state, action_samples[i])).astype(np.float32) # state-action pair
  pred_states = model(stacs) # predicted states of next time step
  gap_change = np.linalg.norm(pred_states[:,7:9],axis=1) - np.linalg.norm(state[7:9])
  i_mingap = np.argmin(gap_change) # find the index minimize the gap
  if gap_change[i_mingap] > 0: # if none action closing the gap
    print(bcolors.WARNING, "No action was found to close the gap", bcolors.ENDC)
    return np.zeros(action_samples.shape[1]) # do not move
  
  return action_samples[i_mingap]

def find_centered(memory):
  """
  Imagine the whole memory is a black hole. 
  Find index of the instance that is nearest to the center of the memory
  Args:
    memory: list of all memories
  Returns:
    index: index of the instance nearest to memory center
  """
  memory_array = np.array(memory)
  center = np.average(memory_array, axis=0)
  smallest_dist = np.inf
  for i, m in enumerate(memory_array):
    dist = np.linalg.norm(m-center)
    if dist <= smallest_dist:
      smallest_dist = dist
      index = i

  return index

if __name__ == "__main__":
  main_start = time.time()
  # init node
  rospy.init_node("crib_nav_mpc", anonymous=True, log_level=rospy.WARN)
  # create env
  env_name = "CribNav-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # set parameters
  num_actions = env.action_space.shape[0]
  num_states = env.observation_space.shape[0] + 4 # add cos and sin of vector from bot to goal
  num_episodes = 256
  num_steps = 128
  # load model from save checkpoint
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(num_states+num_actions,)),  # input shape required
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_states)
  ])
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  model_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/turtlebot/crib_nav/checkpoint"
  model_date = "20181025"
  checkpoint_dir = os.path.join(model_dir, model_date)
  root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())
  root.restore(tf.train.latest_checkpoint(checkpoint_dir))

  # load memories
  memory_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/turtlebot/crib_nav/memories"
  with open(os.path.join(memory_dir, "stacs_memory.txt"), "rb") as pkfile:
    stacs_memory = pickle.load(pkfile)
  with open(os.path.join(memory_dir, "nextstates_memory.txt"), "rb") as pkfile:
    nextstates_memory = pickle.load(pkfile)
  memory_size = 2**16

  # random shoot control with new samples
  num_action_samples = 128
  reward_storage = []
  for episode in range(num_episodes):
    obs, info = env.reset()
    state = obs_to_state(obs, info)
    goal = info["goal_position"]
    total_reward = 0
    done = False
    # compute control policies as long as sampling more
    for step in range(num_steps):
      action_samples = random_action_samples(env, num_samples=num_action_samples)
      action = shoot_action(
        model,
        action_samples,
        state,
        goal
      )
      obs, reward, done, info = env.step(action)
      next_state = obs_to_state(obs, info)
      stac = np.concatenate((state, action)).astype(np.float32)
      # incrementally update memories
      if len(stacs_memory) < memory_size:
        stacs_memory.append(stac)
      else:
        id_pop = find_centered(stacs_memory)
        stacs_memory.pop(id_pop)
        stacs_memory.append(stac)
      if len(nextstates_memory) < memory_size:
        nextstates_memory.append(next_state)
      else:
        id_pop = find_centered(nextstates_memory)
        nextstates_memory.pop(id_pop)
        nextstates_memory.append(next_state)
      total_reward += reward
      state = next_state
      print(
        bcolors.OKGREEN, "Episode: {}, Step: {}".format(episode, step), bcolors.ENDC,
        "\nCurrent position: {}".format(info["current_position"]),
        "\nGoal position: {}".format(info["goal_position"]),
        "\nAction taken: {}".format(action),
        bcolors.BOLD, "\nReward: {:.4f}".format(reward), bcolors.ENDC
      )
      if done:
        break
    reward_storage.append(total_reward)
    # Train with samples at the end of every episode
    dataset = utils.create_dataset(
      np.array(stacs_memory),
      np.array(nextstates_memory),
      batch_size=int(len(stacs_memory)/2),
      num_epochs=num_episodes
    )
    ep_start = time.time()
    for i, (x, y) in enumerate(dataset):
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
        zip(grads, model.variables),
        global_step
      )
      print("Training iter: {:d}, Loss: {:.4f}".format(i, loss_value))
    ep_end=time.time()
    print("Episode {:04d} training takes {:.4f}".format(episode, ep_end-ep_start))

    main_end = time.time()
    print(
      bcolors.HEADER,
      # "{:d} Random Samples was trained {:d} epochs".format(sample_size, num_epochs),
      # "\n{:d} new samples was collected and all samples were trained {:d} epochs".format(num_episodes*num_steps, num_episodes*4),
      "\nTotal execution time: {:.4f}".format(main_end-main_start),
      bcolors.ENDC
    )
