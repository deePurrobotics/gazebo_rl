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
import tensorflow.contrib.eager as tfe
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
    obs: [x, y, v_x, v_y, cos(ori), sin(ori), v_ori]
    info: {"goal_position", ...}
  Returns:
    state: [dx, dy, v_x, v_y, cos(ori), sin(ori), v_ori, cos(goal), sin(goal)]
  """
  # create state based on obs
  state = np.zeros(obs.shape[0]+2)
  state[:obs.shape[0]] = obs
  # compute angle(theta) between vector of "robot to goal" and "x-axis" of world
  robot_position = obs[:2]
  goal_position = info["goal_position"]
  vec_x = np.array([1, 0])
  vec_y = np.array([0, 1])
  vec_r2g = goal_position - robot_position
  cos_theta = np.dot(vec_r2g, vec_x) / (np.linalg.norm(vec_r2g)*np.linalg.norm(vec_x))
  sin_theta = np.dot(vec_r2g, vec_y) / (np.linalg.norm(vec_r2g)*np.linalg.norm(vec_y))
  # append new states
  state[:2] = info["goal_position"] - obs[:2] # distance
  state[-2:] = [cos_theta, sin_theta]
  state = state.astype(np.float32)

  return state

def generate_action_sequence(num_sequences, len_horizon, env):
  """ 
  Generate random action sequences with limited horizon
  Args:
    num_sequences
    len_horizon: length of each sequence
    env: gym environment
  Returns:
    action_sequences: in shape of (num_sequences, len_horizon)
  """
  action_sequences = np.zeros((num_sequences, len_horizon))
  for s in range(num_sequences):
    for h in range(len_horizon):
      action_sequences[s,h] = env.action_space.sample() # random action

  return action_sequences

if __name__ == "__main__":
  # init node
  rospy.init_node("crib_nav_mpc", anonymous=True, log_level=rospy.WARN)
  # create env
  env_name = "CribNav-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  main_start = time.time()
  # set parameters
  num_actions = env.action_space.shape[0]
  num_states = env.observation_space.shape[0] + 2 # add cos and sin of vector from bot to goal
  num_episodes = 128
  num_steps = 256
  num_sequences = 256
  len_horizon = 1024 # number of time steps the controller considers
  batch_size = 4096  
  # setup model
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(num_states+num_actions,)),  # input shape required
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_states)
  ])
  stacs_memory = []
  nextstates_memory = []
  memory_size = 2**16

  # Random Sampling
  sample_size = 50000
  rs_start = time.time()
  rospy.logdebug("Start random sampling...")
  sample_index = 0
  obs, info = env.reset()
  done = False
  state = obs_to_state(obs, info) # to be worked out
  while sample_index < sample_size:
    action = env.action_space.sample()
    obs, _, done, info = env.step(action)
    next_state = obs_to_state(obs, info) # to be worked out
    # append action-state and next state in to the memories
    stac = np.concatenate((state, action)).astype(np.float32)
    stacs_memory.append(stac)
    nextstates_memory.append(next_state.astype(np.float32))
    print(
      bcolors.OKBLUE, "Sample: {}".format(sample_index+1), bcolors.ENDC,
      "\ncurrent_state: {}".format(state),
      "\naction: {}".format(action),
      "\nnext_state: {}".format(next_state)
    )
    sample_index += 1
    state = next_state
    if not sample_index % 100:
      obs, info = env.reset()
      done = False
      state = obs_to_state(obs, info) # to be worked out
  rs_end = time.time()
  print("Random sampling takes: {:.4f}".format(rs_end-rs_start))

  # Train random sampled dataset
  num_epochs = 64
  dataset = utils.create_dataset(
    input_features=np.array(stacs_memory),
    output_labels=np.array(nextstates_memory),
    batch_size=batch_size,
    num_epochs=num_epochs
  )
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  global_step = tf.train.get_or_create_global_step()
  loss_value, grads = grad(
    model,
    np.array(stacs_memory),
    np.array(nextstates_memory)
  )
  # create check point
  model_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/checkpoint"
  today = datetime.datetime.today().strftime("%Y%m%d")
  checkpoint_prefix = os.path.join(model_dir, today, "ckpt")
  if not os.path.exists(os.path.dirname(checkpoint_prefix)):
    try:
      os.makedirs(checkpoint_prefix)
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise
  root = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model,
    optimizer_step=global_step
  )
  root.save(file_prefix=checkpoint_prefix)
  # start training
  rst_start = time.time()
  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    for i, (x,y) in enumerate(dataset):
      batch_start = time.time()
      # optimize model
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
      zip(grads, model.variables),
        global_step
      )
      # track progress
      epoch_loss_avg(loss_value)  # add current batch loss
      # log training
      print("Epoch: {}, Iteration: {}, Loss: {:.3f}".format(epoch, i, epoch_loss_avg.result()))
      batch_end = time.time()
      rospy.logdebug("Batch {} training takes: {:.4f}".format(i, batch_end-batch_start))
  rst_end = time.time()
  print("Random samples training end! It took {:.4f} seconds".format(rst_end-rst_start))

  # random shoot control with new samples
  reward_storage = []
  for episode in range(num_episodes):
    obs, info = env.reset()
    state = obs_to_state(obs, info)
    goal = info["goal_position"]
    total_reward = 0
    done = False
    # compute control policies as long as sampling more
    for step in range(num_steps):
      action_sequences = utils.generate_action_sequence(
        num_sequences,
        len_horizon,
        num_actions
      )
      action = utils.shoot_action(
        model,
        action_sequences,
        state,
        goal
      )
      next_state, reward, done, info = env.step(action)
      next_state = next_state.astype(np.float32)
      stac = np.concatenate((state, np.array([action]))).astype(np.float32)
      stacs_memory.append(stac)
      nextstates_memory.append(next_state)
      total_reward += reward
      state = next_state
      print("Current position: {}, Goal position: {}, Reward: {:.4f}".format(
        info["current_position"],
        info["goal_position"],
        reward
      ))
      if done:
        break
    reward_storage.append(total_reward)
    # Train with samples at the end of every episode
    dataset = utils.create_dataset(
      np.array(stacs_memory),
      np.array(nextstates_memory),
      batch_size=batch_size,
      num_epochs=4
    )
    ep_start = time.time()
    for i, (x, y) in enumerate(dataset):
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
        zip(grads, model.variables),
        global_step
      )
      print("Batch: {:04d}, Loss: {:.4f}".format(i, loss_value))
    ep_end=time.time()
    print("Episode {:04d} training takes {:.4f}".format(episode, ep_end-ep_start))

  #   main_end = time.time()
  #   print(
  #     "{:d} Random Samples was trained {:d} epochs",
  #     "\n{:d} Controlled Samples was trained {:d} epochs",
  #     "\nTotal execution time: {:.4f}".format(
  #       num_epochs*num_iters,
  #       num_epochs/4,
  #       num_episodes*num_steps,
  #       num_episodes*4,
  #       main_end-main_start
  #     )
  #   )
