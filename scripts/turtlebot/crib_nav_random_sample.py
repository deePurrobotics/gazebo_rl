#! /usr/bin/env python

"""
Turtlebot random sampling in crib nav task. 

Model input: state-action pairs
Model output: next state

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
    obs: [x, y, v_x, v_y, cos(yaw), sin(yaw), yaw_dot]
        theta= robot orientation, alpha= angle between r->g and x-axis
    info: {"goal_position", ...}
  Returns:
    state: [x, y, v_x, v_y, cos(yaw), sin(yaw), yaw_dot, dx, dy, cos(alpha), sin(alpha)]
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
  num_states = env.observation_space.shape[0] + 4 # add cos and sin of vector from bot to goal
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
  sample_size = int(memory_size / 10)
  batch_size = sample_size
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
  num_epochs = 1024
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
  model_dir = "/home/linzhank/ros_ws/src/gazebo_rl/scripts/turtlebot/crib_nav/checkpoint"
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
  metrics = tfe.metrics.Mean()
  rst_start = time.time()
  for i, (x,y) in enumerate(dataset):
    # optimize model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(
      zip(grads, model.variables),
      global_step
    )
    # track progress
    metrics(loss_value) 
    # log training
    if not i % 100:
      print("Iteration: {}, Loss: {:.3f}".format(i, metrics.result()))
  rst_end = time.time()
  print("Random samples training end! It took {:.4f} seconds".format(rst_end-rst_start))
