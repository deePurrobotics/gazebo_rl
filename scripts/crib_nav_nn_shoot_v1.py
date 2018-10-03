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
import matplotlib.pyplot as plt

import openai_ros_envs.crib_task_env
import utils

tf.enable_eager_execution()


class DenseModel(tf.keras.Model):
  def __init__(self):
    super(DenseModel, self).__init__()
    self._input_shape = []
    self.dense1 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)
    self.dense3 = tf.keras.layers.Dense(units=7)

  def call(self, input):
    """Run the model."""
    fc1 = self.dense1(input)
    fc2 = self.dense2(result)
    output = self.dense2(result)  # reuse variables from dense2 layer
    return output
  
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.mean_squared_error(labels=y, predictions=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == "__main__":
  # init node
  rospy.init_node("crib_nav_mpc", anonymous=True, log_level=rospy.DEBUG)
  # create env
  env_name = "TurtlebotCrib-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # set parameters
  num_actions = env.action_space.n
  num_states = env.observation_space.shape[0]
  num_episodes = 4
  num_steps = 8
  num_sequences = 100
  horizon = 10 # number of time steps the controller considers
  batch_size = 64
  
  stacs_memory = []
  nextstates_memory = []
  # setup model
  # model = Model()
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(num_states+1,)),  # input shape required
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_states)
  ])
  # set training parameters
  num_epochs = 4

  # Random Sampling, 8 samples
  for _ in range(4):
    state, _ = env.reset()
    for _ in range(8):
      action = random.randrange(num_actions)
      next_state, _, _, _ = env.step(action)
      stac = np.concatenate((state, np.array([action]))).astype(np.float32)
      stacs_memory.append(stac)
      nextstates_memory.append(next_state.astype(np.float32))
      state = next_state
  # Train random sampled dataset
  dataset = utils.create_dataset(
    input_features=np.array(stacs_memory),
    output_labels=np.array(nextstates_memory),
    batch_size=4,
    num_epochs=num_epochs
  )
  # for epoch in range(num_epochs):
  #   for i, (x, y) in enumerate(dataset):
  #     print("epoch: {:03d}, iter: {:03d}".format(epoch, i))
  #     print()
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  global_step = tf.train.get_or_create_global_step()
  loss_value, grads = grad(
    model,
    np.array(stacs_memory),
    np.array(nextstates_memory)
  )
  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    for i, (x,y) in enumerate(dataset):
      # optimize model
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(
      zip(grads, model.variables),
        global_step
      )
      # track progress
      epoch_loss_avg(loss_value)  # add current batch loss
      # log training
      print("Epoch {:03d}: Iteration: {:03d}, Loss: {:.3f}".format(epoch, i, epoch_loss_avg.result()))


  # # Control with more samples
  # for episode in range(num_episodes):
  #   state, info = env.reset()
  #   goal = info["goal_position"]
  #   total_reward = 0
  #   done = False
  #   dataset = utils.create_dataset(
  #     np.array(stacs_memory),
  #     np.array(nextstates_memory),
  #     batch_size
  #   )
  #   # train one epoch with newly collected samples
  #   for x, y in dataset:
  #     loss_value, grads = grad(model, x, y)
  #     optimizer.apply_gradients(
  #       zip(grads, model.variables),
  #       global_step
  #     )
  #     print("Loss: {:3f}".format(loss_value))
  #   # compute control policies as long as sampling more
  #   for step in range(num_steps):
  #     action = utils.shoot_action(model, state, goal, num_sequences, horizon)
  #     next_state, reward, done, info = env.step(action)
  #     stac = np.concatenate((state, np.array([action])))
  #     stacs_memory.append(stac)
  #     nextstates_memory.append(next_state)
  #     total_reward += reward
  #     state = next_state
  #     print("Total reward: {:.4f}".format(total_reward))
  #     if done:
  #       break
  #   reward_storage.append(total_reward)

