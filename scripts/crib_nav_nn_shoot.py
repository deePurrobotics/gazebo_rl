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

class Model:
  def __init__(self, num_states, num_actions, batch_size):
    self._num_inputs = num_states + num_actions
    self._batch_size = batch_size
    # define the placeholders
    self._sa_pairs = None
    # the output operations
    self._logits = None
    self._optimizer = None
    self._var_init = None
    # now setup the model
    self._define_model()

  def _define_model(self):
    self._states = tf.placeholder(shape=[None, self._num_inputs], dtype=tf.float32)
    self._new_states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
    # create a couple of fully connected hidden layers
    fc1 = tf.layers.dense(self._states, 512, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
    self._logits = tf.layers.dense(fc2, self._num_states)
    loss = tf.losses.mean_squared_error(self._new_states, self._logits)
    self._optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    self._var_init = tf.global_variables_initializer()

  def predict_one(self, stac_pair, sess):
    return sess.run(
      self._logits,
      feed_dict={self._sa_pairs: sa_pair.reshape(1, self._num_inputs)}
    )
  
  def predict_batch(self, states, sess):
    return sess.run(self._logits, feed_dict={self._states: states})

  def train_batch(self, sess, x_batch, y_batch):
    sess.run(self._optimizer, feed_dict={self._states: x_batch, self._new_states: y_batch})

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

class ModelBasedController():
  def __init__(self, sess, model):
    self._sess = sess
    self._model = model

  def train(self, memory_batch):
    x_batch = memory_batch[0]
    y_batch = memory_batch[-1]
    self._model.train_batch(self._sess, x_batch, y_batch)

    
  def shoot_action(self, state, goal, num_sequences, horizon):
    action_sequences = utils.generate_action_sequence(
      num_sequences,
      horizon,
      num_actions
    ) # (s,h,a) array
    sequence_rewards = np.zeros(num_sequences)
    # Compute reward for every sequence 
    for s in range(action_sequences.shape[0]):
      reward_in_horizon = 0
      for h in range(horizon):
        stac_pair = np.concatenate((state, action_sequences[s,h]))
        new_state = self._model.predict_one(stac_pair, self._sess)
        if np.linalg.norm(state[:2]-goal) <= 0.2:
          reward = 1
        else:
          reward = 0
        reward_in_horizon += reward
      sequence_rewards[s] = reward_in_horizon

    best_seq_id = np.argmax(sequence_rewards)
    optimal_action = action_sequences[best_seq_id,0] # take first action of each sequence

    return optimal_action


if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", anonymous=True, log_level=rospy.INFO)
  env_name = "TurtlebotCrib-v0"
  env = gym.make(env_name)
  rospy.loginfo("Gazebo gym environment set")
  # Set parameters
  num_episodes = 10
  num_steps = 10
  horizon = 10 # number of time steps the controller considers
  batch_size =  16
  nn_model = Model()
  memory = Memory(50000)
  agent = ModelBasedController()
  # Sample a bunch of random moves
  state = env.reset()
  for i in range(num_steps):
    action = agent.random_action()
    next_state, reward, done, info = self._env.step(action)
    memory.add_sample((state, action, reward, next_state))

  reward_storage = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(num_episodes):
      state, info = env.reset()
      goal = info["goal"]
      total_reward = 0
      done = False
      train_samples = memory.sample(num_samples=batch_size)
      agent.train(train_samples)
      for st in range(num_steps):
        action = agent.shoot_action(state, goal, num_sequences, horizon)
        next_state, reward, done, info = env.step(action)
        memory.add_sample((state, action, reward, next_state))
        total_reward += reward
        state = next_state
        rospy.loginfo("Total reward = {}".format(total_reward))
        if done:
          break
      reward_storage.append(total_reward)

    sess.close()

  plt.plot(reward_list)
