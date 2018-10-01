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
import utils

def model(graph, input_tensor, num_outputs):
  """Create the model which consists of
  a bidirectional rnn (GRU(10)) followed by a dense classifier

  Args:
    graph (tf.Graph): Tensors' graph
    input_tensor (tf.Tensor): Tensor fed as input to the model

  Returns:
    tf.Tensor: the model's output layer Tensor
  """
  cell = tf.nn.rnn_cell.GRUCell(10)
  with graph.as_default():
    fc1 = tf.layers.dense(input_tensor, 32, activation=tf.nn.relu)
    fc2 = tf.layers.dense(input_tensor, 16, activation=tf.nn.relu)
    logits = tf.layers.dense(fc2, num_outputs)

    return logits

def optimize_op(graph, logits, labels_tensor):
  """Create optimization operation from model's logits and labels

  Args:
    graph (tf.Graph): Tensors' graph
    logits (tf.Tensor): The model's output without activation
    labels_tensor (tf.Tensor): Target labels

  Returns:
    tf.Operation: the operation performing a stem of Adam optimizer
  """
  with graph.as_default():
    with tf.variable_scope('loss'):
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=logits,
          labels=labels_tensor,
          name='xent'
        ),
        name="mean-xent"
      )
    with tf.variable_scope('optimizer'):
      opt_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

    return opt_op
    
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
  num_episodes = 64
  num_steps = 128
  num_sequences = 100
  horizon = 10 # number of time steps the controller considers
  batch_size = 128
  
  # random samples features and labels
  features = np.zeros((65536, 8))
  labels = np.zeros((65536, 7))
  i = 0
  for _ in range(2048):
    state, _ = env.reset()
    for _ in range(32):
      action = random.randrange(num_actions)
      next_state, reward, done, _ = env.step(action)
      features[i] = np.concatenate((state, np.array([action])))
      labels[i] = next_state
      state = next_state
  # create graph
  graph = tf.Graph()
  with graph.as_default():
    # define placeholders
    batch_size_ph = tf.placeholder(dtype=tf.int64, name="batch_size_ph")
    features_data_ph = tf.placeholder(
      dtype=tf.float32,
      shape=[None, 8],
      name="features_data_ph"
    )
    labels_data_ph = tf.placeholder(
      dtype=tf.float32,
      shape=[None, 7],
      name='labels_data_ph'
    )
    # dataset
    dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph))
    dataset = dataset.batch(batch_size_ph)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
    input_tensor, labels_tensor = iterator.get_next()
    # nn-model and optimizer_op
    logits = model(graph, input_tensor, num_states)
    opt_op = optimize_op(graph, logits, labels_tensor)

    with tf.Session(graph=graph) as sess:
      # initialize variables
      tf.global_variables_initializer().run(session=sess)
      # train on random samples
      for epoch in range(64):
        batch = 0
        # initialize dataset
        sess.run(
          dataset_init_op,
          feed_dict={
            features_data_ph: features,
            labels_data_ph: labels,
            batch_size: batch_size
          }
        )
        value = []
