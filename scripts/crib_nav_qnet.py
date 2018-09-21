#! /usr/bin/env python

"""

QNet example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""
from __future__ import print_function

import gym
from gym import wrappers
import tensorflow as tf
import rospy
import numpy as np
import matplotlib.pyplot as plt
import utils

import openai_ros_envs.crib_task_env

# Enable font colors
class bcolors:
  """ For the purpose of print in terminal with colors """
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", anonymous=True, log_level=rospy.INFO)
  env_name = 'TurtlebotCrib-v0'
  env = gym.make(env_name)
  # env.seed(0)
  rospy.loginfo("Gazebo gym environment set")
  # np.random.seed(0) 
  rospy.loginfo("----- using Q Learning -----")
  # Load parameters
  num_states = 7
  num_actions = 4
  epsilon = get_explore_rate(0) # explore_rate
  Alpha = 1. # learning rate
  Gamma = 0.95 # reward discount
  num_episodes = 2000
  num_steps = 500
  low = env.observation_space.low

  tf.reset_default_graph()
  #These lines establish the feed-forward part of the network used to choose actions
  input_node = tf.placeholder(dtype=tf.float32, shape=(1, num_states)) # states
  output_node = tf.placeholder(dtype=tf.float32, shape=(1, num_actions)) # actions
  w1 = tf.Variable(tf.truncated_normal(
    shape=[num_states, 128],
    stddev = 0.01,
    dtype=tf.float32
  ))
  b1 = tf.Variable(tf.constant(
    value=0.01,
    shape=[128],
    dtype=tf.float32
  ))
  w2 = tf.Variable(tf.truncated_normal(
      shape=[128, num_actions],
    stddev=0.01,
    dtype=tf.float32
  ))
  b2 = tf.Variable(tf.constant(
    value=0.01,
    shape=[num_actions],
    dtype=tf.float32
  ))
  def model(input_states):
    hidden = tf.nn.relu(tf.matmul(input_states, w1) + b1)
    return tf.matmul(hidden, w2) + b2

  q_values = model(input_node)
  prediction = tf.argmax(q_values, axis=1)
  loss = tf.reduce_sum(tf.square(output_node - q_values))
  regularizers = (tf.nn.l2_loss(w1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(b2))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(num_episodes):
      # Reset env and get first observation
      state = env.reset()
      total_reward = 0
      done = False
      for st in range(num_steps):
        epsilon = get_explore_rate(ep)
        rospy.loginfo(bcolors.OKBLUE, "Epoch {0:d}, Step {1:d}".format(ep, st), bcolors.ENDC)
        # Choose action greedily
        q, argmax_q = sess.run(
          [q_values, prediction],
          feed_dict={states_node:state.reshape(1,num_states)}
        )
        action = argmax_q[0]
        if np.random.rand(1)[0] < epsilon:
          action = random.randrange(0,NUM_ACTIONS)
          rospy.loginfo(bcolors.WARNING, "!!! Action selected randomly !!!", bcolors.ENDC)

        # Get new state and reward
        state1, reward, done, _ = env.step(action)
        # Update Q table
        q_next = sess.run(
          q_values,
          feed_dict={states_node:np.array(state1).reshape(1,num_states)}
        )
        max_q_net = np.max(q_next)
        target_q = q
        target_q[0, action] = reward + gamma * max_q_next
        total_reward += reward
        # train network using target_q and q, remember loss = square(target_q-q)
        opt = sess.run(optimizer,
          feed_dict={states_node:np.array(state).reshape(1,num_states),
          qvalue_node:target_q})
        total_reward += reward
        state = state1
        rospy.loginfo("Total reward = {}".format(total_reward))
        if done:
          break

    
    
def get_explore_rate(episode):
  return max(0.05, min(1, 1.0-math.log10((episode+1)/25.)))
