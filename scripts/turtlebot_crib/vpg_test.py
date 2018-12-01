# #! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

import envs.crib_nav_task_env
from utils import bcolors, obs_to_state


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

# specify sta / act dimensions
sta_dim = 6 # due to obs_to_state
n_acts = 2
# make core of policy network
sta_ph = tf.placeholder(shape=(None, sta_dim), dtype=tf.float32)
logits = mlp(sta_ph, sizes=hidden_sizes+[n_acts])
# make action selection op (outputs int actions, sampled from policy)
actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
# make loss function whose gradient, for the right data, is policy gradient
weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
action_masks = tf.one_hot(act_ph, n_acts)
log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
loss = -tf.reduce_mean(weights_ph * log_probs)
# make train op
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

