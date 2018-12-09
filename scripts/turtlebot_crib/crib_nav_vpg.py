#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment

Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

import envs.crib_nav_task_env
from utils import bcolors, obs_to_state


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(env_name='CribNav-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):
  # make environment, check spaces
  env = gym.make(env_name)
  # specify sta / act dimensions
  sta_dim = env.observation_space.shape[0]-1 # due to obs_to_state
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

  # for training policy
  def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = [] # for observations
    batch_sta = [] # for states
    batch_acts = [] # for actions
    batch_weights = [] # for R(tau) weighting in policy gradient
    batch_rets = [] # for measuring episode returns
    batch_lens = [] # for measuring episode lengths
    # reset episode-specific variables
    obs, info = env.reset() # first obs comes from starting distribution
    sta = obs_to_state(obs, info)
    done = False # signal from environment that episode is over
    ep_rews = [] # list for rewards accrued throughout ep
    # collect experience by acting in the environment with current policy
    while True:
      # save obs
      batch_sta.append(sta.copy())
      # act in the environment
      action = sess.run(actions, feed_dict={sta_ph: sta.reshape(1,-1)})[0]
      if not action:
        act = np.array([env.action_space.high[0], env.action_space.low[1]]) # id=0 => [high_lin, low_ang]
      else:
        act = env.action_space.high # id=1 => [high_lin, high_ang]
      print(bcolors.WARNING, "action: {}".format(act), bcolors.ENDC)
      obs, rew, done, info = env.step(act)
      sta = obs_to_state(obs, info)
      # save action, reward
      batch_acts.append(action)
      ep_rews.append(rew)
      if done:
        # if episode is over, record info about episode
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_rets.append(ep_ret)
        batch_lens.append(ep_len)
        # the weight for each logprob(a|s) is R(tau)
        batch_weights += [ep_ret] * ep_len
        # reset episode-specific variables
        obs, info= env.reset()
        # end experience loop if we have enough of it
        if len(batch_sta) > batch_size:
          break

    # take a single policy gradient update step
    batch_loss, _ = sess.run(
      [loss, train_op],
      feed_dict={
        sta_ph: np.array(batch_sta),
        act_ph: np.array(batch_acts),
        weights_ph: np.array(batch_weights)
      }
    )
    return batch_loss, batch_rets, batch_lens

  # training loop
  for i in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print(
      "epoch: {:3d}\t loss: {:.3f}\t return: {:.3f}\t ep_len: {:.3f}".format(
        i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)
      )
    )
    
if __name__ == "__main__":
  rospy.init_node("crib_nav_vpg", anonymous=True, log_level=rospy.WARN)
  train()
