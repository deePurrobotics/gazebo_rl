from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import datetime

import envs.crib_nav_task_env

tf.enable_eager_execution()

# init node
rospy.init_node("load_model_test", anonymous=True, log_level=rospy.WARN)
# create env
env_name = "CribNav-v0"
env = gym.make(env_name)
rospy.loginfo("Gazebo gym environment set")
# set parameters
action_space_dim = env.action_space.shape[0]
state_space_dim = env.observation_space.shape[0] + 4 # add cos and sin of vector from bot to goal
num_episodes = 128
num_steps = 256
# load model from save checkpoint
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation=tf.nn.relu,input_shape=(state_space_dim+action_space_dim,)),  # input shape required
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(state_space_dim)
  ]
)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
model_dir = "/home/linzhank/ros_ws/src/turtlebot_rl/scripts/turtlebot/crib_nav/checkpoint"
model_date = "20181025"
checkpoint_dir = os.path.join(model_dir, model_date)
root = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model,
    optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

sas = np.zeros((8,state_space_dim+action_space_dim))
s = np.random.randn(state_space_dim)
for i in range(sas.shape[0]):
  a = env.action_space.sample()
  sas[i] = np.concatenate((s,a)).astype(np.float32)

nss = model(sas)
