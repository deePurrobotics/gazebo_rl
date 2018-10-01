from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random

import openai_ros_envs.crib_task_env

rospy.init_node("dataset_test", anonymous=True, log_level=rospy.DEBUG)
env_name = "TurtlebotCrib-v0"
env = gym.make(env_name)
rospy.loginfo("Gazebo gym environment set")
# Set parameters
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
num_episodes = 64
num_steps = 128
num_sequences = 100
horizon = 10 # number of time steps the controller considers
batch_size = 64
mem = {"pre_state": [], "action": [], "reward": [], "new_state": []}

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Sample a bunch of random moves
  rospy.logdebug("Initial random sampling start...")
  for i in range(2):
    state, info = env.reset()
    for j in range(10):
      if j%100 == 0:
        rospy.logdebug("Initial random sampling in ep.{}, step{}".format(i+1,j+1))
      action = random.randrange(num_actions)
      next_state, reward, done, info = env.step(action)
      mem["pre_state"].append(state)
      mem["action"].append(action)
      mem["reward"].append(reward)
      mem["new_state"].append(next_state)
      
      state = next_state
    rospy.logdebug("Initial random sampling finished.")

  features = np.concatenate((np.array(mem["pre_state"]), np.array([mem["action"]]).T), axis=1)
  labels = np.array(mem["new_state"])
  dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  
