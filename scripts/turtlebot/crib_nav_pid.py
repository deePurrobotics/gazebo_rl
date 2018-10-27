#! /usr/bin/env python

"""
PID control for crib nav task.

Author: LinZHanK (linzhank@gmail.com)

"""
from __future__ import absolute_import, division, print_function

import numpy as np
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

import pdb

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

def goal_reached(err_lin):
  if err_lin <= 0.1:
    return True
  else:
    return False


if __name__ == "__main__":
  # init node
  rospy.init_node("crib_nav_pid", anonymous=True, log_level=rospy.WARN)
  # create env
  env_name = "CribNav-v0"
  env = gym.make(env_name)
  rospy.logwarn("CribNav-v0 environment set")
  # PID parameters
  kp_lin = 5
  kd_lin = 0.5
  kp_ang = 10
  kd_ang = 0.1
  # start control
  obs, info = env.reset()
  done = False
  state = obs_to_state(obs, info)
  err_lin = np.linalg.norm(state[7:9])
  del_err_lin = 0
  err_ang = np.arctan2(state[-1], state[-2]) - np.arctan2(state[5], state[4])
  if err_ang > np.pi:
    err_ang -= np.pi*2
  elif err_ang < -np.pi:
    err_ang += np.pi*2
  del_err_ang = 0
  step = 0
  for step in range(128):
    # compute cmd_vel.linear
    v_lin = kp_lin*err_lin + kd_lin*del_err_lin
    if v_lin > env.action_space.high[0]:
      v_lin = env.action_space.high[0]
    elif v_lin < env.action_space.low[0]:
      v_lin = env.action_space.low[0]
    v_ang = kp_ang*err_ang + kd_ang*del_err_ang
    if v_ang > env.action_space.high[1]:
      v_ang = env.action_space.high[1]
    elif v_ang < env.action_space.low[1]:
      v_ang = env.action_space.low[1]
    action = np.array([v_lin,v_ang])
    # implement action 
    obs, _, done, info = env.step(action)
    new_state = obs_to_state(obs, info)
    print(
      bcolors.OKBLUE, "Step: {}".format(step+1), bcolors.ENDC,
      "\ncurrent_position: {}".format(state[:2]),
      "\ngoal_position: {}".format(info["goal_position"]),
      "\naction: {}".format(action)
    )
    # compute linear error
    new_err_lin = np.linalg.norm(new_state[7:9])
    del_err_lin = new_err_lin - err_lin
    err_lin = new_err_lin
    # compute angular error
    new_err_ang = np.arctan2(new_state[-1], new_state[-2]) - \
                  np.arctan2(new_state[5], new_state[4])
    if new_err_ang > np.pi:
      new_err_ang -= np.pi*2
    elif new_err_ang < -np.pi:
      new_err_ang += np.pi*2
    del_err_ang = new_err_ang - err_ang
    err_ang = new_err_ang
    print(
      bcolors.OKGREEN,
      "linear error: {}, angular error: {}".format(err_lin, err_ang),
      bcolors.ENDC
    )

    if goal_reached(err_lin):
      print(bcolors.WARNING,"===\n Goal reached!!!\n ===", bcolors.ENDC)
      break
    state = new_state

  if not goal_reached(err_lin):
    print(bcolors.FAIL, "~~~Not able to reach the goal~~~", bcolors.ENDC)
    
