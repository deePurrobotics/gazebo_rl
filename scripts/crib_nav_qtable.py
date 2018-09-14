#! /usr/bin/env python

"""

Q-Learning example using turtlebot crib environment
Navigate towards preset goal

Author: LinZHanK (linzhank@gmail.com)

Inspired by: https://gist.github.com/malzantot/9d1d3fa4fdc4a101bc48a135d8f9a289

"""
import gym
from gym import wrappers
import numpy as np

import openai_ros_envs.crib_task_env

# Parameters
n_states = 10
iter_max = 1000

initial_lr = 1.0 # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 1000
eps = 0.02

def obs_to_state(env, obs):
  """ Maps an observation to state """
  env_low = env.observation_space.low
  env_high = env.observation_space.high
  env_dx = (env_high - env_low) / n_states
  a = int((obs[0] - env_low[0])/env_dx[0])
  b = int((obs[1] - env_low[1])/env_dx[1])
  return a, b

if __name__ == "__main__":
  rospy.init_node("turtlebot2_crib_qlearn", annonymous=True, log_level=rospy.INFO)
  env_name = 'TurtleBotCrib-v0'
  env = gym.make(env_name)
  env.seed(0)
  rospy.loginfo("Gazebo gym environment set")
  # np.random.seed(0) 
  rospy.loginfo("----- using Q Learning -----")
  q_table = np.zeros((n_states, n_states, 4))
  for i in range(iter_max):
    obs = env.reset()
    total_reward = 0
    ## eta: learning rate is decreased at each step
    eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
    for j in range(t_max):
      a, b = obs_to_state(env, obs)
      if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
      else:
        logits = q_table[a][b]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        action = np.random.choice(env.action_space.n, p=probs)
      obs, reward, done, _ = env.step(action)
      total_reward += reward
      # update q table
      a_, b_ = obs_to_state(env, obs)
      q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
      if done:
        break
      if i % 100 == 0:
        print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, solution_policy, True)
