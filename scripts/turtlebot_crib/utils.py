from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import random

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

def obs_to_state(obs, info):
  """
  This function converts observation into state
  Args: 
    obs: [x, y, v_x, v_y, cos(theta), sin(theta), theta_dot]
        theta= robot orientation, alpha= angle between r->g and x-axis
    info: {"goal_position", ...}
  Returns:
    state: [r_norm, p_norm, alpha, alpha_dot, beta, beta_dot]
      r_norm: distance from map origin to robot
      p_norm: distance from robot to goal
      alpha: angle from map's x to r
      beta: angle from robot's x to p
      *_dot: angular velocity
  """
  # compute states
  r = obs[:2]
  p = info["goal_position"] - obs[:2]
  r_norm = np.linalg.norm(r) # sqrt(x^2+y^2)
  p_norm = np.linalg.norm(p)
  alpha = np.arctan2(obs[1], obs[0])
  alpha_dot = np.arctan2(obs[3], obs[2])
  # comput phi: angle from map's x_axis to p  
  x_axis = np.array([1, 0])
  y_axis = np.array([0, 1])
  cos_phi = np.dot(p, x_axis) / (np.linalg.norm(p)*np.linalg.norm(x_axis))
  sin_phi = np.dot(p, y_axis) / (np.linalg.norm(p)*np.linalg.norm(y_axis))
  phi = np.arctan2(sin_phi, cos_phi)
  # compute beta in [-pi, pi]
  beta = phi - np.arctan2(obs[-2], obs[-3])
  if beta > np.pi:
    beta -= 2*np.pi
  elif beta < -np.pi:
    beta += 2*np.pi
  beta_dot = obs[-1]
  state = np.array([r_norm, p_norm, alpha, alpha_dot, beta, beta_dot]).astype(np.float32)

  return state

def discretize_state(state, boxes):
  """
  Converts continuous state into discrete states
  Args: 
    state:
    boxes:
  Returns:
    index: state index in Q table, represent in tuple
  """
  # match state into box
  index = []
  for i_s, st in enumerate(state):
    for i_b, box in enumerate(boxes[i_s]):
      if st >= box[0] and st <= box[1]:
        index.append(i_b)
        break
  assert len(index) == 6
  
  return tuple(index)

def generate_action_sequence(num_sequences, len_horizon, num_actions):
  """ Generate S random action sequences with H horizon
  """
  action_sequences = np.zeros((num_sequences, len_horizon))
  for s in range(num_sequences):
    for h in range(len_horizon):
      action_sequences[s,h] = random.randrange(num_actions)

  return action_sequences

def sample_to_batch(samples_list, num_states, num_actions):
  """ Create training batch from sampled memory
  """
  x_batch = np.zeros((len(samples_list), num_states+num_actions))
  y_batch = np.zeros((len(samples_list), num_states))
  for i, s in enumerate(samples_list):
    onehot_action = np.zeros(num_actions)
    onehot_action[s[1]] = 1
    x_batch[i] = np.concatenate((s[0], onehot_action))
    y_batch[i] = s[-1]

  return x_batch, y_batch

def create_dataset(input_features, output_labels, batch_size, shuffle=True, num_epochs=None):
  """ Create TF dataset from numpy arrays
  """
  dataset = tf.data.Dataset.from_tensor_slices((input_features, output_labels))
  if shuffle:
    dataset = dataset.shuffle(buffer_size = 1000)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)

  return dataset
  
def shoot_action(model, action_sequences, state, goal):
  """ Find an action with most reward using random shoot
  """
  sequence_rewards = np.zeros(action_sequences.shape[0])
  # Compute reward for every sequence 
  for seq in range(action_sequences.shape[0]):
    old_state = np.array(state).reshape(1,-1).astype(np.float32)
    # print("old_state: {}".format(old_state)) # debug
    reward_in_horizon = 0
    for hor in range(action_sequences.shape[1]):
      action = np.array([[action_sequences[seq,hor]]])
      stac = np.concatenate((old_state, action), axis=1).astype(np.float32) # state-action pair
      new_state = model(stac)
      # print("new_state: {}".format(new_state)) # debug
      if np.linalg.norm(new_state[0,:2]-goal) < np.linalg.norm(old_state[0,:2]-goal):
        reward = 1
      else:
        reward = 0
      reward_in_horizon += reward
      old_state = new_state
      sequence_rewards[seq] = reward_in_horizon

    idx = np.argmax(sequence_rewards) # action sequence index with max reward
    optimal_action = int(action_sequences[idx,0]) # take first action of each sequence

    return optimal_action

def greedy_action(model, num_actions, state, goal):
  """ Find an action with most reward
  """
  action_list = range(num_actions)
  reward = 0
  reward_list = []
  old_state = np.array(state).reshape(1,-1).astype(np.float32)
  for action in action_list:
    stac = np.concatenate((old_state, np.array([[action]])), axis=1).astype(np.float32) # state-action pair
    new_state = model(stac)
    if np.linalg.norm(new_state[0,:2]-goal) < np.linalg.norm(old_state[0,:2]-goal):
      reward = 1
    else:
      reward = 0
    reward_list.append(reward)
    old_state = new_state

  optimal_action = np.argmax(reward_list) 

  return optimal_action

    
