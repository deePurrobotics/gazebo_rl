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


def obs2state(observation, low):
  """
  Helper function convert observation to discrete state
  We only use x and y to represent state, so the first 2 element in observation
  Args:
    observation: 0-d numpy array e.g. (-1.2, 3.3)
    low: lower bound of observation
  Return:
    state: a scalar
  """
  x = observation[0]
  y = observation[1]
  ind_x = int(x - low[0]) # index of x
  ind_y = int(y - low[1])
  state = ind_x*10 + ind_y

  return state

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
