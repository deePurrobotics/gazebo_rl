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

def generate_action_sequence(num_sequences, horizon, num_actions):
  """ Generate S random action sequences with H horizon
  """
  action_sequences = np.zeros((num_sequences, horizon, num_actions))
  rand_i = random.randrange(num_actions)
  for s in range(num_sequences):
    for h in range(horizon):
      action_sequences[s,h,rand_i] = 1

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

def create_dataset(input_features, output_labels, shuffle=True, num_epochs=None, batch_size):
  """ Create TF dataset from numpy arrays
  """
  dataset = tf.data.Dataset.from_tensor_slices((input_features, output_labels))
  if shuffle:
    dataset = dataset.shuffle(buffer_size = 1000)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)

  return dataset
