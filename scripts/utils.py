import numpy as np

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
