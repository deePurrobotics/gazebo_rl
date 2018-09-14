import numpy as np

def ob2state(observation):
  """
  Helper function convert observation to discrete state
  Args:
    observation: 0-d numpy array (-1.2, 3.3)
  Return:
    state: string ("-23")
  """
  intermediate = np.floor(observation).astype(int) # intermediate state
  state = "".join(map(str, intermediate))
