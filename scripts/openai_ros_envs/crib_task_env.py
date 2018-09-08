import rospy
import numpy as np
import time
from gym import spaces
import turtlebot_env
from gym.envs.registration import register

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
  id='TurtleBotCrib-v0',
  entry_point='openai_ros_envs.crib_task_env:TurtleBotCribEnv',
  timestep_limit=timestep_limit_per_episode,
)

class TurtleBotCribEnv(turtlebot_env.TurtleBotEnv):
  def __init__(self):
    """
    This Task Env is designed for having the TurtleBot2 in a walled world.
    It will learn how to move toward designated point.
    """
        
    # Only variable needed to be set here
    number_actions = rospy.get_param('/turtlebot2/n_actions')
    self.action_space = spaces.Discrete(number_actions)
        
    # We set the reward range, which is not compulsory but here we do it.
    self.reward_range = (-np.inf, np.inf)
        
        
    #number_observations = rospy.get_param('/turtlebot2/n_observations')
    """
    We set the Observation space for the 6 observations
    cube_observations = [
      x,
      y,
      v_x,
      v_y,
      cos(yaw),
      sin(yaw),
      w_yaw
    ]
    """
        
    # Actions and Observations
    self.linear_speed = rospy.get_param('/turtlebot2/linear_speed')
    self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
    self.init_linear_speed = rospy.get_param('/turtlebot2/init_linear_speed')
    self.init_angular_speed = rospy.get_param('/turtlebot2/init_angular_speed')
    self.min_range = rospy.get_param('/turtlebot2/min_range')
    self.obs_high = rospy.get_param('/turtlebot2/obs_high')
    self.obs_low = rospy.get_param('/turtlebot2/obs_low')

    self.new_ranges = rospy.get_param('/turtlebot2/new_ranges')
    self.min_range = rospy.get_param('/turtlebot2/min_range')
    self.max_laser_value = rospy.get_param('/turtlebot2/max_laser_value')
    self.min_laser_value = rospy.get_param('/turtlebot2/min_laser_value')
    # We create two arrays based on the binary values that will be assigned
    # In the discretization method.
    high = np.array(self.obs_high) # upper bound of observations
    low = np.array(self.obs_low) # lower bound of observations 
    # We only use two integers
    self.observation_space = spaces.Box(low, high)
        
    rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
    rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
    # Rewards
    self.reward = rospy.get_param("/turtlebot2/reward")
    self.cumulated_steps = 0.0

    # Here we will add any init functions prior to starting the MyRobotEnv
    super(TurtleBotCribEnv, self).__init__()

  def _set_init_pose(self):
    """Sets the Robot in its init pose
    """
    self.move_base(
      self.init_linear_speed,
      self.init_angular_speed,
      epsilon=0.05,
      update_rate=10,
      min_laser_distance=-1
    )

    return True


  def _init_env_variables(self):
    """
    Inits variables needed to be initialised each time we reset at the start
    of an episode.
    :return:
    """
    # For Info Purposes
    self.cumulated_reward = 0.0
    # Set to false Done, because its calculated asyncronously
    self._episode_done = False
        
    # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
    # and sometimes still have values from the prior position that triguered the done.
    time.sleep(0.2)


  def _set_action(self, action):
    """
    This set action will Set the linear and angular speed of the turtlebot2
    based on the action number given.
    :param action: The action integer that set s what movement to do next.
    """
    rospy.logdebug("Start Set Action ==>"+str(action))
    # We convert the actions to speed movements to send to the parent class turtlebot_env
    lin = self.linear_speed # for the sake of create speed pool
    ang = self.angular_speed
    lin_spd_pool = [-2*lin, -lin, lin, 2*lin]
    ang_spd_pool = [-2*ang, -ang, ang, 2*ang]
    i_lin = action/len(ang_spd_pool)
    i_ang = action%len(ang_spd_pool) 
    # We tell TurtleBot2 the linear and angular speed to set to execute
    self.move_base(
      linear_speed=lin_spd_pool[i_lin],
      angular_speed=ang_spd_pool[i_ang],
      epsilon=0.05,
      update_rate=10,
      min_laser_distance=self.min_range
    )
    rospy.logdebug("END Set Action ==>"+str(action))

  def _get_obs(self):
    """
    Here we define what sensor data defines our robots observations
    To know which Variables we have acces to, we need to read the
    TurtleBot2Env API DOCS
    :return:
    """
    rospy.logdebug("Start Get Observation ==>")
    # We get the laser scan data
    laser_scan = self.get_laser_scan()

    discretized_observations = self.discretize_observation(
      laser_scan,
      self.new_ranges
    )

    rospy.logdebug("Observations==>"+str(discretized_observations))
    rospy.logdebug("END Get Observation ==>")
    return discretized_observations
        

  def _is_done(self, observations):
    if self._episode_done:
      rospy.logerr("TurtleBot2 is Too Close to wall==>")
    else:
      rospy.logerr("TurtleBot2 is Ok ==>")

    return self._episode_done

  def _compute_reward(self, observations, done):
    if not done:
      reward = -1
    else:
      reward = 0


    rospy.logdebug("reward=" + str(reward))
    self.cumulated_reward += reward
    rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
    self.cumulated_steps += 1
    rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
    
    return reward


    # Internal TaskEnv Methods
    
  def discretize_observation(self,data,new_ranges):
    """
    Discards all the laser readings that are not multiple in index of new_ranges
    value.
    """
    self._episode_done = False
        
    discretized_ranges = []
    mod = len(data.ranges)/new_ranges
    
    rospy.logdebug("data=" + str(data))
    rospy.logwarn("new_ranges=" + str(new_ranges))
    rospy.logwarn("mod=" + str(mod))
    
    for i, item in enumerate(data.ranges):
      if (i%mod==0):
        if item == float ('Inf') or np.isinf(item):
          discretized_ranges.append(self.max_laser_value)
        elif np.isnan(item):
          discretized_ranges.append(self.min_laser_value)
        else:
          discretized_ranges.append(int(item))

        if (self.min_range > item > 0):
          rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
          self._episode_done = True
        else:
          rospy.logwarn("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
        

    return discretized_ranges

