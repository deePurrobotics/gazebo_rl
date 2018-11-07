#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .gym_gazebo_env import GymGazeboEnv
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Float32MultiArray


class CableJointRobotEnv(GymGazeboEnv):
  """
  Superclass for all cable driven robot environments. Contains all sensors and actuators methods.
  """

  def __init__(self):
    """
    Initializes a new TurtleBot2Env environment.
    
    Sensor Topic List:
    * /odom : Odometry readings of the base of the robot
    * /camera/depth/image_raw: 2d Depth image of the depth sensor.
    * /camera/depth/points: Pointcloud sensor readings
    * /camera/rgb/image_raw: RGB camera
    * /kobuki/laser/scan: Laser Readings
    * /gazebo/link_states: Gazebo simulated model states

    Actuators Topic List: 
    * /mobile_base/commands/velocity: velocity command for driving the base of the robot
    """
    rospy.logdebug("Start CableJointRobotEnv INIT...")
    # Variables that we give through the constructor.
    # None in this case

    # Internal Vars
    # Doesnt have any accesibles
    self.controllers_list = []

    # It doesnt use namespace
    self.robot_name_space = ""

    # We launch the init function of the Parent Class gym_gazebo_env.GymGazeboEnv
    super(CableJointRobotEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="SIMULATION"
    )

    self.gazebo.unpauseSim()
    #self.controllers_object.reset_controllers()
    self._check_all_sensors_ready()

    # We Start all the ROS related Subscribers and publishers
    rospy.Subscriber("/gazebo/link_states", LinkStates, self._link_states_callback)

    self._cmd_frc_pub = rospy.Publisher("/gazebo_client/force", Float32MultiArray, queue_size=1)
    self._check_publishers_connection()

    self.gazebo.pauseSim()
        
    rospy.logdebug("Finished CableJointRobotEnv INIT...")

    
  # Methods needed by the RobotGazeboEnv
  # ----------------------------
  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    self._check_all_sensors_ready()
    return True


  # TurtleBotEnv virtual methods
    # ----------------------------

  def _check_all_sensors_ready(self):
    rospy.logdebug("START ALL SENSORS READY")
    self._check_link_states_ready()
    rospy.logdebug("ALL SENSORS READY")

  def _check_link_states_ready(self):
    self.link_states = None
    rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
    while self.link_states is None and not rospy.is_shutdown():
      try:
        self.link_states = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
        rospy.logdebug("Current /gazebo/link_states READY=>")
      except:
        rospy.logerr("Current /gazebo/link_states not ready yet, retrying for getting link_states")
        
    return self.link_states

  # Call back functions read subscribed sensors' data
  # ----------------------------
  def _link_states_callback(self, data):
    self.link_states = data


  def _check_publishers_connection(self):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(10)  # 10hz
    while self._cmd_frc_pub.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to _cmd_frc_pub yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    rospy.logdebug("_cmd_frc_pub Publisher Connected")
    rospy.logdebug("All Publishers READY")

  def get_link_states(self):
    return self.link_states

  # Methods that the TrainingEnvironment will need to define here as virtual
  # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
  # TrainingEnvironment.
  # ----------------------------
  def _set_init(self):
    """Sets the Robot in its init pose
    """
    raise NotImplementedError()
    
  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    raise NotImplementedError()

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _get_observation(self):
    raise NotImplementedError()

  def _post_information(self):
    raise NotImplementedError()

  def _is_done(self):
    """Checks if episode done based on observations given.
    """
    raise NotImplementedError()
