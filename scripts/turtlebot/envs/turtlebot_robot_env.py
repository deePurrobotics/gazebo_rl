from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .gym_gazebo_env import GymGazeboEnv
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class TurtlebotRobotEnv(GymGazeboEnv):
  """
  Superclass for all TurtleBot environments. Contains all sensors and actuators methods.
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
    * /gazebo/model_states: Gazebo simulated model states

    Actuators Topic List: 
    * /mobile_base/commands/velocity: velocity command for driving the base of the robot
    """
    rospy.logdebug("Start TurtleBotEnv INIT...")
    # Variables that we give through the constructor.
    # None in this case

    # Internal Vars
    # Doesnt have any accesibles
    self.controllers_list = []

    # It doesnt use namespace
    self.robot_name_space = ""

    # We launch the init function of the Parent Class gym_gazebo_env.GymGazeboEnv
    super(TurtlebotRobotEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="SIMULATION"
    )

    self.gazebo.unpauseSim()
    #self.controllers_object.reset_controllers()
    self._check_all_sensors_ready()

    # We Start all the ROS related Subscribers and publishers
    rospy.Subscriber("/odom", Odometry, self._odom_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)
    rospy.Subscriber("/camera/depth/points", PointCloud2, self._camera_depth_points_callback)
    rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)
    rospy.Subscriber("/kobuki/laser/scan", LaserScan, self._laser_scan_callback)
    rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_callback)

    self._cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=1)
    self._check_publishers_connection()

    self.gazebo.pauseSim()
        
    rospy.logdebug("Finished TurtleBot2Env INIT...")

    
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
    self._check_odom_ready()
    # We dont need to check for the moment, takes too long
    #self._check_camera_depth_image_raw_ready()
    #self._check_camera_depth_points_ready()
    #self._check_camera_rgb_image_raw_ready()
    #self._check_laser_scan_ready()
    self._check_model_states_ready()
    rospy.logdebug("ALL SENSORS READY")

  def _check_odom_ready(self):
    self.odom = None
    rospy.logdebug("Waiting for /odom to be READY...")
    while self.odom is None and not rospy.is_shutdown():
      try:
        self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
        rospy.logdebug("Current /odom READY=>")
      except:
        rospy.logerr("Current /odom not ready yet, retrying for getting odom")

    return self.odom

  def _check_camera_depth_image_raw_ready(self):
    self.camera_depth_image_raw = None
    rospy.logdebug("Waiting for /camera/depth/image_raw to be READY...")
    while self.camera_depth_image_raw is None and not rospy.is_shutdown():
      try:
        self.camera_depth_image_raw = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5.0)
        rospy.logdebug("Current /camera/depth/image_raw READY=>")
      except:
        rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting camera_depth_image_raw")

    return self.camera_depth_image_raw        
        
  def _check_camera_depth_points_ready(self):
    self.camera_depth_points = None
    rospy.logdebug("Waiting for /camera/depth/points to be READY...")
    while self.camera_depth_points is None and not rospy.is_shutdown():
      try:
        self.camera_depth_points = rospy.wait_for_message("/camera/depth/points", PointCloud2, timeout=10.0)
        rospy.logdebug("Current /camera/depth/points READY=>")
      except:
        rospy.logerr("Current /camera/depth/points not ready yet, retrying for getting camera_depth_points")
        
    return self.camera_depth_points    
        
  def _check_camera_rgb_image_raw_ready(self):
    self.camera_rgb_image_raw = None
    rospy.logdebug("Waiting for /camera/rgb/image_raw to be READY...")
    while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
      try:
        self.camera_rgb_image_raw = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=5.0)
        rospy.logdebug("Current /camera/rgb/image_raw READY=>")
      except:
        rospy.logerr("Current /camera/rgb/image_raw not ready yet, retrying for getting camera_rgb_image_raw")

    return self.camera_rgb_image_raw
        

  def _check_laser_scan_ready(self):
    self.laser_scan = None
    rospy.logdebug("Waiting for /kobuki/laser/scan to be READY...")
    while self.laser_scan is None and not rospy.is_shutdown():
      try:
        self.laser_scan = rospy.wait_for_message("/kobuki/laser/scan", LaserScan, timeout=5.0)
        rospy.logdebug("Current /kobuki/laser/scan READY=>")
      except:
        rospy.logerr("Current /kobuki/laser/scan not ready yet, retrying for getting laser_scan")
        
    return self.laser_scan    

  def _check_model_states_ready(self):
    self.model_states = None
    rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
    while self.model_states is None and not rospy.is_shutdown():
      try:
        self.model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
        rospy.logdebug("Current /gazebo/model_states READY=>")
      except:
        rospy.logerr("Current /gazebo/model_states not ready yet, retrying for getting model_states")
        
    return self.model_states

  # Call back functions read subscribed sensors' data
  # ----------------------------
  def _odom_callback(self, data):
    self.odom = data
    
  def _camera_depth_image_raw_callback(self, data):
    self.camera_depth_image_raw = data
        
  def _camera_depth_points_callback(self, data):
    self.camera_depth_points = data
        
  def _camera_rgb_image_raw_callback(self, data):
    self.camera_rgb_image_raw = data
        
  def _laser_scan_callback(self, data):
    self.laser_scan = data

  def _model_states_callback(self, data):
    self.model_states = data


  def _check_publishers_connection(self):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(10)  # 10hz
    while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    rospy.logdebug("_cmd_vel_pub Publisher Connected")
    rospy.logdebug("All Publishers READY")
    
  def get_odom(self):
    return self.odom
        
  def get_camera_depth_image_raw(self):
    return self.camera_depth_image_raw
        
  def get_camera_depth_points(self):
    return self.camera_depth_points
        
  def get_camera_rgb_image_raw(self):
    return self.camera_rgb_image_raw
        
  def get_laser_scan(self):
    return self.laser_scan

  def get_model_states(self):
    return self.model_states


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
