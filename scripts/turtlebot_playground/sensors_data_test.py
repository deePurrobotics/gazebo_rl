#!/usr/bin/env python
from __future__ import print_function

import gym
import numpy as np
import time
import random
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
from gazebo_msgs.msg import ModelStates, LinkStates
import cv2
from cv_bridge import CvBridge, CvBridgeError
# import our training environment
from envs import playground_fetch_task_env # need write task env


def rgb_cb(data):
  global cv_image
  bridge = CvBridge()
  cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
  # cv2.imshow("Image window", cv_image)
  # cv2.waitKey(3)

def depth_cb(data):
  global cv_depth
  bridge = CvBridge()
  cv_depth = bridge.imgmsg_to_cv2(data, "32FC1")
  # cv_depth_array = np.array(cv_depth, dtype = np.dtype('f8'))
  # cv_depth_norm = cv2.normalize(cv_depth_array, cv_depth_array, 0, 1, cv2.NORM_MINMAX)
  # cv_depth_resized = cv2.resize(cv_depth_norm, (200,200), interpolation = cv2.INTER_CUBIC)
  # cv2.imshow("Depth window", cv_depth)
  # cv2.waitKey(3)

def point_cb(data):
  global point_cloud
  point_cloud = data.data
  # rospy.logdebug("point_cloud: \n---\n{}".format(point_cloud))
    
def laser_cb(data):
  global laser_scan
  laser_scan = data.ranges
  # rospy.logdebug("laser_scan: \n---\n{}".format(laser_scan))

def imu_cb(data):
  global orientation
  global ang_vel
  global lin_acc
  orientation = data.orientation
  ang_vel = data.angular_velocity
  lin_acc = data.linear_acceleration
  rospy.logdebug("Orientaion: \n{}\n".format(orientation),
                 "---\nAngular_velocity: \n{}\n".format(ang_vel),
                 "---\nLinear_acceleration: \n{}\n".format(lin_acc))
  
if __name__ == "__main__":
  rospy.init_node('sensors_test', anonymous=True, log_level=rospy.DEBUG)
  rospy.Subscriber("/camera/rgb/image_raw", Image, rgb_cb)
  rospy.Subscriber("/camera/depth/image_raw", Image, depth_cb)
  rospy.Subscriber("/camera/depth/points", PointCloud2, point_cb)
  rospy.Subscriber("/kobuki/laser/scan", LaserScan, laser_cb)
  rospy.Subscriber("/mobile_base/sensors/imu_data", Imu, imu_cb)
  rospy.spin()
  cv2.destroyAllWindows()
