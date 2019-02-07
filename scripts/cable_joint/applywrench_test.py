#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import rospy
import numpy as np
import time

# ROS packages required
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetLinkState, ApplyBodyWrench
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose, Wrench


def position_to_array(position):
  array = np.array([position.x, position.y, position.z]).astype(np.float32)
  return array

def array_to_wrench(array):
  wrench = Wrench()
  wrench.force.x = array[0]
  wrench.force.y = array[1]
  wrench.force.z = array[2]
  return wrench

class CableTester(object):
  def __init__(self):
    # pins and cubes positions
    self.eastpin_position = np.array([0.1, 0, 0.01])
    self.westpin_position = np.array([-0.1, 0, 0.01])
    self.necube_position = np.array([0.0707, 0.0707, 0.19])
    self.nwcube_position = np.array([-0.0707, 0.0707, 0.19])
    self.swcube_position = np.array([-0.0707, -0.0707, 0.19])
    self.secube_position = np.array([0.0707, -0.0707, 0.19])
    self.vec_necube_eastpin = self.eastpin_position - self.necube_position
    self.vec_nwcube_westpin = self.westpin_position - self.nwcube_position
    self.vec_swcube_westpin = self.westpin_position - self.swcube_position
    self.vec_secube_eastpin = self.eastpin_position - self.secube_position
    # rostopic related
    self.rate = rospy.Rate(100)
    self.ne_publisher = rospy.Publisher("force_northeast", Wrench, queue_size=1)  
    self.nw_publisher = rospy.Publisher("force_northwest", Wrench, queue_size=1)  
    self.sw_publisher = rospy.Publisher("force_southwest", Wrench, queue_size=1)  
    self.se_publisher = rospy.Publisher("force_southeast", Wrench, queue_size=1)
    rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_cb)
    
  def apply_force(self, force_array):
    # clean force array
    assert len(force_array)==4
    force_array = np.clip(force_array, 0, 0.0004)
    # define wrenches
    force_ne = force_array[0] * (self.vec_necube_eastpin / np.linalg.norm(self.vec_necube_eastpin))
    force_nw = force_array[0] * (self.vec_nwcube_westpin / np.linalg.norm(self.vec_nwcube_westpin))
    force_sw = force_array[0] * (self.vec_swcube_westpin / np.linalg.norm(self.vec_swcube_westpin))
    force_se = force_array[0] * (self.vec_secube_eastpin / np.linalg.norm(self.vec_secube_eastpin))
    wrench_ne = array_to_wrench(force_ne)
    wrench_nw = array_to_wrench(force_nw)
    wrench_sw = array_to_wrench(force_sw)
    wrench_se = array_to_wrench(force_se)
    # apply wrenches
    zero_force = array_to_wrench(np.zeros(3))
    for _ in range(10):
      self.ne_publisher.publish(wrench_ne)
      self.nw_publisher.publish(wrench_nw)
      self.sw_publisher.publish(wrench_sw)
      self.se_publisher.publish(wrench_se)
      self.rate.sleep()
    # # disarm forces
    # self.ne_publisher.publish(zero_force)
    # self.nw_publisher.publish(zero_force)
    # self.sw_publisher.publish(zero_force)
    # self.se_publisher.publish(zero_force)  
      
  def link_states_cb(self, data):
    id_eastpin = data.name.index("cable_joint::link_pin_east")
    id_westpin = data.name.index("cable_joint::link_pin_west")
    id_necube = data.name.index("cable_joint::link_cube_northeast")
    id_nwcube = data.name.index("cable_joint::link_cube_northwest")
    id_swcube = data.name.index("cable_joint::link_cube_southwest")
    id_secube = data.name.index("cable_joint::link_cube_southeast")
    # get pins and cubes position
    self.eastpin_position = position_to_array(data.pose[id_eastpin].position)
    self.westpin_position = position_to_array(data.pose[id_westpin].position)
    self.necube_position = position_to_array(data.pose[id_necube].position)
    self.nwcube_position = position_to_array(data.pose[id_nwcube].position)
    self.swcube_position = position_to_array(data.pose[id_swcube].position)
    self.secube_position = position_to_array(data.pose[id_secube].position)
    # compute vectors between corresponding cubes and pins
    self.vec_necube_eastpin = self.eastpin_position - self.necube_position
    self.vec_nwcube_westpin = self.westpin_position - self.nwcube_position
    self.vec_swcube_westpin = self.westpin_position - self.swcube_position
    self.vec_secube_eastpin = self.eastpin_position - self.secube_position
        

if __name__ == '__main__':
  reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
  reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
  rospy.init_node('apply_wrench_test', anonymous=True, log_level=rospy.DEBUG)
  rospy.logwarn('initializing node finished')
  # start testing
  for _ in range(10):
    reset_simulation()
  tester = CableTester()

  rospy.logerr("Simulation Reset")
  force = np.zeros(4)
  for i in range(100):
    force = np.random.randn(4)
    tester.apply_force(force)
    rospy.logdebug("iter:{}, force: {}".format(i, force))

  rospy.logwarn("Cable Joint Test Complete!")
  

    
