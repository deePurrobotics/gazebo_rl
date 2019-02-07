#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import rospy
import numpy as np
import time

# ROS packages required
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetLinkState, ApplyBodyWrench
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Position, Pose, Wrench


class CableTester(object):
  def __init_(self):
    self.rate = rospy.Rate(10)
    self.ne_publisher = rospy.Publisher("force_northeast", Wrench, queue_size=1)  
    self.nw_publisher = rospy.Publisher("force_northwest", Wrench, queue_size=1)  
    self.sw_pub = rospy.Publisher("force_southeast", Wrench, queue_size=1)  
    self.se_pub = rospy.Publisher("force_southwest", Wrench, queue_size=1)  
    rospy.Subscriber("/gazebo/link_states", LinkStates, link_states_cb)
    
  def apply_force(self, force_array):
    wrench_ne = Wrench()
    wrench_nw = Wrench()
    wrench_sw = Wrench()
    wrench_se = Wrench()
    
    self.rate.sleep()

  def link_states_cb(self, data):
    id_eastpin = data.name.index("cable_joint::link_pin_east")
    id_westpin = data.name.index("cable_joint::link_pin_west")
    id_necube = data.name.index("cable_joint::link_cube_northeast")
    id_nwcube = data.name.index("cable_joint::link_cube_northwest")
    id_swcube = data.name.index("cable_joint::link_cube_southwest")
    id_secube = data.name.index("cable_joint::link_cube_southeast")
    # get pins and cubes position
    self.eastpin_position = position_to_array(data.pose[id_eastpin].position)
    self.westpin_position = data.pose[id_westpin].position
    self.necube_position = data.pose[id_necube].position
    self.nwcube_position = data.pose[id_nwcube].position
    self.swcube_position = data.pose[id_swcube].position
    self.secube_position = data.pose[id_secube].position
    # compute vectors between corresponding cubes and pins
    
    def position_to_array(position):
      pass
    
    

if __name__ == '__main__':
  reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
  rospy.init_node('apply_wrench_test', anonymous=True, log_level=rospy.DEBUG)
  rospy.logwarn('initializing node finished')

  # start testing
  reset_world()
  force = np.zeros(4)
  rate = rospy.Rate(10)
  for i in range(100):
    force[i%4] = 0.01
    apply_force(rate, force)

  rospy.logwarn("Cable Joint Test Complete!")
  

    
