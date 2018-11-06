#!/usr/bin/env python

from __future__ import print_function

import rospy
from gazebo_msgs.msg import LinkStates


def linkstates_callback(data):
  global state
  state = data

rospy.init_node("link_states_test")
rospy.Subscriber("/gazebo/link_states", LinkStates, linkstates_callback)
rospy.spin()
