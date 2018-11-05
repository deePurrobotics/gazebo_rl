#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import rospy
import numpy as np
import time

# ROS packages required
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.msg import LinkState


if __name__ == '__main__':
    rospy.init_node('cablearning_node', anonymous=True, log_level=rospy.DEBUG)
    rospy.logwarn('initializing node finished')

    force_publisher=rospy.Publisher('/gazebo_client/force', Float32MultiArray, queue_size=10)
    reset_sim_proxy=rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    reset_world_proxy=rospy.ServiceProxy('/gazebo/reset_world', Empty)
    cmd_force = Float32MultiArray()
    # rate = rospy.Rate(10)
    while True:
        for _ in range(20):
            cmd_force.data = np.random.randn(4)
            force_publisher.publish(cmd_force)
            rospy.logdebug("Force: {}".format(cmd_force.data))
            # rate.sleep()
            rospy.wait_for_service('/gazebo/reset_simulation')
            time.sleep(.1)
        try:
            reset_sim_proxy()
            rospy.logwarn("Simulation reset!")
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")
            rospy.logwarn("Reset Simulation!")
