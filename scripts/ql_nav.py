#!/usr/bin/env python

import gym
import numpy
import time
import utils
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
import openai_ros_envs.crib_task_env


if __name__ == '__main__':
    rospy.init_node('turtlebot2_crib_qlearn', anonymous=True, log_level=rospy.INFO)
    # Create the Gym environment
    env = gym.make('TurtleBotCrib-v0')
    rospy.loginfo("Gym environment done")
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot_rl')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")
    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot2/alpha") # learning rate
    Epsilon = rospy.get_param("/turtlebot2/epsilon") # exploration rate
    Gamma = rospy.get_param("/turtlebot2/gamma") # reward discount
    epsilon_discount = rospy.get_param("/turtlebot2/epsilon_discount") # exploration decay
    n_episodes = rospy.get_param("/turtlebot2/nepisodes")
    n_steps = rospy.get_param("/turtlebot2/nsteps")

    running_step = rospy.get_param("/turtlebot2/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    low = env.observation_space.low

    # Starts the main training loop: the one about the episodes to do
    for epi in range(n_episodes):
      rospy.logdebug("############### START EPISODE=>" + str(epi))
      env.reset()
      total_reward = 0
      # done = False
      if qlearn.epsilon > 0.05:
        qlearn.epsilon *= epsilon_discount
        # Initialize the environment and get first state of the robot
        observation = env.reset()
        obs = observation[:2] # we only need x, y
        state = utils.obs2state(obs, low)
        # state = ''.join(map(str, observation))

        # For each episode, we test the robot for nsteps
        for ste in range(n_steps):
          rospy.logwarn("############### Start Step=>" + str(ste))
          # Pick an action based on the current state
          action = qlearn.chooseAction(state)
          rospy.logwarn("Next action is:%d", action)
          # Execute the action in the environment and get feedback
          observation, reward, done, info = env.step(action)
          total_reward += reward
          # if highest_reward < cumulated_reward:
          #     highest_reward = cumulated_reward
          
          # nextState = ''.join(map(str, observation))
          nextState = utils.obs2state(observation[:2], low)
          rospy.loginfo("# state we were=> {}".format(state))
          rospy.loginfo("# action that we took=>" + str(action))
          rospy.loginfo("# reward that action gave=>" + str(reward))
          rospy.loginfo("# episode cumulated_reward=>" + str(total_reward))
          rospy.loginfo("# State in which we will start next step=>" + str(nextState))
          qlearn.learn(state, action, reward, nextState)
          state = nextState

          if done:
            break
        rospy.logwarn("Iteration {} -- Total reward = {}".format(epi, total_reward))
        #   if not (done):
        #     rospy.logwarn("NOT DONE")
        #     state = nextState
        #   else:
        #     rospy.logwarn("DONE")
        #     last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
        #     break
        #   rospy.logwarn("############### END Step=>" + str(i))
        #   #raw_input("Next Step...PRESS KEY")
        #   # rospy.sleep(2.0)
        # m, s = divmod(int(time.time() - start_time), 60)
        # h, m = divmod(m, 60)

    env.close()
