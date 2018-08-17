# Setup Turtlebot Reinforcement Learning Simulation Environment
Assume you have set up an ros workspace at `/home/yourname/ros_ws`, and ready to config the environment in it. 


## Config Turtlebot
`$ sudo apt install ros-kinetic-turtlebot-gazebo`
Sometimes, you may need to `$ cd /opt/ros/kinetic`, then `$ source setup.bash` <br/>
To make sure `turtlebot_gazebo` is launchable, try `$ roslaunch turtlbot_gazebo turtlebot_world.launch`

## Config [openai_ros](http://wiki.ros.org/openai_ros)
`$ cd ~/ros_ws/src` <br/>
`$ git clone https://bitbucket.org/theconstructcore/openai_ros.git` <br/>
`$ cd ~/ros_ws && catkin build openai_ros` <br/>
`$ source devel/setup.bash` <br/>
`$ rosdep install openai_ros`

## Start Q Learning
`$ cd ~/ros_ws/src`, then clone this repo `$ git clone https://github.com/linZHank/turtlebot_rl.git`
to build this package, `$ cd ~/ros_ws`, then `$ catkin build`. Don't forget to `$ source devel/setup.bash`, 
or you can add this line at the bottom of your system bash file `echo "source ~/ros_ws/devel/setup.bash" >> ~/.bashrc`<br/>
To initiate the gazebo simulation, `$ roslaunch turtlebot_rl turtlebot_maze.launch` <br/>
To start Q learning, `$ roslaunch turtlebot_rl start_qlearning.launch` <br/>

## Notes
This repo was created to test the contents in this openai_ros's [tutorial](http://wiki.ros.org/openai_ros/TurtleBot2%20with%20openai_ros).
However, may be different configurations were implemented, I made following changes to realize my turtlebot bumping around in the maze. <br/>
- In original tutorial, they must have a hokuyo lidar included in their simulation, 
but using their launch file would not bringup anything related to laser scan. 
So I have to copy the `fake laser` part in `turtlebot_gazebo`'s launch file.
Besides, the original tutorial was using the topic of `/kobuki/laser/scan` to publish laser's data,
hence I have to remap `scan` to `/kobuki/laser/scan` in my launch file.
- In original tutorial, they use topic of `cmd_vel` to publish command to control the turtlebot.
However, I found my turtlebot use topic of `/mobile_base/commands/velocity` to publish commands.
So, I have to edit ` ~/ros_ws/src/openai_ros/src/openai_ros/robot_envs/turtlebot2_env.py`.
> I don't understand why I cannot simply use a `remap` in my launch file to fix it.
Anyway, you'll find the topic around "line 76", just change the publisher's topic from `/cmd_vel` to `/mobile_base/commands/velocity`.
Then you should good to go.

