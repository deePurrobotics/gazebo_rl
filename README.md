# Note
This repo contains gazebo based reinforcement learning environements and example scripts greatly inspired by [openai_ros](https://bitbucket.org/theconstructcore/openai_ros.git).
The environments are build with [Gazebo-7.0](http://gazebosim.org/tutorials?tut=install_ubuntu&ver=7.0) simulation software and [OpenAI-gym](https://github.com/openai/gym) toolkit.

## Pre-requisite
- Install [ROS-Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu), recommand Desktop-Full install.
- **(Optional but recommend)** Install [catkin-command-line-tools](https://catkin-tools.readthedocs.io/en/latest/)
- **(Optional, skip this step if you've installed ROS in Desktop-Full mode)** Install [Gazebo-7.0](http://gazebosim.org/tutorials?tut=install_ubuntu&ver=7.0)
- Install [OpenAI-gym](https://github.com/openai/gym#installation)
  > `pip install gym`
- ~~**(Optional)** Install [openai_ros](https://bitbucket.org/theconstructcore/openai_ros.git)~~
- Setup turtlebot Gazebo simulation environment<br/>
  - Assume you have set up an ros workspace at `/home/yourname/ros_ws`, and ready to config the environment in it. 
  `$ sudo apt install ros-kinetic-turtlebot-gazebo`
  - To make sure `turtlebot_gazebo` is launchable, try `$ roslaunch turtlbot_gazebo turtlebot_world.launch`
- To get ready for reinforcement learning, `$ cd ~/ros_ws/src`, then clone this repo `$ git clone https://github.com/deePurrobotics/gazebo_rl.git`
- `$ cd ~/ros_ws`, then `$ catkin build `

  
## Reinforcement Learning Environments
> All codes are located in `*this_repo*/scripts/`

### CribNav-v0
1. Launch learning environment, `$ roslaunch gazebo_rl turtlebot_crib.launch` 
2. Open a new terminal, you can `$ rosrun gazebo_rl crib_nav_qtable` to start a q-learning"
   > make sure to `$ chmod +x *this_repo*/scripts/turtlebot/crib_nav_qtable.py`
### CablePoint-v0
1. Copy the model files of cable-driven joint into local gazebo models library
   `$ cp -a *this_repo*/gazebo_models/cable_joint/ ~/.gazebo/models/cable_joint/`
2. CMake force plugin for the cable-driven joint
   ```bash
   $ cd *this_repo*/worlds/cable_world/
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```
3. **Add `export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:**this_repo**/worlds/cable_world/build` to `~/.bashrc`**
4. Launch cable joint simulation, `$ roslaunch gazebo_rl cable_joint.launch`
5. Open a new terminal to test this env, `$ rosrun gazebo_rl cable_env_test`
   > make sure to `$ chmod +x *this_repo*/scripts/cable_joint/cable_env_test.py`
