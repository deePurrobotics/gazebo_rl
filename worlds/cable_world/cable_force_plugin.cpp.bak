#ifndef _TESTFORCE_PLUGIN_HH_
#define _TESTFORCE_PLUGIN_HH_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/math/gzmath.hh>
#include <gazebo/physics/Link.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32MultiArray.h"
#include <gazebo/common/common.hh>
//#include "ros/Quaternion.h"
//#include "ros/Matrix3x3.h"
//#include "sensor_msgs/ChannelFloat32.h"

namespace gazebo
{
  /// \brief A plugin to control the cable driven joint.
  class CableForcePlugin : public WorldPlugin
  {
    /// \brief Constructor
    public: CableForcePlugin() {}

    public: virtual void Load(physics::WorldPtr _parent, sdf::ElementPtr _sdf)
    {
     this->world=_parent;
     this->model = this->world->GetModel("cable_joint");
     this->toplink=model->GetLink	(	"link_3"	);
     this->toprod=model->GetLink	(	"link_2"	);
     //this->toplink->SetForce(	this->force1);


     // Initialize ros, if it has not already bee initialized.
     if (!ros::isInitialized())
     {
       int argc = 0;
       char **argv = NULL;
       ros::init(argc, argv, "gazebo_client",
       ros::init_options::NoSigintHandler);
     }

     // Create our ROS node. This acts in a similar manner to
     // the Gazebo node
     this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
     this->prevtime=this->world->GetSimTime();
// Create a named topic, and subscribe to it.
    ros::SubscribeOptions so =
    ros::SubscribeOptions::create<std_msgs::Float32MultiArray>(
      "force",
      1,
      boost::bind(&CableForcePlugin::OnRosMsg, this, _1),
      ros::VoidPtr(), &this->rosQueue);
      this->rosSub = this->rosNode->subscribe(so);
// Spin up the queue helper thread.
  this->rosQueueThread =
    std::thread(std::bind(&CableForcePlugin::QueueThread, this));

    //use another node to publish the pose data
    /*ros::NodeHandle n;
    ros::Publisher posepub = n.advertise<std_msgs::Float32MultiArray>("pose", 10);
    //ros::Rate loop_rate(100);
    //while (ros::ok())
    //{
      std_msgs::Float32MultiArray posemsg;
      posemsg.data={this->toplinkpose.pos.x,this->toplinkpose.pos.y,this->toplinkpose.pos.z};
      posepub.publish(posemsg);
      ros::spinOnce();*/
      //loop_rate.sleep();
    //}
    std::cerr <<"testcycle\n";
    }
    //
    public: void ApplyForce(const double &_force1,const double &_force2,const double &_force3,const double &_force4)
    {
      //calculate the force
      this->force1=this->CalForce(_force1,this->pos1,this->bot_pos1);
      this->force2=this->CalForce(_force2,this->pos2,this->bot_pos2);
      this->force3=this->CalForce(_force3,this->pos3,this->bot_pos3);
      this->force4=this->CalForce(_force4,this->pos4,this->bot_pos4);
      //display the magnitude of force on the terminal
          std::cerr <<"force1 magnitude:"<< this->force1.GetLength()<<'\n';
          std::cerr <<"force2 magnitude:"<< this->force2.GetLength()<<'\n';
          std::cerr <<"force3 magnitude:"<< this->force3.GetLength()<<'\n';
          std::cerr <<"force4 magnitude:"<< this->force4.GetLength()<<'\n';
      //set the force direction and magnitude, apply the force to the link
      //here a time control is used to cycle the add force process

      //count is the number of loops for force application
        this->toplink->AddForceAtRelativePosition(	this->force1,this->pos1);
        this->toplink->AddForceAtRelativePosition(	this->force2,this->pos2);
        this->toplink->AddForceAtRelativePosition(	this->force3,this->pos3);
        this->toplink->AddForceAtRelativePosition(	this->force4,this->pos4);
        std::cerr <<"toplinkacceleration:"<<this->toplink->GetWorldLinearAccel()<<'\n';
        std::cerr <<"toprodacceleration:"<<this->toprod->GetWorldLinearAccel()<<'\n';
   }
     //std::cerr << "Force applied\n";
  public: math::Vector3 CalForce(const double &_force, const math::Vector3 &pos, const math::Vector3 &bot_pos)
  {
    //Calculate the force vector according to the pose of link
    //Get the absolute position of
    this->toplinkpose=this->toplink->GetWorldCoGPose	(	);
    this->toplinkposition=this->toplinkpose.pos;
    this->toplinkattitude=this->toplinkpose.rot;
    this->force_dir=bot_pos - this->toplinkposition - this->toplinkattitude.RotateVector(pos);
    std::cerr <<"force direction:"<< this->force_dir.x<< this->force_dir.y<< this->force_dir.z<<'\n';
    return _force * this->force_dir.Normalize();

/*remain to complete
this-force1=balabala;
this-force2=balabala;
this-force3=balabala;
this-force4=balabala;
this->force1=math::Vector3(0,0,-10);
this->force2=math::Vector3(0,0,-10);
this->force3=math::Vector3(0,0,-20);
this->force4=math::Vector3(0,0,-10);*/

  }

    /// \brief Handle an incoming message from ROS

public: void OnRosMsg(const std_msgs::Float32MultiArrayConstPtr &_msg)
{
  //Here this if is to judge 2 situation
  //1.whether a step passed, if current-prev>inerval, a step passed
  //2.whether the world has been reseted here is a solution, if current<prev,world is reseted.
  if (this->world->GetSimTime() - this->prevtime >= this->timeinterval||this->world->GetSimTime() - this->prevtime<0)
  {
    if (this->world->GetSimTime() - this->prevtime > this->timeinterval)failcount+=1;
    std::cerr <<"timedifference:"<<this->world->GetSimTime() - this->prevtime<<'\n';
    this->prevtime=this->world->GetSimTime();
    this->ApplyForce(_msg->data[0],_msg->data[1],_msg->data[2],_msg->data[3]);
    this->globalcount+=1;
std::cerr <<globalcount<<'\n'<<'\n';
std::cerr <<failcount<<'\n'<<'\n';
std::cerr <<common::Time::GetWallTime()<<'\n'<<'\n';
}
  //std::cerr << "Msg arrived\n";
}

/// \brief ROS helper function that processes messages
private: void QueueThread()
{
  static const double timeout = 0.015;
  while (this->rosNode->ok())
  {
    this->rosQueue.callAvailable(ros::WallDuration(timeout));
  }
}
    //set the forces and positions
  private:
    //4 cable-drag force
       double fx1=0, fy1=0, fz1=0;
       double fx2=0, fy2=0, fz2=0;
       double fx3=0, fy3=0, fz3=0;
       double fx4=0, fy4=0, fz4=0;
    //parameters of 8 cable nodes' position
       //top link nodes positions in its own coordinate
       double const posx1=0.0707, posy1=-0.0707,posz1=0;
       double const posx2=0.0707, posy2=0.0707,posz2=0;
       double const posx3=-0.0707, posy3=0.0707,posz3=0;
       double const posx4=-0.0707, posy4=-0.0707,posz4=0;
       //bottom link nodes positions in WORLD coordinate
       double const bot_posx1=0.1, bot_posy1=0,bot_posz1=0.01;
       double const bot_posx2=0.1, bot_posy2=0,bot_posz2=0.01;
       double const bot_posx3=-0.1, bot_posy3=0,bot_posz3=0.01;
       double const bot_posx4=-0.1, bot_posy4=0,bot_posz4=0.01;
    //state the class of the variable used
    private:
    physics::ModelPtr model;
    physics::WorldPtr world;
    physics::LinkPtr toplink;
    physics::LinkPtr toprod;
    //four forces on the cable
    math::Vector3 force1=math::Vector3(fx1,fy1,fz1);
    math::Vector3 force2=math::Vector3(fx2,fy2,fz2);
    math::Vector3 force3=math::Vector3(fx3,fy3,fz3);
    math::Vector3 force4=math::Vector3(fx4,fy4,fz4);
    //relative position of nodes on the toplink
    math::Vector3 const pos1= math::Vector3(posx1,posy1,posz1);
    math::Vector3 const pos2= math::Vector3(posx2,posy2,posz2);
    math::Vector3 const pos3= math::Vector3(posx3,posy3,posz3);
    math::Vector3 const pos4= math::Vector3(posx4,posy4,posz4);
    //absolute position of nodes on the bottom link
    math::Vector3 const bot_pos1= math::Vector3(bot_posx1,bot_posy1,bot_posz1);
    math::Vector3 const bot_pos2= math::Vector3(bot_posx2,bot_posy2,bot_posz2);
    math::Vector3 const bot_pos3= math::Vector3(bot_posx3,bot_posy3,bot_posz3);
    math::Vector3 const bot_pos4= math::Vector3(bot_posx4,bot_posy4,bot_posz4);
    //temporaty variable for force calculation
    math::Vector3 force_dir;

    math::Pose toplinkpose;
    math::Vector3 toplinkposition;
    math::Quaternion toplinkattitude;
    common::Time timeinterval=common::Time(0, common::Time::SecToNano(0.01));
    common::Time prevtime;

    /// \brief A node use for ROS transport
private: std::unique_ptr<ros::NodeHandle> rosNode;

/// \brief A ROS subscriber
private: ros::Subscriber rosSub;

/// \brief A ROS callbackqueue that helps process messages
private: ros::CallbackQueue rosQueue;

/// \brief A thread the keeps running the rosQueue
private: std::thread rosQueueThread;

private: int globalcount=0;
private: int failcount=0;
  };

  // Tell Gazebo about this plugin, so that Gazebo can call Load on this plugin.
  GZ_REGISTER_WORLD_PLUGIN(CableForcePlugin)
}
#endif
