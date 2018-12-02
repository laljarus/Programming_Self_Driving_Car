# Self Driving Car Capstone Project - Individual Submission
The aim of this project is to integrate different systems of a autonomous car using Robot Operating System (ROS). The software drives a car in the Udacity - simulator autonomously.
The main requirements for the project are
 - The car should follow given waypoints smoothly
 - The car should drive within the speed limit.
 - The car should stop for the traffic lights.
 
 The program is able to drive the car successfully as per the requirements
 
 ## Software Architecture
 
 ![img](final-project-ros-graph-v2.png)
 
 The above picture shows the software architecture of the project. As mentioned before ROS is used to develop this project. ROS is robotics middleware platform which can integrate different systems together. ROS provides the framework for different systems to communicate among each other. There three main subsystems used in this project the Perception, Planning and Control subsystems as shown in the picture above.
 
 ### Perception Subsystem
 
 The perception subsystem of an autonomous car would consists of system which detects the obstacles such as other cars, road signs,pedestrians etc. Only the traffic light detection is done in this implemtation of driving the car in the simulator. For traffic light detection the simulator provides the position of the car and the position of the traffic lights. Usually these informaiton comes from the localization and mapping subsystems in a self-driving car for this project the simulator provides it. The traffic light detection is implemented in the tl_detector node. This node has two functions one is to find the location of the next traffic light ahead of the car and provide the waypoint id for the upcoming traffic light. This is done by comparing the actual position of the car and the list of traffic light position given in the map of the track. The second part is to detect the state of traffic light. For doing this the simulator provides the image of the road ahead using a camera. This image is processed using color thresholding methd to find the status of the traffic light. Since the position of the car and traffic lights are known for this case this simple method is sufficient however in real world more sophosticated object detection algorithms must be used to detect this.
 
 ## Planning Subsystem 
 
 The planning subsystem gets the map of the track and the current position of the car and based on this information it plans a trajectory of the car. This is sent to the control subsystem which realizes the trajectory. This is implemented in the waypoint updater which calculates the path by finding car's nearest waypoint in the map using a KDtree. The KDtree algorithms helps to search the nearest point in the map more efficiently than looping through all the points in the map. The second part of the waypoint updater node is to provide a smooth trajectories for decelerating and accelerating the car depending the traffic lights. The perception subsystem sends the planning subsystem the location of the next Red/Yellow traffic light. Based on this information waypoint updater provides smooth deceleration profile using exponential function. When the car stops for red light the waypoint updater node will provide acceleration profile once the light is green again.
 
 ## Control Subsystem
 
 The control subsystem realizes the trajectory which the planning subsystem sends. The control subsystem has two nodes waypoint follower and the drive by wire node. The waypoint follower nodes receives the trajectory informaiton from the planning subsystem and converts it into vehicle velocity commands in the linear x,y and angular z(yaw) directions. Based on the velocity commands the dbw calculates the steering, throttle and brake commands. The longitudinal control of car which is done by trottle and brake command is done using a simple PID controller. The steering command is calculated using a feed forward controller which calculates steering angle based on yaw rate. For steering controller an integrating controller with a small gain is used because it reduces the offset in between the commanded and actual yaw. The lateral control is done using a feed forward controller because of the latency between the simulator and the program which makes the car unstable when PID contorller were used. However the feedforward control for steering may not be suitable for a real car.

## Summary

The program runs quite well in the simulator. Only simple algorithms were used in this implementation because it is aimed only for the simulator. However this can be used as a base for developing algorithm for driving a real car
 
 
