#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint',Int32,self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_idx = -1


        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while  not rospy.is_shutdown():
            if not None in (self.pose,self.base_waypoints,self.waypoints_tree):
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
    def get_closest_waypoint_idx(self):

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x,y],1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        cl_vector = np.array(closest_coord)
        prev_vector = np.array(prev_coord)
        pos_vector = np.array([x,y])

        val = np.dot(cl_vector - prev_vector, pos_vector - cl_vector)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def generate_waypoints(self,closest_idx):
        lane = Lane()
        fathest_wp_idx = closest_idx + LOOKAHEAD_WPS
        lane.header = self.base_waypoints.header
        base_waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        
        if self.stopline_idx == -1 or (self.stopline_idx >= fathest_wp_idx):
            lane.waypoints = base_waypoints            
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints,closest_idx)

        return lane

    def decelerate_waypoints(self,waypoints,closest_idx):
        temp = []
        for i,wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            if i == 0:
                init_vel = wp.twist.twist.linear.x
                stop_idx = max(self.stopline_idx - closest_idx - 8,0)
                init_dist = self.distance(waypoints,i,stop_idx)
                if init_dist < 1:
                    decel = MAX_DECEL
                else:
                    decel = min(MAX_DECEL,(init_vel)/init_dist)            

            stop_idx = max(self.stopline_idx - closest_idx - 8,0)
            dist = self.distance(waypoints,i,stop_idx)
            #vel =  init_vel  + decel*(init_dist - dist)
            vel = math.exp(MAX_DECEL*dist)-8
            rospy.loginfo("velocity:"+str(vel)+"\n")
            #vel = decel * dist
            if vel < 1.:
                vel = 0.

            p.twist.twist.linear.x = min(vel,wp.twist.twist.linear.x)
            temp.append(p)

        return temp



    def publish_waypoints(self,closest_idx):
        final_waypoints = self.generate_waypoints(closest_idx)
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(final_waypoints)

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)        

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_idx = msg.data
        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
