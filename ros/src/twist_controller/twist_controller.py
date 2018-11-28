
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

class Controller(object):
    def __init__(self, wheel_base,steer_ratio,max_lat_acc,max_steer_angle):

        # TODO: Implement
        self.wheel_base 		= wheel_base
        self.steer_ratio 		= steer_ratio
        self.min_speed 			= 0.1 				# m/s minimum speed for steering
        self.max_lat_acc 		= max_lat_acc
        self.max_steer_angle		= max_steer_angle

        self.kp_lon				= 0
        self.kd_lon				= 0
        self.ki_lon				= 0

        self.kp_lat				= 0.1
        self.kd_lat				= 0.01
        self.ki_lat				= 0

        #self.steer 				= 0

        self.tau_vel_filt 			= 0.3
        self.ts 				= 1/20.0  # sample time

	self.vel_low_pass_filt  = LowPassFilter(self.tau_vel_filt,self.ts)
	self.yaw_rate_filt	= LowPassFilter(self.tau_vel_filt,self.ts)

	self.pid_lat 		= PID(self.kp_lat,self.ki_lat,self.kd_lat,-self.max_steer_angle,self.max_steer_angle)

	self.last_time 		= 0


	self.YawController = YawController(self.wheel_base,self.steer_ratio,self.min_speed,self.max_lat_acc,self.max_steer_angle)

    def control(self, cmd_x_dot,cmd_yaw_rate,act_x_dot,act_yaw_rate,dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        
        act_x_dot = self.vel_low_pass_filt.filt(act_x_dot)
	act_yaw_rate = self.yaw_rate_filt.filt(act_yaw_rate)

	if not dbw_enabled:
		self.pid_lat.reset()
		return 0.,0.,0.

	yaw_rate_error = cmd_yaw_rate - act_yaw_rate
	current_time = rospy.get_time()
	dt =  current_time - self.last_time
	self.last_time = current_time
	steer_ff = self.YawController.get_steering(cmd_x_dot,cmd_yaw_rate,act_x_dot)
	steer = self.pid_lat.step(yaw_rate_error,steer_ff,dt)


        # Return throttle, brake, steer
        return 0.2, 0., steer
