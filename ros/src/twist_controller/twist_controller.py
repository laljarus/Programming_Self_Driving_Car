
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

class Controller(object):
    def __init__(self, wheel_base,steer_ratio,max_lat_acc,max_steer_angle):

        # TODO: Implement
        self.wheel_base 		= wheel_base
        self.steer_ratio 		= steer_ratio
        self.min_speed 			= 1.0 				# m/s minimum speed for steering
        self.max_lat_acc 		= max_lat_acc
        self.max_steer_angle	= max_steer_angle

        self.kp_lon				= 0
        self.kd_lon				= 0
        self.ki_lon				= 0

        self.kp_lat				= 0
        self.kd_lat				= 0
        self.ki_lat				= 0

        self.steer 				= 0

        self.tau_vel_filt 		= 0.3
        self.ts 				= 1/20.0  # sample time

        self.vel_low_pass_filt  = LowPassFilter(self.tau_vel_filt,self.ts)



        self.YawController = YawController(self.wheel_base,self.steer_ratio,self.min_speed,self.max_lat_acc,self.max_steer_angle)

    def control(self, cmd_x_dot,cmd_yaw_rate,act_x_dot):
        # TODO: Change the arg, kwarg list to suit your needs
        
        act_x_dot = self.vel_low_pass_filt.filt(act_x_dot)

        self.steer = self.YawController.get_steering(cmd_x_dot,cmd_yaw_rate,act_x_dot)

        # Return throttle, brake, steer
        return 0.2, 0., self.steer
