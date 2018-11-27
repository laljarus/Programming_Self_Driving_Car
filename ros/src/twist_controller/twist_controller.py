    
GAS_DENSITY = 2.858
ONE_MPH = 0.44704

from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

class Controller(object):
    def __init__(self, wheel_base,steer_ratio,max_lat_acc,max_steer_angle,decel_limit,vehicle_mass,wheel_radius):

        # TODO: Implement
        self.wheel_base 		= wheel_base
        self.steer_ratio 		= steer_ratio
        self.min_speed 			= 1.0 				# m/s minimum speed for steering
        self.max_lat_acc 		= max_lat_acc
        self.max_steer_angle		= max_steer_angle

        self.kp_lon				= 0.1
        self.kd_lon				= 0.02
        self.ki_lon				= 0

        self.kp_lat				= 0.1
        self.kd_lat				= 0.05
        self.ki_lat				= 0        

        self.tau_vel_filt 			= 0.3
        self.ts 				= 1/20.0  # sample time
        self.min_throttle 			= decel_limit
        self.max_throttle			= 0.3
        self.steering				= 0

        self.vel_low_pass_filt  = LowPassFilter(self.tau_vel_filt,self.ts)
        self.pid_lon 		= PID(self.kp_lon,self.ki_lon,self.kd_lon,self.min_throttle,self.max_throttle)
        self.pid_lat            = PID(self.kp_lat,self.ki_lat,self.kd_lat,-self.max_steer_angle,self.max_steer_angle)

        self.last_time 		= 0	
        self.decel_limit 	= decel_limit
        self.vehicle_mass	= vehicle_mass
        self.wheel_radius 	= wheel_radius


        self.YawController = YawController(self.wheel_base,self.steer_ratio,self.min_speed,self.max_lat_acc,self.max_steer_angle)

    def control(self, cmd_x_dot,cmd_yaw_rate,act_x_dot,act_yaw_rate,dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        
        act_x_dot = self.vel_low_pass_filt.filt(act_x_dot)

        self.steer = self.YawController.get_steering(cmd_x_dot,cmd_yaw_rate,act_x_dot)

        if not dbw_enabd:
		    self.pid_lon.ret()
		    return 00.

	    speed_error = cmd_x_doct_x_dot
	    current_time py.get_time()
	    sample_time = currentself.last_time

        yaw_rate_erd_yaw_rate - act_yaw_rate
	
	    throttle = self.pidstep(spr,sample_time,0.)
	    brake = 0.

        #steer =step(yaw_rate_err,sample_timsteer)
        
	    if cmd_x_d 0. and act_x_dot < 0.1:
		    throte = 0.
		    brake = 400.
	    elif throttle < 0. and speed_err < 0.:
		    throttle = 0.
		    decel = max(speed_errorelf.decel_limit)
		    brake = abs(decel)*self.vehicle_mass*self.wheel_radius
		


        # Return throttle, brake, steer
        return 0.2, 0., self.steer
