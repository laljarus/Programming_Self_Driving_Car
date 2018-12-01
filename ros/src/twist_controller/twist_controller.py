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
        self.min_speed 			= 0.1 				# m/s minimum speed for steering
        self.max_lat_acc 		= max_lat_acc
        self.max_steer_angle	= max_steer_angle

        self.kp_lon				= 0.2
        self.kd_lon				= 0.05
        self.ki_lon				= 0.

        self.kp_lat				= -1
        self.kd_lat				= -0.5
        self.ki_lat				= 0
        
        self.min_throttle       = decel_limit
        self.max_throttle       = 0.3       

        self.steer 				= 0

        self.tau_vel_filt 		= 0.3
        self.ts 				= 1/20.0  # sample time

        self.vel_low_pass_filt  = LowPassFilter(self.tau_vel_filt,self.ts)



        
        #self.yaw_rate_filt	= LowPassFilter(self.tau_vel_filt,self.ts)
        self.pid_lon        = PID(self.kp_lon,self.ki_lon,self.kd_lon,self.min_throttle,self.max_throttle)
        self.pid_lat 		= PID(self.kp_lat,self.ki_lat,self.kd_lat,-self.max_steer_angle,self.max_steer_angle)
        self.last_time 		= 0
        self.YawController = YawController(self.wheel_base,self.steer_ratio,self.min_speed,self.max_lat_acc,self.max_steer_angle)

        self.decel_limit    = decel_limit
        self.vehicle_mass   = vehicle_mass
        self.wheel_radius   = wheel_radius


    def control(self, cmd_x_dot,cmd_yaw_rate,act_x_dot,cte,dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        
        act_x_dot = self.vel_low_pass_filt.filt(act_x_dot)
        #act_yaw_rate = self.yaw_rate_filt.filt(act_yaw_rate)       

        if not dbw_enabled:
        	self.pid_lat.reset()
        	return 0.,0.,0.        

        current_time = rospy.get_time()
        dt =  current_time - self.last_time
        self.last_time = current_time       
          

        speed_error = cmd_x_dot - act_x_dot
        throttle = self.pid_lon.step(speed_error,0.,dt)
        brake = 0.

        steer_ff = self.YawController.get_steering(act_x_dot,cmd_yaw_rate,cmd_x_dot)
        steer = self.pid_lat.step(cte,steer_ff,dt)

        if cmd_x_dot < 0. and act_x_dot < 0.1:
            throttle = 0.
            brake = 400.
        elif throttle < 0. and speed_error < 0.:
            #throttle = throttle*abs(self.decel_limit)
            #decel = max(speed_error,self.decel_limit)
            brake = abs(throttle)*self.vehicle_mass*self.wheel_radius
            throttle = 0.


        # Return throttle, brake, steer
        return throttle, brake, steer_ff
