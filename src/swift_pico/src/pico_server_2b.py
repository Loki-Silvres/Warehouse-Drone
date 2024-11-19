#!/usr/bin/env python3

import time
import math
# from tf_transformations import euler_from_quaternion
import numpy as np

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

#import the action
from waypoint_navigation.action import NavToWaypoint

#pico control specific libraries
from swift_msgs.msg import SwiftMsgs
from geometry_msgs.msg import PoseArray
from pid_msg.msg import PIDTune, PIDError
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray

class WayPointServer(Node):

    def __init__(self):
        super().__init__('waypoint_server')

        self.pid_callback_group = ReentrantCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()

        self.time_inside_sphere = 0
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.duration = 0


        self.drone_position = [0.0, 0.0, 0.0, 0.0]
        self.setpoint = [0, 0, 27, 0] 
        self.prev_setpoint = [0, 0, 27, 0] 
        self.dtime = 0

        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        #Kp, Ki and Kd values here: roll, pitch, throttle

        self.Kp = np.array([20, 20, 20, 0])
        self.Ki = np.array([0.1, 0.1, 0.1, 0])
        self.Kd = np.array([500, 500, 300, 0])

        #variables for storing different kinds of errors
        
        self.pid_pos_error = np.zeros_like(self.drone_position).astype(float)
        self.prev_pos_error = np.zeros_like(self.drone_position).astype(float)
        self.sum_pos_error = np.zeros_like(self.drone_position).astype(float)

        self.max_values = {'roll': 1700, 'pitch': 1700, 'yaw': 1700, 'throttle': 1700} # max 2000
        self.min_values = {'roll': 1000, 'pitch': 1000, 'yaw': 1000, 'throttle': 1000}
        self.hover_throttle = 1533  # Hover throttle value

        self.pid_error = PIDError()
        self.pid_error.yaw_error = 0.0
        self.pid_error.roll_error = 0.0
        self.pid_error.pitch_error = 0.0
        self.pid_error.throttle_error = 0.0

        self.rc_roll_sum = 0.0
        self.rc_pitch_sum = 0.0
        self.rc_throttle_sum = 0.0
        self.rc_yaw_sum = 0.0    

        self.sample_time = 0.060

        self.first_point = None
        self.second_point = None

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)

        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        #Add other sunscribers here

        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10)
        self.random_points_sub = self.create_subscription(Int32MultiArray, '/random_points', self.get_goalpoints, 10)

        #create an action server for the action 'NavToWaypoint'. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        #action name should 'waypoint_navigation'.
        #include the action_callback_group in the action server. Refer to executors in ROS 2 concepts

        self.waypoint_action_server = ActionServer(
            self,
            NavToWaypoint,
            'waypoint_navigation',
            self.execute_callback,
            callback_group=self.action_callback_group
        )
        
        self.arm()
        self.timer = self.create_timer(self.sample_time, self.pid, callback_group=self.pid_callback_group)
        self.D_new = 0
        self.stable_error = 0.4
        self.stablise_time = 1.0
    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)


    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)


    def whycon_callback(self, msg):
        #Set the remaining co-ordinates of the drone from msg
        self.drone_position[0] = msg.poses[0].position.x  # x position
        self.drone_position[1] = msg.poses[0].position.y  # y position
        self.drone_position[2] = msg.poses[0].position.z  # z (altitude) position


        self.dtime = msg.header.stamp.sec

    def altitude_set_pid(self, alt):
        self.Kp[1] = alt.kp * 1.0 
        self.Ki[1] = alt.ki * 0.001
        self.Kd[1] = alt.kd * 1.0

    #Define callback function like altitide_set_pid to tune pitch, roll


    def odometry_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # roll, pitch, yaw = euler_from_quaternion(orientation_list)

        # self.roll_deg = math.degrees(roll)
        # self.pitch_deg = math.degrees(pitch)
        # self.yaw_deg = math.degrees(yaw)
        # self.drone_position[3] = self.yaw_deg	

    def transform_point(self, point: list[int, int]) -> list[float, float]:
        point[0] = point[0] - 500
        point[0] = point[0] / 41.42

        point[1] = point[1] - 500
        point[1] = point[1] / 41.42

        return point
          
    
    def get_goalpoints(self, msg: Int32MultiArray):
        self.first_point = [msg.data[0], msg.data[1]]
        self.second_point = [msg.data[2], msg.data[3]]

        self.first_point = self.transform_point(self.first_point)
        self.second_point = self.transform_point(self.second_point)

        self.destroy_subscription(self.random_points_sub)

    def pid(self):

        #write your PID algorithm here. This time write equations for throttle, pitch, roll and yaw. 
        #Follow the steps from task 1b.

        '''PID control function to calculate control inputs and send them to the drone'''

        # Calculate position errors (difference between current and target positions)
        self.pid_pos_error = np.subtract(self.drone_position, self.setpoint)
        self.sum_pos_error = np.clip(self.sum_pos_error,-6.0,6.0)
        # PID components for roll, pitch, and throttle
        P = self.Kp * self.pid_pos_error  # Proportional term
        I = self.Ki * self.sum_pos_error  # Integral term
        D = self.Kd * (self.pid_pos_error - self.prev_pos_error)  # Derivative term
        beta = 0.3
        self.D_new  = beta * D + (1 - beta) * self.D_new
        # Compute yaw control value
        # Compute roll, pitch, throttle, and yaw control values
        rc_roll = 1500 - (P[0] + I[0] + D[0])
        rc_pitch = 1500 + P[1] + I[1] + D[1]
        rc_throttle = self.hover_throttle + P[2] + I[2] + self.D_new[2]
        rc_yaw = 1500  # Fixed yaw (can be updated later)

        # Clip values to the defined min and max limits
        rc_roll = np.clip(rc_roll, self.min_values['roll'], self.max_values['roll'])
        rc_pitch = np.clip(rc_pitch, self.min_values['pitch'], self.max_values['pitch'])
        rc_throttle = np.clip(rc_throttle, self.min_values['throttle'], self.max_values['throttle'])
        rc_yaw = np.clip(rc_yaw, self.min_values['yaw'], self.max_values['yaw'])

        # Update previous errors and accumulate the sum of errors for the integral term
        self.prev_pos_error = self.pid_pos_error
        self.sum_pos_error += self.pid_pos_error

        # Update the command message with computed control values
        self.cmd.rc_roll = int(rc_roll)
        self.cmd.rc_pitch = int(rc_pitch)
        self.cmd.rc_throttle = int(rc_throttle)
        self.cmd.rc_yaw = int(rc_yaw)

        # Update PID error message
        self.pid_error.roll_error = self.pid_pos_error[0]
        self.pid_error.pitch_error = self.pid_pos_error[1]
        self.pid_error.throttle_error = self.pid_pos_error[2]
 
        self.command_pub.publish(self.cmd)
        self.pid_error_pub.publish(self.pid_error)

    def execute_callback(self, goal_handle):

        self.get_logger().info('Executing goal...')
        self.setpoint[0] = goal_handle.request.waypoint.position.x
        self.setpoint[1] = goal_handle.request.waypoint.position.y
        self.setpoint[2] = goal_handle.request.waypoint.position.z
        self.get_logger().info(f'New Waypoint Set: {self.setpoint}')
        self.max_time_inside_sphere = 0
        self.point_in_sphere_start_time = None
        self.time_inside_sphere = 0
        self.duration = self.dtime

        goal_flag = False
        # if self.first_point is not None:
        #     goal_flag = self.is_drone_in_sphere(self.first_point, goal_handle, 0.4)
        


        #create a NavToWaypoint feedback object. Refer to Writing an action server and client (Python) in ROS 2 tutorials.

        feedback_msg = NavToWaypoint.Feedback()

        #--------The script given below checks whether you are hovering at each of the waypoints(goals) for max of 3s---------#
        # This will help you to analyse the drone behaviour and help you to tune the PID better.

        while True:
            feedback_msg.current_waypoint.pose.position.x = self.drone_position[0]
            feedback_msg.current_waypoint.pose.position.y = self.drone_position[1]
            feedback_msg.current_waypoint.pose.position.z = self.drone_position[2]
            feedback_msg.current_waypoint.header.stamp.sec = self.max_time_inside_sphere

            goal_handle.publish_feedback(feedback_msg)

            drone_is_in_sphere = self.is_drone_in_sphere(self.drone_position, goal_handle, self.stable_error) #the value '0.4' is the error range in the whycon coordinates that will be used for grading. 
            #You can use greater values initially and then move towards the value '0.4'. This will help you to check whether your waypoint navigation is working properly. 
            # error1 = self.euclidian_distance(self.drone_position,self.first_point)
            # error2 = self.euclidian_distance( self.drone_position,self.second_point)
            # if error1<self.stable_error or error2<self.stable_error:
            #       self.stablise_time = 4.0
            #       break 
            
            if drone_is_in_sphere and self.point_in_sphere_start_time is None :
                  break
            
            if not drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        pass
            
            elif drone_is_in_sphere and self.point_in_sphere_start_time is None:
                        self.point_in_sphere_start_time = self.dtime
                        # self.stablise_time = 4.0
                        self.get_logger().info('Drone in sphere for 1st time')                        #you can choose to comment this out to get a better look at other logs
                        
            elif drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        self.time_inside_sphere = self.dtime - self.point_in_sphere_start_time
                        # self.stablise_time = 4.0
                        self.get_logger().info('Drone in sphere')                                     #you can choose to comment this out to get a better look at other logs
                             
            elif not drone_is_in_sphere and self.point_in_sphere_start_time is not None:
                        # self.stablise_time = 4.0
                        self.get_logger().info('Drone out of sphere')                                 #you can choose to comment this out to get a better look at other logs
                        self.point_in_sphere_start_time = None

            if self.time_inside_sphere > self.max_time_inside_sphere:
                 self.max_time_inside_sphere = self.time_inside_sphere

            if self.max_time_inside_sphere >= self.stablise_time:
                 self.stablise_time = 2.0
                 break

            if(self.first_point[0]==goal_handle.request.waypoint.position.x
            and self.first_point[1]==goal_handle.request.waypoint.position.y
            ):
                self.stablise_time = 4.0
            elif(self.second_point[0]==goal_handle.request.waypoint.position.x
            and self.second_point[1]==goal_handle.request.waypoint.position.y
            ):
                self.stablise_time = 4.0
            else:
                self.stablise_time = 2.0        

        goal_handle.succeed()

        #create a NavToWaypoint result object. Refer to Writing an action server and client (Python) in ROS 2 tutorials
        self.prev_setpoint = self.setpoint

        result = NavToWaypoint.Result()
        result.hov_time = self.dtime - self.duration #this is the total time taken by the drone in trying to stabilize at a point
        return result
    
    def is_drone_in_sphere(self, drone_pos, sphere_center, radius):
        return (
            (drone_pos[0] - sphere_center.request.waypoint.position.x) ** 2
            + (drone_pos[1] - sphere_center.request.waypoint.position.y) ** 2
            + (drone_pos[2] - sphere_center.request.waypoint.position.z) ** 2
        ) <= radius**2
    # def euclidian_distance(self, point1, point2):
    #     return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 )

def main(args=None):
    rclpy.init(args=args)

    waypoint_server = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(waypoint_server)
    
    try:
         executor.spin()
    except KeyboardInterrupt:
        waypoint_server.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
         waypoint_server.destroy_node()
         rclpy.shutdown()


if __name__ == '__main__':
    main()