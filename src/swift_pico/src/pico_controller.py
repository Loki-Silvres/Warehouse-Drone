#!/usr/bin/env python3

'''
This python file runs a ROS 2 node named `pico_control`, which controls the position of the Swift Pico Drone.
This node publishes and subscribes to the following topics:

    Publications                 Subscriptions
    /drone_command                /whycon/poses
    /pid_error                    /throttle_pid
                                  /pitch_pid
                                  /roll_pid

Rather than using different variables, list structures are used to store related data (e.g., self.setpoint = [x, y, z]).
'''

from swift_msgs.msg import SwiftMsgs   # Importing SwiftMsgs for drone command
from geometry_msgs.msg import PoseArray  # Importing PoseArray for position data
from pid_msg.msg import PIDTune, PIDError  # PID messages for tuning and errors
import rclpy  # ROS2 Python library
import numpy as np  # For array handling and mathematical operations
import math
from rclpy.node import Node  # ROS2 node class
from nav_msgs.msg import Odometry

class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # Initialize the node with the name 'pico_controller'

        # Initializing drone position and orientation
        self.drone_position = [0.0, 0.0, 0.0]
        self.drone_position_ = [0.0, 0.0, 0.0]
        self.drone_orientation = [0.0, 0.0, 0.0]

        # Setpoints for x, y, and z position of the drone
        self.setpoint = [2, 2, 27] 
        # self.setpoint = [0, 0, 31]  
        self.cmd = SwiftMsgs()  # Command message for the drone
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # PID coefficients for roll, pitch, and throttle (Kp, Ki, Kd)
        self.Kp = np.array([20, 20, 20])
        self.Ki = np.array([0.1, 0.1, 0.1]) 
        self.Kd = np.array([500, 500, 400]) 

        # Variables to store PID errors and past values for integration/differentiation
        self.pid_pos_error = np.array([0, 0, 0]).astype(float)
        self.prev_pos_error = np.array([0, 0, 0]).astype(float)
        self.sum_pos_error = np.array([0, 0, 0]).astype(float)

        # Limits for the drone's control inputs
        self.max_values = {'roll': 1600, 'pitch': 1600, 'yaw': 2000, 'throttle': 2000}
        self.min_values = {'roll': 1000, 'pitch': 1000, 'yaw': 1000, 'throttle': 1000}
        self.hover_throttle = 1533  # Hover throttle value

        # Initializing the PID error message
        self.pid_error = PIDError()
        self.pid_error.yaw_error = 0.0
        self.pid_error.roll_error = 0.0
        self.pid_error.pitch_error = 0.0
        self.pid_error.throttle_error = 0.0

        # Variables to accumulate control inputs (for potential future use)
        self.rc_roll_sum = 0.0
        self.rc_pitch_sum = 0.0
        self.rc_throttle_sum = 0.0
        self.rc_yaw_sum = 0.0    

        self.sample_time = 0.060  # Sample time for the PID loop

        # Publisher for sending drone commands
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        # Publisher for sending PID error values
        self.pid_error_pub = self.create_publisher(PIDError, '/pid_error', 10)
        self.create_subscription(Odometry, '/rotors/odometry', self.odometry_callback, 10)
        # Subscriber to get the drone's position from the 'whycon' system
        self.create_subscription(PoseArray, '/whycon/poses', self.whycon_callback, 1)

        self.arm()  # Arm the drone (initially)
        # Create a timer to call the PID function periodically based on sample time
        self.create_timer(self.sample_time, self.pid)


    def disarm(self):
        '''Function to disarm the drone by setting all control inputs to minimum values'''
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)
        

    def arm(self):
        '''Function to arm the drone by setting control inputs to neutral values'''
        self.disarm()  # First, disarm the drone to reset inputs
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000  # Setting aux channel to arm
        self.command_pub.publish(self.cmd)  # Publishing the command to arm the drone

    def odometry_callback(self, msg: Odometry):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        

        self.drone_position[0] = 1.625 * ( - msg.pose.pose.position.y ) + 0.12 # x position
        self.drone_position[1] = 1.625 * ( - msg.pose.pose.position.x ) + 0.12 # y position
        self.drone_position[2] = 31.84 - msg.pose.pose.position.z * 1.75  # z (altitude) position
        self.dtime = msg.header.stamp.sec

        print(msg.pose.pose, self.drone_position)

    def whycon_callback(self, msg):
        '''Callback function to update the drone's position from the PoseArray message'''
        self.drone_position_[0] = msg.poses[0].position.x  # x position
        self.drone_position_[1] = msg.poses[0].position.y  # y position
        self.drone_position_[2] = msg.poses[0].position.z  # z (altitude) position

        # if self.is_drone_in_sphere_sp(self.drone_position, self.setpoint, 0.6):
        #     self.Kp = np.array([30, 30, 20])
        #     self.Kd = np.array([200, 200, 300])
        #     print("Drone in sphere")
        # else:
        #     self.Kp = np.array([45, 45, 45])
        #     self.Kd = np.array([500, 500, 400])
        #     print("Drone not in sphere")

    def is_drone_in_sphere_sp(self, drone_pos, sphere_center, radius):
        return (
            (drone_pos[0] - sphere_center[0]) ** 2
            + (drone_pos[1] - sphere_center[1]) ** 2
            + (drone_pos[2] - sphere_center[2]) ** 2
        ) <= radius**2

    def pid(self):
        '''PID control function to calculate control inputs and send them to the drone'''

        # Calculate position errors (difference between current and target positions)
        self.pid_pos_error = np.subtract(self.drone_position, self.setpoint)
        self.sum_pos_error = np.clip(self.sum_pos_error, -6.0, 6.0)

        # PID components for roll, pitch, and throttle
        P = self.Kp * self.pid_pos_error  # Proportional term
        I = self.Ki * self.sum_pos_error  # Integral term
        D = self.Kd * (self.pid_pos_error - self.prev_pos_error)  # Derivative term

        # Compute roll, pitch, throttle, and yaw control values
        rc_roll = 1500 - (P[0] + I[0] + D[0])
        rc_pitch = 1500 + P[1] + I[1] + D[1]
        rc_throttle = self.hover_throttle + P[2] + I[2] + D[2]
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

        print(self.pid_pos_error)
        print(self.cmd)

        # Publish command and error messages
        self.command_pub.publish(self.cmd)
        self.pid_error_pub.publish(self.pid_error)


def main(args=None):
    '''Main function to initialize the ROS2 node and keep it running'''
    rclpy.init(args=args)  # Initialize the ROS2 Python environment
    swift_pico = Swift_Pico()  # Create an instance of the Swift_Pico class
    rclpy.spin(swift_pico)  # Keep the node running
    swift_pico.disarm()  # Disarm the drone when the node is shut down
    swift_pico.destroy_node()  # Destroy the node
    rclpy.shutdown()  # Shut down the ROS2 Python environment


if __name__ == '__main__':
    main()
