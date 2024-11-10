#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints
from std_msgs.msg import Int32MultiArray
import cv2 as cv
import numpy as np
from scipy.spatial import distance
import heapq
import matplotlib.pyplot as plt

class WayPoints(Node):

    def __init__(self, debug = False):
        super().__init__('waypoints_service')
        self.bit_map_path = '/home/loki/pico_ws/src/swift_pico/scripts/2D_bit_map.png'
        

        self.get_logger().info('Waypoints service started') 

        self.random_points_sub = self.create_subscription(Int32MultiArray, '/random_points', self.get_waypoints, 10)

        self.initial_point = (500, 500)
        self.first_point = None
        self.second_point = None
        self.step = 1

        self.scale = 1
        self.debug = debug
        self.paths = []

        img = cv.imread(self.bit_map_path, cv.IMREAD_GRAYSCALE)
        # img = cv.bitwise_not(img)
        # img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)
        try:
            self.height, self.width, _ = img.shape
        except:
            self.height, self.width = img.shape

        self.resized_img = (img) 
        self.srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)
        self.waypoints = [[2.0, 2.0, 27.0], [2.0, -2.0, 27.0], [-2.0, -2.0, 27.0], [-2.0, 2.0, 27.0], [1.0, 1.0, 27.0]]

    def adjust_scale(self):
        self.resized_img = cv.resize(self.resized_img, (int(self.width * self.scale), int(self.height * self.scale)), interpolation=cv.INTER_NEAREST)
        try:
            self.height, self.width, _ = self.resized_img.shape
        except:
            self.height, self.width = self.resized_img.shape
        self.initial_point = (int(self.initial_point[0] * self.scale), int(self.initial_point[1] * self.scale))
        self.first_point = (int(self.first_point[0] * self.scale), int(self.first_point[1] * self.scale))
        self.second_point = (int(self.second_point[0] * self.scale), int(self.second_point[1] * self.scale))

    def de_adjust_scale(self):
        for iu in range(len(self.paths)):
            path = self.paths[iu]
            for i in range(len(path)):
                path[i] = (int(path[i][1] / self.scale), int(path[i][0] / self.scale))
            self.paths[iu] = path

        self.initial_point = (int(self.initial_point[0] / self.scale), int(self.initial_point[1] / self.scale))
        self.first_point = (int(self.first_point[0] / self.scale), int(self.first_point[1] / self.scale))
        self.second_point = (int(self.second_point[0] / self.scale), int(self.second_point[1] / self.scale))

        self.resized_img = cv.resize(self.resized_img, (int(self.width / self.scale), int(self.height / self.scale)), interpolation=cv.INTER_NEAREST)
        try:
            self.height, self.width, _ = self.resized_img.shape
        except:
            self.height, self.width = self.resized_img.shape

    def scale_path(self, scale):
        for iu in range(len(self.paths)):
            path = self.paths[iu]
            for i in range(len(path)):
                path[i] = ((path[i][1] * scale), (path[i][0] * scale))
            self.paths[iu] = path
        
    def translate_path(self):
        for iu in range(len(self.paths)):
            path = self.paths[iu]
            for i in range(len(path)):
                path[i] = (path[i][0] - self.initial_point[0], path[i][1] - self.initial_point[1])
            self.paths[iu] = path

    def get_waypoints(self, msg: Int32MultiArray):
        self.first_point = (msg.data[0], msg.data[1])
        self.second_point = (msg.data[2], msg.data[3])

        self.adjust_scale()

        self.get_logger().info(f"Received random points. \nStart point: {self.first_point}, Finish point: {self.second_point}")

        path1 = self.get_trajectory(self.initial_point, self.first_point)
        path2 = self.get_trajectory(self.first_point, self.second_point)

        if path1 is None or path2 is None:
            print("No path found between the points.")
            return
        self.paths.append(path1)
        self.paths.append(path2)

        print(len(self.paths))
        # Plot the path on the image
        if len(self.paths):
            path = []
            for i in range(len(self.paths)):
                path += self.paths[i]
            print("Path found between the points.")
            if self.debug:
                plt.imshow(self.resized_img, cmap='gray')
                plt.plot(self.initial_point[0], self.initial_point[1], 'go', label='Start')  # Start in green
                plt.plot(self.first_point[0], self.first_point[1], 'yo', label='First')  # Start in green
                plt.plot(self.second_point[0], self.second_point[1], 'ro', label='Second')   # Goal in red
                path_x, path_y = zip(*path)
                plt.plot(path_y, path_x, 'b-', linewidth=2, label='Path')  # Path in blue
                plt.legend()
                plt.savefig("/home/loki/pico_ws/src/swift_pico/scripts/path.png")
                plt.show()
        else:
            print("No path found between the points.")
        self.destroy_subscription(self.random_points_sub)
        self.translate_path()
        self.scale_path(self.scale / 40)
        # self.de_adjust_scale()



    def get_trajectory(self, first_point, second_point):
        resized_img = (self.resized_img == 255).astype(int)
        resized_img_blur = cv.blur(self.resized_img, (100, 100))
        resized_img_blur = cv.blur(resized_img_blur, (100, 100))
        # cv.imshow("Arena", resized_img_blur)
        # if cv.waitKey(0) & 0xFF == ord('q'):
        #     raise KeyboardInterrupt
        rows, cols = self.height, self.width
        start_point = (first_point[1], first_point[0])
        finish_point = (second_point[1], second_point[0])
        open_set = [(0, start_point)]
        heapq.heapify(open_set)
        came_from = {}
        g_score = {start_point: 0}
        f_score = {start_point: distance.euclidean(start_point, finish_point)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == finish_point:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_point)
                path.reverse()
                return path
            for dx, dy in [(-self.step, 0), (self.step, 0), (0, -self.step), (0, self.step)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and resized_img[neighbor[0], neighbor[1]] == 1:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + distance.euclidean(neighbor, finish_point) - resized_img_blur[neighbor[0], neighbor[1]]
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None  # No path found
        
    def waypoint_callback(self, request, response):
        
        if not self.paths:
            return 
        self.waypoints = []
        for i in range(len(self.paths)):
                self.waypoints += self.paths[i]
        for i in range(len(self.waypoints)):
            self.waypoints[i] = (float(self.waypoints[i][0]), float(self.waypoints[i][1]), 27.0)

        ways = []
        for i in range(0, len(self.waypoints), 10):
            ways += [self.waypoints[i]]
        self.waypoints = ways
        if request.get_waypoints == True :
            response.waypoints.poses = [Pose() for _ in range(len(self.waypoints))]
            for i in range(len(self.waypoints)):
                response.waypoints.poses[i].position.x = self.waypoints[i][0]
                response.waypoints.poses[i].position.y = self.waypoints[i][1]
                response.waypoints.poses[i].position.z = self.waypoints[i][2]
            self.get_logger().info("Incoming request for Waypoints")
            return response

        else:
            self.get_logger().info("Request rejected")

def main():
    rclpy.init()
    waypoints = WayPoints(debug=True)

    try:
        rclpy.spin(waypoints)
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
        

        