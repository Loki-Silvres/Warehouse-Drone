#!/usr/bin/env python3

import copy
import cv2
import numpy as np
import cv2.aruco as aruco
import argparse  
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BitMap(Node):

    def __init__(self, debug = False):
        super().__init__('bit_map_node')
        self.get_logger().info('BitMap node started')
        self.width = 1000
        self.height = 1000
        self.detected_markers = []
        self.obstacles = 0
        self.total_area = 0
        self.detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
                            aruco.DetectorParameters())
        
        self.img_sub = self.create_subscription(Image, 
                                    '/image_raw', 
                                    self.image_callback, 
                                    10)
        
        self.bridge = CvBridge()
        self.cv_image = None
        self.inflation_pixels = 10
        # self.inflation_pixels_2 = 100
        self.drone_area = 200
        self.debug = debug

        self.get_bitmap_timer = self.create_timer(1, self.identification)
        

    def image_callback(self, msg: Image):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def identification(self):
        if self.cv_image is None:
            print("Waiting for CV image.")
            return
        frame = self.cv_image
        corners, ids, reject = self.detector.detectMarkers(frame)
        self.detected_markers = ids.flatten().tolist()
        corners = np.array(corners)
        ids = np.array(ids)
        top_left = corners[ids == 80][-1][0]
        bottom_right = corners[ids == 90][-1][2]
        top_right = corners[ids == 85][-1][1]
        bottom_left = corners[ids == 95][-1][3]

        inner_corners = np.float32([top_left, bottom_right, top_right, bottom_left])

        new_top_left = (0,0)
        new_bottom_right = (self.width, self.height)
        new_top_right = (self.width, 0)
        new_bottom_left = (0, self.height)
        new_corners = np.float32([new_top_left, new_bottom_right, new_top_right, new_bottom_left])

        M = cv2.getPerspectiveTransform(inner_corners, new_corners)
        transformed_image = cv2.warpPerspective(frame, M, (self.width, self.height))

        if self.debug:
            cv2.imshow("Arena", transformed_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        inflated_image = transformed_image.copy()

        gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, threshold1 = 50, threshold2= 100)

        border_thickness = self.inflation_pixels  
        bordered_img = cv2.copyMakeBorder(canny, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=255)
        contours, hierarchy = cv2.findContours(bordered_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        mask = np.ones_like(gray, dtype=np.uint8) * 255
        mask2 = copy.deepcopy(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.drone_area and area < 900000:
                if self.debug:
                    print(area)

                self.obstacles += 1
                self.total_area += area
                temp_mask = np.zeros_like(gray, dtype=np.uint8)  # Black background for dilation
                cv2.drawContours(temp_mask, [contour], -1, 255, thickness=cv2.FILLED)  # Draw filled contour on temp mask
                inflated_mask = cv2.dilate(temp_mask, np.ones((self.inflation_pixels, self.inflation_pixels), np.uint8))
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(inflated_mask))

                # inflated_mask_2 = cv2.dilate(temp_mask, np.ones((self.inflation_pixels_2, self.inflation_pixels_2), np.uint8))
                # mask2 = cv2.bitwise_and(mask2, cv2.bitwise_not(inflated_mask_2))


        if self.debug:
            cv2.imshow("Bit Image of Arena", mask)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                raise KeyboardInterrupt
        img_write_path = "/home/loki/pico_ws/src/swift_pico/scripts/2D_bit_map.png"
        print(f" Image saved succesfully: {cv2.imwrite(img_write_path, mask)}")
        # cv2.imwrite(img_write_path.replace(".png", "_2.png"), mask)
        # print(f"Timer destroyed: {self.destroy_timer(self.get_bitmap_timer)}")

    def text_file(self):
        with open("obstacles.txt", "w") as file:
            file.write(f"Aruco ID: {self.detected_markers}\n")
            file.write(f"Obstacles: {self.obstacles}\n")
            file.write(f"Area: {self.total_area}\n")

if __name__ == '__main__':

    rclpy.init()
    map = BitMap(debug = True)

    try:
        rclpy.spin(map)
    except KeyboardInterrupt:
        map.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        map.destroy_node()
        rclpy.shutdown()
    