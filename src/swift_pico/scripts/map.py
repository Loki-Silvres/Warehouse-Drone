import cv2
import numpy as np
import cv2.aruco as aruco
import argparse  

class Arena:

    def __init__(self, image_path):
        self.width = 1000
        self.height = 1000
        self.image_path = image_path
        self.detected_markers = []
        self.obstacles = 0
        self.total_area = 0
        self.detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_1000),
                            aruco.DetectorParameters())

    def identification(self):
        frame = cv2.imread(self.image_path)
        corners, ids, reject = self.detector.detectMarkers(frame)
        # print(np.shape(corners))
        # print(corners)
        # print(ids)
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

        gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, threshold1 = 50, threshold2= 100)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            
            if area > 100:
                self.obstacles += 1
                self.total_area += area


    def text_file(self):
        with open("obstacles.txt", "w") as file:
            file.write(f"Aruco ID: {self.detected_markers}\n")
            file.write(f"Obstacles: {self.obstacles}\n")
            file.write(f"Area: {self.total_area}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aruco Marker Detection and Obstacle Area Calculation')
    parser.add_argument('--image', required=True)

    args = parser.parse_args()
    image_path = args.image
    arena = Arena(image_path)
    arena.identification()
    arena.text_file()
