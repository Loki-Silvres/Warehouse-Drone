import cv2 as cv
import numpy as np
import os


points = []
img = None
distance = 0

def tap_callback(event, x, y, flags, param):
    global points, img, distance
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(x, y)
        points.append((x, y))
        cv.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv.imshow("image", img)
    if len(points) >= 2:
        cv.line(img, points[-2], points[-1], (0, 255, 0), 2)
        cv.imshow("image", img)
        distance = [(points[-1][0] - points[-2][0]), (points[-1][1] - points[-2][1])]
        points.clear()
        print(distance)
    

def main():
    global img, points, distance
    overlays_path = "photos/overlays/"
    img_paths = os.listdir(overlays_path)
    dist = {}
    for img_name in img_paths:
        img_path = f"photos/overlays/{img_name}"
        img = cv.imread(img_path)
        print(img_name)
        cv.namedWindow("image")
        cv.setMouseCallback("image", tap_callback)
        cv.imshow("image", img)
        if cv.waitKey(0) & 0xFF == ord("q"):
            break
        cv.destroyAllWindows()
        dist[img_name] = distance
    print(dist)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        cv.destroyAllWindows()