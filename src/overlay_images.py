import cv2 as cv
import numpy as np


def overlay_images(img1, img2):
    return cv.addWeighted(img1, 0.5, img2, 0.5, 0)

if __name__ == "__main__":
    for i in range(0, 6):
        for j in range(i, 6):
            img1 = cv.imread(f"photos/captures/{i}.jpeg")
            img2 = cv.imread(f"photos/captures/{j}.jpeg")
            overlay = overlay_images(img1, img2)
            cv.imwrite(f"photos/overlays/overlay{str(i)+str(j)}.png", overlay)
    # id1 = "0"
    # id2 = "1"
    # img1 = cv.imread(f"photos/{id1}.jpeg")
    # img2 = cv.imread(f"photos/{id2}.jpeg")

    # overlay = overlay_images(img1, img2)
    # cv.imwrite(f"photos/overlay{id1+id2}.png", overlay)