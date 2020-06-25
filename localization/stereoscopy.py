import numpy as np
import cv2


DEPTH_VISUALIZATION_SCALE = 2048


class Stereoscopy:
    def __init__(self):
        self.stereo = cv2.StereoBM_create()

    def run(self, img_l, img_r):
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img_l", img_l)
        cv2.imshow("img_r", img_r)

        disparity = self.stereo.compute(img_l, img_r)
        cv2.imshow("depth", disparity/DEPTH_VISUALIZATION_SCALE)
        return disparity
