import numpy as np
import cv2


DEPTH_VISUALIZATION_SCALE = 2048


def nothing(x):
    """
    just for calibration trackbars
    """
    pass


class Stereoscopy:
    def __init__(self, calibration_mode=False):
        self.calibration_mode = calibration_mode
        self.stereo = cv2.StereoBM_create()
        self.stereo.setMinDisparity(4)
        self.stereo.setNumDisparities(128)
        self.stereo.setBlockSize(21)
        self.stereo.setSpeckleRange(16)
        self.stereo.setSpeckleWindowSize(45)
        cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)
        if self.calibration_mode:
            pass
            #cv2.createTrackbar("edge minVal", "processed img", EDGE_MIN_VAL, 255, nothing)

    def run(self, img_l, img_r):
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img_l", img_l)
        cv2.imshow("img_r", img_r)

        disparity = self.stereo.compute(img_l, img_r)
        cv2.imshow("depth", disparity/DEPTH_VISUALIZATION_SCALE)
        return disparity
