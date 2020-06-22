import numpy as np
import cv2
from matplotlib import pyplot as plt

from utils import camera


class Stereoscopy:
    def __init__(self):
        pass

    def run(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l, img_r = camera.split_stereo_image(img_gray, img_gray.shape[0], img_gray.shape[1])
        stereo = cv2.StereoBM_create(16, 15)
        disparity = stereo.compute(img_l, img_r)
        return disparity

    def disparity2img(self, disparity):
        pass
