import numpy as np
import cv2


class ShapeAnalyzer:
    def __init__(self):
        self.contour = None

    def make_contour(self, mask):
        """
        returns countour of a given mask
        mask has to be uniform, without multiple blobs
        :param mask: can be boolean
        :return: contour - OpenCV contour; list of contour points locations (x, y)
        """
        mask = mask.astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.contour = contours[0]
        return contours[0]

    @staticmethod
    def draw_contour(contour, img_size):
        """
        draws given contour on an empty image
        """
        img = np.zeros(img_size, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
        return img

