import numpy as np
import cv2


class ShapeAnalyzer:
    def __init__(self):
        self.contour = None
        self.centroid = None

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

    def get_contour_parameters(self):
        """
        gets contour parameters used in shape analysis and saves them in class-level holders
        """
        # contour moments
        M = cv2.moments(self.contour)
        # contour centroid (c_x, c_y)
        self.centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

    def get_centroid_distance(self, points):
        """
        returns distance between center of grasp and object centroid
        :param points: pair of points (x,y) (tuple or list) defining grasp points
        """
        grasp_center = ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2)
        distance = ((grasp_center[0] - self.centroid[0])**2 + (grasp_center[1] - self.centroid[1])**2)**0.5
        return distance
