import numpy as np
import cv2
import math


class ShapeAnalyzer:
    def __init__(self):
        self.centroid = None
        self.cont_len = None

    def make_contour(self, mask):
        """
        returns countour of a given mask
        mask has to be uniform, without multiple blobs
        :param mask: can be boolean
        :return: contour - OpenCV contour; list of contour points locations (x, y)
        """
        mask = mask.astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]
        self.get_contour_parameters(contour)
        return contour

    def analyze_shape(self, mask):
        contour = self.make_contour(mask)
        self.draw_contour(contour, (mask.shape[0], mask.shape[1]))
        contour = self.decimate_contour(contour)
        grasp_points = self.get_grasp_points(contour)

    @staticmethod
    def draw_contour(contour, img_size):
        """
        draws given contour on an empty image
        """
        img = np.zeros(img_size, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
        cv2.imshow("contour", img)
        cv2.waitKey()
        return img

    def get_contour_parameters(self, contour):
        """
        gets contour parameters used in shape analysis and saves them in class-level holders
        """
        # contour moments
        M = cv2.moments(contour)
        # contour centroid (c_x, c_y)
        self.centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        self.cont_len = len(contour)

    def decimate_contour(self, contour):
        """
        reduces number of contour points uniformly
        """
        sample_rate = 5
        return contour[::sample_rate]

    def get_grasp_points(self, contour):
        """
        returns a list of all potential grasp points pairs
        """
        grasp_points = []
        # make local variable for speed
        cont_len = self.cont_len
        min_pair_distance = int(cont_len / 5)

        for i in range(cont_len - min_pair_distance):
            for j in range(i + min_pair_distance, min((cont_len, i + cont_len - min_pair_distance + 1))):
                grasp_points.append((contour[i], contour[j]))
        return grasp_points

    def get_centroid_distance(self, points, centroid):
        """
        returns distance between center of grasp and object centroid
        :param points: pair of points (x,y) (tuple or list) defining grasp points
        :param centroid: contour centroid (x,y)
        """

        grasp_center = ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2)
        distance = ((grasp_center[0] - centroid[0])**2 + (grasp_center[1] - centroid[1])**2)**0.5
        return distance

    def get_grasp_points_distance(self, points):
        """
        returns distance between 2 given points
        """
        distance = ((points[0][0] - points[1][0])**2 + (points[0][0] - points[1][0])**2)**0.5
        return distance

    def get_grasp_orientation(self, points):
        """
        returns grasp orientation in radians (-pi/2, pi/2]
        """
        line = (points[0][0] - points[1][0], points[0][1] - points[1][1])
        if line[0] == 0:
            # avoid division by 0
            orientation = math.pi / 2
        else:
            # no need for atan2
            orientation = math.atan(line[1] / line[0])
        return orientation
