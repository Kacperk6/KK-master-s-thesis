import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class ShapeAnalyzer:
    def __init__(self, cam_mat):
        self.centroid = None
        self.cont_len = None

        # focal length of a camera in px, used for pixel-millimeter conversion
        # self.cam_focal_length = self.get_camera_focal_length(cam_mat)

        self.size_factor = None

    def make_contour(self, mask, distance):
        """
        returns countour of a given mask
        mask has to be uniform, without multiple blobs
        :param mask: can be boolean
        :return: contour - OpenCV contour; list of contour points locations (x, y)
        """
        def get_largest_contour(contours):
            """
            in case of (rare) multiple contours, return largest one
            """
            largest_area = 0
            largest_contour = None
            for contour in contours:
                if cv2.contourArea(contour) > largest_area:
                    largest_contour = contour
            return largest_contour
        mask = mask.astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = get_largest_contour(contours)
        contour = self.decimate_contour(contour)
        contour = self.scale_contour(contour, distance)
        self.get_contour_parameters(contour)
        return contour

    def analyze_shape(self, mask, distance):
        contour = self.make_contour(mask, distance)
        self.draw_contour(contour, (int(mask.shape[0]*self.size_factor), int(mask.shape[1]*self.size_factor)))
        grasp_points_idx_all = self.get_grasp_points_idx(contour)
        self.save(contour, grasp_points_idx_all)
        for i in range(len(grasp_points_idx_all)):
            pass

    def analyze_shape_test(self):
        contour, grasp_points_idx_all = self.load_data()
        # contour_x = contour[:, 0, 0]
        # contour_y = contour[:, 0, 1]
        # grasp_points_x = grasp_points[:, 0, 0]
        # grasp_points_y = grasp_points[:, 0, 1]
        for i in range(len(grasp_points_idx_all)):
            grasp_points_idx = grasp_points_idx_all[i]

            # # display contour with line between grasp points
            # _ = plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'b.',
            #              [contour[grasp_points_idx[0]][0][0], contour[grasp_points_idx[1]][0][0]],
            #              [contour[grasp_points_idx[0]][0][1], contour[grasp_points_idx[1]][0][1]], 'r-')
            #
            # plt.show()

            self.evaluate_grasp_points_orientation(contour, grasp_points_idx)

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

    def get_grasp_points_idx(self, contour):
        """
        returns a list of indexes of all potential grasp point pairs
        """
        grasp_points_idx = []
        # make local variable for speed
        cont_len = self.cont_len
        min_pair_distance = int(cont_len / 5)

        for i in range(cont_len - min_pair_distance):
            for j in range(i + min_pair_distance, min((cont_len, i + cont_len - min_pair_distance + 1))):
                grasp_points_idx.append((i, j))
        return grasp_points_idx

    def get_centroid_distance(self, points, centroid):
        """
        returns distance between center of grasp and object centroid
        :param points: pair of points (x,y) (tuple or list) defining grasp points
        :param centroid: contour centroid (x,y)
        """

        grasp_center = ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2)
        distance = ((grasp_center[0] - centroid[0])**2 + (grasp_center[1] - centroid[1])**2)**0.5
        return distance

    def get_points_distance(self, points):
        """
        returns distance between 2 given points
        """
        distance = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)**0.5
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

    def scale_contour(self, contour, distance):
        """
        scales given contour from pixels to millimeters, based on size factor = object distance / camera focal length
        """
        size_factor = distance / self.cam_focal_length
        self.size_factor = size_factor
        return (contour * size_factor).astype('int')

    def get_camera_focal_length(self, cam_mat):
        """
        returns focal length of a camera in px, read from given camera matrix
        focal length is mean of x and y focal lengths, as they are usually pretty much the same
        """
        f_x = cam_mat[0][0]
        f_y = cam_mat[1][1]
        f = (f_x + f_y) / 2
        return f

    def evaluate_grasp_points_orientation(self, contour, grasp_points_idx):
        """
        evaluates curves surrounding given contour points for their orientation and potential grasp width change
        :return: total orientation error [0, 1], where 0 is perfectly fit to grasp
                 additional grasp width [mm] - a value to add to original grasp width
        """
        def evaluate_grasp_point_orientation(contour, point_idx, grasp_vector):
            """
            fits line into curve surrounding given grasp point, checks it's orientation against gripper
            and corrects grasp width
            :return: fitted line absolute orientation error [0, pi/2]
                     additional grasp width [mm]
            """
            gripper_size = 20

            def get_curve_points():
                """
                returns points surrounding contour[point_idx] within gripper_size,
                projected into UV (gripper) coordinate system
                """
                def project_point(point, coord_sys):
                    """
                    returns point projected to given coordinate system (u,v, origin(x,y))
                    """
                    return np.array([(point - coord_sys[2]) @ coord_sys[0], (point - coord_sys[2]) @ coord_sys[1]])
                # middle curve point in original XY coord system
                curve_points = contour[point_idx][0]
                # curve point projected into UV coord system
                curve_points_uv_new = np.array([project_point(curve_points, uv_coord_system)])
                # backup array is needed, as the last one to pass test is returned
                curve_points_uv = curve_points_uv_new
                i = 1
                # check value for contour length
                curve_length_prev = 0
                while True:
                    # a circular iterator (itertools.cycle wasn't enough)
                    last_idx = point_idx - i
                    if last_idx < 0:
                        last_idx = len(contour) + last_idx
                    first_idx = point_idx + i
                    if first_idx >= len(contour):
                        first_idx = first_idx - len(contour)

                    point_last = contour[last_idx][0]
                    point_last_uv = project_point(point_last, uv_coord_system)
                    point_first = contour[first_idx][0]
                    point_first_uv = project_point(point_first, uv_coord_system)

                    curve_points_uv_new = np.insert(curve_points_uv_new, 0, point_last_uv, axis=0)
                    curve_points_uv_new = np.append(curve_points_uv_new, [point_first_uv], axis=0)
                    # _ = plt.plot(curve_points_new[:, 0], curve_points_new[:, 1], 'b.')
                    # plt.show()

                    # length of U axis points projection
                    curve_length = abs(curve_points_uv_new[len(curve_points_uv_new) - 1][0] - curve_points_uv_new[0][0])
                    # stop adding points when length of curve projection (onto gripper) is greater than gripper size
                    # or smaller than previous value (which happens when curve begins to close)
                    if curve_length > gripper_size or curve_length < curve_length_prev:
                        return curve_points_uv
                    else:
                        curve_length_prev = curve_length
                        curve_points_uv = curve_points_uv_new
                        i += 1

            # local coord system definition
            v = grasp_vector
            u = np.array([v[1], -v[0]])
            origin = (contour[point_idx][0])
            uv_coord_system = (u, v, origin)

            # # display local coordinate system
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.set_aspect(aspect=1)
            # ax.plot([p_0[0], p_1[0]], [p_0[1], p_1[1]], 'b.-',
            #         [origin[0], origin[0] + u[0]], [origin[1], origin[1] + u[1]], 'r-',
            #         [origin[0], origin[0] + v[0]], [origin[1], origin[1] + v[1]], 'g-')
            # plt.show()

            # points to fit line to, in UV coordinate system
            curve_points = get_curve_points()
            # line fit to points (least squares method)
            line_coefficients = np.polyfit(curve_points[:, 0], curve_points[:, 1], 1)

            # # display line fit to curve in local coordinate system
            # line_eq = np.poly1d(line_coefficients)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.set_aspect(aspect=1)
            # ax.plot(curve_points[:, 0], curve_points[:, 1], 'b.',
            #         [curve_points[0][0], curve_points[len(curve_points) - 1][0]],
            #         [line_eq(curve_points[0][0]), line_eq(curve_points[len(curve_points) - 1][0])], 'r-')
            # plt.show()

            # absolute error in fit line orientation [0, pi/2]
            orientation_err = abs(math.atan(line_coefficients[0]))
            # additional distance from given middle point to actual highest (in UV system) point of the curve
            # used to correct grasp parameters
            add_grasp_width = curve_points[:, 1].max()
            return orientation_err, add_grasp_width

        # grasp points given by indexes
        p_0 = contour[grasp_points_idx[0]][0]
        p_1 = contour[grasp_points_idx[1]][0]
        # vector between given grasp points
        grasp_points_vector = p_1 - p_0
        # normalized grasp points vector (length = 1); not really necessary, but aesthetic
        grasp_points_vector_norm = grasp_points_vector / self.get_points_distance((p_0, p_1))

        # get absolute orientation error of lines fit to curves about both grasp points
        # and additional grasp width due to curve shape
        # local coordinate system bonded with gripper is used for line fitting
        # local coord sys. vertical axis points outward contour, hence sign at grasp vector parameter
        orientation_err_0, add_grasp_width_0 = evaluate_grasp_point_orientation(contour, grasp_points_idx[0],
                                                                    -grasp_points_vector_norm)
        orientation_err_1, add_grasp_width_1 = evaluate_grasp_point_orientation(contour, grasp_points_idx[1],
                                                                    grasp_points_vector_norm)
        # total orientation error, scaled to [0, 1] range for easier weight setting
        orientation_err = (orientation_err_0 + orientation_err_1) / math.pi
        # total grasp width update
        add_grasp_width = add_grasp_width_0 + add_grasp_width_1
        

    def save(self, contour, grasp_points_idx_all):
        np.savez_compressed('shape_analysis', contour=contour, grasp_points_idx_all=grasp_points_idx_all)

    def load_data(self):
        data = np.load('shape_analysis.npz')
        contour = data['contour']
        grasp_points_idx_all = data['grasp_points_idx_all']
        self.get_contour_parameters(contour)
        return contour, grasp_points_idx_all


shape_analyzer = ShapeAnalyzer(None)
shape_analyzer.analyze_shape_test()
