import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


class ShapeAnalyzer:
    def __init__(self, cam_mat):
        self.centroid = None
        self.cont_len = None
        self.x, self.y, self.width, self.height = None, None, None, None

        self.weights = {'surface_orientation': 1.0, 'inertia': 1, 'grasp_orientation': 0.2}

        if cam_mat is not None:
            # focal length of a camera in px, used for pixel-millimeter conversion
            self.cam_focal_length = self.get_camera_focal_length(cam_mat)
        else:
            # just for tests. and this number is real xD
            self.cam_focal_length = 420.3396289938836

    def make_contour(self, mask, position):
        """
        returns contour of a given mask
        mask has to be uniform, without multiple blobs
        :param mask: shape to analyze, can be boolean
        :param position: distance to object in Z axis; for scaling purpose
        :return: contour - OpenCV contour; list of contour points locations (x, y), scaled to millimeters
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
        # get contour of a given mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # handle multiple contours (vary rare case)
        if len(contours) == 1:
            contour = contours[0]
        else:
            contour = get_largest_contour(contours)
        # display contour with line between winner grasp points
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(aspect=1)
        ax.invert_yaxis()
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'k.')
        plt.show()
        # scale contour point locations from pixels to real size millimeters
        contour = self.scale_contour(contour, position)
        # # display contour with line between winner grasp points
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect(aspect=1)
        # ax.invert_yaxis()
        # ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'k.')
        # plt.show()
        # decrease number of contour points (uniformly) to decrease computational load
        contour = self.decimate_contour(contour)
        # display contour with line between winner grasp points
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(aspect=1)
        ax.invert_yaxis()
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'k.')
        plt.show()
        # store contour parameters for further use
        self.get_contour_parameters(contour)
        return contour

    def analyze_shape(self, mask, position):
        self.save(mask, position)
        # get a contour to analyze
        contour = self.make_contour(mask, position)
        # draw contour; real scale in millimeters
        # self.draw_contour(contour, (int(mask.shape[0]*self.size_factor), int(mask.shape[1]*self.size_factor)))

        # get index pairs of all contour points making potential grasp pairs
        grasp_points_idx_all = self.get_grasp_points_idx()

        # list to store grasp point pairs and all their scores of different categories
        grasp_points_dict_list = []
        # individual list elements as dictionaries
        for grasp_points_idx in grasp_points_idx_all:
            grasp_points_dict_list.append({'points': (contour[grasp_points_idx[0]][0], contour[grasp_points_idx[1]][0]),
                                           'idx': grasp_points_idx})

        # remove point pairs which lie on the ground, therefore are inaccessible
        grasp_points_dict_list = self.filter_bottom_points(grasp_points_dict_list)

        # get distances between each point pair
        for grasp_points_dict in grasp_points_dict_list:
            grasp_points_dict['distance'] = self.get_points_distance(grasp_points_dict['points'])

        # remove point pairs that don't fit into gripper
        grasp_points_dict_list = self.filter_grasp_points_distance(grasp_points_dict_list)
        if len(grasp_points_dict_list) == 0:
            return None, None, None

        # compute scores for different categories for each point pair
        centroid = self.centroid
        for grasp_points_dict in grasp_points_dict_list:
            grasp_points_dict['surface_orientation'], grasp_points_dict['distance'] \
                = self.evaluate_grasp_points_orientation(contour,
                                                         grasp_points_dict['idx'],
                                                         grasp_points_dict['distance'])

            grasp_points_dict['grasp_orientation'] = self.get_grasp_orientation(grasp_points_dict['points'])

            grasp_points_dict['inertia'] = self.get_inertia(grasp_points_dict['points'], centroid)

        # apply weights to scores
        weights = self.weights
        scores = []
        for grasp_points_dict in grasp_points_dict_list:
            score = 0
            for key in weights:
                score += grasp_points_dict[key] * weights[key]
            scores.append(score)
        # get point pair with lowest (best) score
        grasp_points_winner = grasp_points_dict_list[np.argmin(scores)]

        # display contour with line between winner grasp points
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect(aspect=1)
        ax.invert_yaxis()
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'k.',
                [point[0] for point in grasp_points_winner['points']],
                [point[1] for point in grasp_points_winner['points']], 'b-',
                self.centroid[0], self.centroid[1], 'r.')
        plt.show()

        grasp_parameters = self.get_grasp_parameters(grasp_points_winner, position[2])
        return grasp_parameters

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
        cv2.destroyWindow("contour")
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
        self.x, self.y, self.width, self.height = cv2.boundingRect(contour)

    def decimate_contour(self, contour):
        """
        if average distance between contour points is smaller than min_point_distance,
        reduces number of contour points uniformly
        """
        min_point_distance = 4

        def get_avg_point_distance(contour):
            distance = 0
            num_samples = len(contour) - 1
            for i in range(num_samples):
                points = (contour[i][0], contour[i+1][0])
                distance += self.get_points_distance(points)
            avg_distance = distance / num_samples
            return avg_distance

        avg_distance = get_avg_point_distance(contour)
        if avg_distance < min_point_distance:
            sample_rate = int(min_point_distance / avg_distance)
            if sample_rate != 0:
                contour = contour[::sample_rate]
        return contour

    def get_grasp_points_idx(self):
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

    @staticmethod
    def get_inertia(points, centroid):
        """
        returns relative inertia momentum, i.e. (just) quadratic distance between grasp center point
        and contour centroid; normalized, so reaches 1 at distance of 100 mm (value rises with distance^2)
        :param points: pair of points (x,y) (tuple or list) defining grasp points
        :param centroid: contour centroid (x,y)
        """

        grasp_center = ((points[0][0] + points[1][0])/2, (points[0][1] + points[1][1])/2)
        inertia = ((grasp_center[0] - centroid[0])**2 + (grasp_center[1] - centroid[1])**2)
        inertia_norm = inertia / 10000

        # # display distance between grasp center point and contour centroid
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect(aspect=1)
        # ax.invert_yaxis()
        # ax.plot(contour[:, 0, 0], contour[:, 0, 1], 'k.',
        #         [point[0] for point in points],
        #         [point[1] for point in points], 'b-',
        #         grasp_center[0], grasp_center[1], 'b.',
        #         centroid[0], centroid[1], 'r.',
        #         [grasp_center[0], centroid[0]], [grasp_center[1], centroid[1]], 'r-.')
        # plt.show()
        return inertia_norm

    @staticmethod
    def filter_grasp_points_distance(grasp_points_dict_list):
        """
        removes grasp point pairs that are too distant to fit into gripper
        """
        # maximal extension of gripper jaws
        grasp_size = 150
        # holder for grasp points indexes list length, for efficiency reasons
        grasp_points_idx_all_len = len(grasp_points_dict_list)
        i = 0
        while i < grasp_points_idx_all_len:
            distance = grasp_points_dict_list[i]['distance']
            if distance > grasp_size:
                grasp_points_dict_list.pop(i)
                grasp_points_idx_all_len -= 1
            else:
                i += 1
        return grasp_points_dict_list

    def filter_bottom_points(self, grasp_points_dict_list):
        """
        removes grasp point pairs, where at least one point lies on the ground (given horizontal camera orientation),
        therefore is inaccessible
        """
        # height threshold below which point pairs are filtered out (y axis points down)
        height_thresh = self.y + self.height * 0.9
        grasp_points_idx_all_len = len(grasp_points_dict_list)
        i = 0
        while i < grasp_points_idx_all_len:
            points = grasp_points_dict_list[i]['points']
            if points[0][1] > height_thresh or points[0][1] > height_thresh:
                grasp_points_dict_list.pop(i)
                grasp_points_idx_all_len -= 1
            else:
                i += 1
        return grasp_points_dict_list

    @staticmethod
    def get_points_distance(points):
        """
        returns distance between 2 given points
        """
        distance = ((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)**0.5
        return distance

    @staticmethod
    def get_grasp_orientation(points):
        """
        returns scaled absolute value of grasp orientation in range [0, 1], where 0 is horizontal and 1 is vertical
        """
        line = (points[0][0] - points[1][0], points[0][1] - points[1][1])
        if line[0] == 0:
            # avoid division by 0
            orientation = math.pi / 2
        else:
            # no need for atan2
            orientation = abs(math.atan(line[1] / line[0]))
        orientation_norm = orientation / (math.pi / 2)
        return orientation_norm

    def scale_contour(self, contour, position):
        """
        scales given contour from pixels to millimeters, based on scale factor = object distance / camera focal length
        """
        def get_centroid(contour):
            # contour moments
            M = cv2.moments(contour)
            # contour centroid (c_x, c_y)
            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            return centroid
        distance = position[2]
        scale_factor = distance / self.cam_focal_length
        # scale to mm
        contour = (contour * scale_factor).astype('int')
        # move whole contour so centroid matches given object position
        displacement = (position[:2] - get_centroid(contour)).astype('int')
        contour[:, 0] += displacement
        return contour

    @staticmethod
    def get_camera_focal_length(cam_mat):
        """
        returns focal length of a camera in px, read from given camera matrix
        focal length is mean of x and y focal lengths, as they are usually pretty much the same
        """
        f_x = cam_mat[0][0]
        f_y = cam_mat[1][1]
        f = (f_x + f_y) / 2
        return f

    @staticmethod
    def evaluate_grasp_points_orientation(contour, grasp_points_idx, distance):
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
                    point_origin_vector = point - coord_sys[2]
                    return np.array([point_origin_vector @ coord_sys[0], point_origin_vector @ coord_sys[1]])
                # middle curve point projected into UV coord system
                curve_points_uv_new = np.array([[0, 0]])
                # backup array is needed, as the last one to pass test is returned
                curve_points_uv = curve_points_uv_new
                i = 1
                # check value for contour length
                curve_length = 0
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

                    # length of U axis points projection
                    curve_length_new = abs(curve_points_uv_new[len(curve_points_uv_new) - 1][0] - curve_points_uv_new[0][0])
                    # stop adding points when length of curve projection (onto gripper) is greater than gripper size
                    # or smaller than previous value (which happens when curve begins to close)
                    if curve_length_new > gripper_size or curve_length_new < curve_length:
                        return curve_points_uv
                    else:
                        curve_length = curve_length_new
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
            # ax.invert_yaxis()
            # ax.plot(contour[:, :, 0], contour[:, :, 1], 'k.',
            #         [p_0[0], p_1[0]], [p_0[1], p_1[1]], 'b-',
            #         [origin[0], origin[0] + u[0] * 10], [origin[1], origin[1] + u[1] * 10], 'r-',
            #         [origin[0], origin[0] + v[0] * 10], [origin[1], origin[1] + v[1] * 10], 'g-')
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
            # ax.plot(curve_points[:, 0], curve_points[:, 1], 'k.',
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
        grasp_points_vector_norm = grasp_points_vector / distance

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
        distance += add_grasp_width_0 + add_grasp_width_1
        return orientation_err, distance
        
    @staticmethod
    def save(mask, position):
        np.savez_compressed('shape_analysis_input', mask=mask, position=position)

    @staticmethod
    def load_data():
        data = np.load('shape_analysis_input.npz')
        mask = data['mask']
        position = data['position']
        return mask, position

    @staticmethod
    def get_grasp_parameters(points, distance):
        """
        returns grasp parameters for given point pair
        :param points: grasp point pair dict
        :param distance: distance to point pair in Z axis
        :return: midpoint - middle point of grasp (x, y, z) in millimeters
                 orientation - orientation of gripper vs horizontal plane
                 width - needed width of grasp
        """
        point_0 = points['points'][0]
        point_1 = points['points'][1]
        midpoint = ((point_0[0] + point_1[0])/2, (point_0[1] + point_1[1])/2, distance)
        orientation = points['grasp_orientation'] * (math.pi / 2)
        width = points['distance']
        return midpoint, orientation, width


# shape_analyzer = ShapeAnalyzer(None)
# mask, distance = shape_analyzer.load_data()
# shape_analyzer.analyze_shape(mask, distance)
