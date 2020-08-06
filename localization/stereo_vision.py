import numpy as np
import cv2
import logging
import open3d as o3d


from utils import camera


DEPTH_VISUALIZATION_SCALE = 2048


def nothing(x):
    """
    just for calibration trackbars
    """
    pass


class StereoVision:
    def __init__(self, calibrate_camera=False, calibrate_stereo_matching=False):
        self.calibrate_stereo_matching = calibrate_stereo_matching
        if calibrate_camera:
            camera.calibrate_camera()
        self.image_size, self.map_l_x, self.map_l_y, self.map_r_x, self.map_r_y, self.Q, self.cam_mat_l, self.cam_mat_r = self.load_calibration_data()
        '''
        self.stereo.setMinDisparity(4)
        self.stereo.setNumDisparities(128)
        self.stereo.setBlockSize(21)
        self.stereo.setSpeckleRange(16)
        self.stereo.setSpeckleWindowSize(45)
        '''
        # load stereo matching parameters from file
        self.minDisparity, self.numDisparities, self.blockSize, self.p1, self.p2, self.disp12MaxDiff,\
            self.preFilterCap, self.uniquenessRatio, self.speckleWindowSize, self.speckleRange, self.fullDP\
            = self.load_stereo_matching_parameters()

        self.stereo = cv2.StereoSGBM_create(self.minDisparity, self.numDisparities, self.blockSize, self.p1, self.p2,
                                            self.disp12MaxDiff, self.preFilterCap, self.uniquenessRatio,
                                            self.speckleWindowSize, self.speckleRange, self.fullDP)

        if self.calibrate_stereo_matching:
            cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("minDisparity", "depth", self.minDisparity, 100, nothing)
            cv2.createTrackbar("numDisparities x 16", "depth", int(self.numDisparities/16), 20, nothing)
            cv2.createTrackbar("blockSize x 2 + 1", "depth", int(self.blockSize/2), 15, nothing)
            cv2.createTrackbar("p1", "depth", self.p1, 2000, nothing)
            cv2.createTrackbar("p2", "depth", self.p2, 5000, nothing)
            cv2.createTrackbar("disp12MaxDiff", "depth", self.disp12MaxDiff, 500, nothing)
            cv2.createTrackbar("preFilterCap", "depth", self.preFilterCap, 50, nothing)
            cv2.createTrackbar("uniquenessRatio", "depth", self.uniquenessRatio, 30, nothing)
            cv2.createTrackbar("speckleWindowSize", "depth", self.speckleWindowSize, 500, nothing)
            cv2.createTrackbar("speckleRange", "depth", self.speckleRange, 5, nothing)
            cv2.createTrackbar("fullDP", "depth", self.fullDP, 1, nothing)

            self.calibrate_stereo_matcher()

    def get_3d_scene(self, img_l, img_r, show_3d_model=False):
        MIN_DISPARITY = 5
        img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disparity_map = self.stereo.compute(img_l_gray, img_r_gray)
        # scale disparity map
        disparity_map = (disparity_map / 16).astype('float32')
        disparity_map[disparity_map < MIN_DISPARITY] = disparity_map.min()
        points_3d = cv2.reprojectImageTo3D(disparity_map, self.Q)
        if show_3d_model:
            self.draw_point_cloud(points_3d, disparity_map, img_l)
        return points_3d, disparity_map

    def calibrate_stereo_matcher(self):
        img_l_color = cv2.imread('img_l.png')
        img_l = cv2.cvtColor(img_l_color, cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(cv2.imread('img_r.png'), cv2.COLOR_BGR2GRAY)
        while True:
            correct_parameters = True
            # 0 or less
            minDisparity = cv2.getTrackbarPos("minDisparity", "depth")
            # number of depth levels, dividable by 16
            numDisparities = 16 * cv2.getTrackbarPos("numDisparities x 16", "depth")
            if numDisparities < 1:
                logging.error("number of disparity levels must be > 0")
                correct_parameters = False
            # matched block size, odd number, usually in 3-11 range
            blockSize = 2 * cv2.getTrackbarPos("blockSize x 2 + 1", "depth") + 1
            # disparity smoothness parameter
            # penalty on disparity change by +/- 1 between neighbor pixels
            p1 = cv2.getTrackbarPos("p1", "depth")
            # disparity smoothness parameter
            # penalty on disparity change by more than +/- 1 between neighbor pixels
            # p2 > p1
            p2 = cv2.getTrackbarPos("p2", "depth")
            if p2 < p1:
                logging.error("p2 must be greater than p1")
                correct_parameters = False
            # maximum difference in left-right disparity check; <0 to disable
            disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "depth")
            # threshold for prefiltered pixels
            preFilterCap = - cv2.getTrackbarPos("preFilterCap", "depth")
            # % by which best score must be better than next one to be considered correct, usually in 5-15 range
            uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "depth")
            # max size of smooth region to consider speckle and invalidate
            # usually in 50-100 range, 0 to disable
            speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "depth")
            # max disparity variation inside a speckle
            # usually 1 or 2, implicitly multiplied by 16
            speckleRange = cv2.getTrackbarPos("speckleRange", "depth")
            # enable full-scale two-pass dynamic programming algorithm, consumes great amount of memory
            fullDP = - cv2.getTrackbarPos("fullDP", "depth")

            key = cv2.waitKey()

            if correct_parameters:
                self.stereo.setMinDisparity(minDisparity)
                self.stereo.setNumDisparities(numDisparities)
                self.stereo.setBlockSize(blockSize)
                self.stereo.setP1(p1)
                self.stereo.setP2(p2)
                self.stereo.setDisp12MaxDiff(disp12MaxDiff)
                self.stereo.setPreFilterCap(preFilterCap)
                self.stereo.setUniquenessRatio(uniquenessRatio)
                self.stereo.setSpeckleWindowSize(speckleWindowSize)
                self.stereo.setSpeckleRange(speckleRange)
                self.stereo.setMode(fullDP)

            disparity = self.stereo.compute(img_l, img_r)
            disparity_visualiztion = disparity / DEPTH_VISUALIZATION_SCALE
            cv2.imshow("depth", disparity_visualiztion)
            if key == ord('p'):
                points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
                self.draw_point_cloud(points_3d, disparity, img_l_color)
            # key = cv2.waitKey(int(1000/cap.get(5)))

            if key == ord('y') and correct_parameters:
                np.savez_compressed('config/stereo_matching_parameters/SGBM', minDisparity=minDisparity,
                                    numDisparities=numDisparities, blockSize=blockSize, p1=p1, p2=p2,
                                    disp12MaxDiff=disp12MaxDiff, preFilterCap=preFilterCap,
                                    uniquenessRatio=uniquenessRatio,
                                    speckleWindowSize=speckleWindowSize, speckleRange=speckleRange, fullDP=fullDP)
                self.calibrate_stereo_matching = False

                self.minDisparity = minDisparity
                self.numDisparities = numDisparities
                self.blockSize = blockSize
                self.p1 = p1
                self.p2 = p2
                self.disp12MaxDiff = disp12MaxDiff
                self.preFilterCap = preFilterCap
                self.uniquenessRatio = uniquenessRatio
                self.speckleWindowSize = speckleWindowSize
                self.speckleRange = speckleRange
                self.fullDP = fullDP
                break
            elif key == ord('q'):
                break
        #cap.release()
        cv2.destroyWindow("depth")

    def load_stereo_matching_parameters(self):
        path = 'config/stereo_matching_parameters/SGBM.npz'
        parameters = np.load(path, allow_pickle=False)
        minDisparity = parameters['minDisparity']
        numDisparities = parameters['numDisparities']
        blockSize = parameters['blockSize']
        p1 = parameters['p1']
        p2 = parameters['p2']
        disp12MaxDiff = parameters['disp12MaxDiff']
        preFilterCap = parameters['preFilterCap']
        uniquenessRatio = parameters['uniquenessRatio']
        speckleWindowSize = parameters['speckleWindowSize']
        speckleRange = parameters['speckleRange']
        fullDP = parameters['fullDP']
        return minDisparity, numDisparities, blockSize, p1, p2, disp12MaxDiff, preFilterCap, uniquenessRatio,\
               speckleWindowSize, speckleRange, fullDP

    def preprocess_stereo_image(self, stereo_image, height, width):
        """
        splits stereo image and undistorts it
        :param stereo_image: image to split
        :param height: input image height
        :param width: input image width
        :return: frame_left, frame_right: preprocessed input image
        """
        frame_left, frame_right = camera.split_stereo_image(stereo_image, height, width)
        frame_left, frame_right = self.rectify(frame_left, frame_right)
        return frame_left, frame_right

    def rectify(self, img_l, img_r):
        img_l = cv2.remap(img_l, self.map_l_x, self.map_l_y, cv2.INTER_LINEAR)
        img_r = cv2.remap(img_r, self.map_r_x, self.map_r_y, cv2.INTER_LINEAR)
        return img_l, img_r

    def load_calibration_data(self):
        path = 'config/camera_calibration/stereo.npz'
        calibration = np.load(path, allow_pickle=False)
        image_size = tuple(calibration["image_size"])
        map_l_x = calibration["map_l_x"]
        map_l_y = calibration["map_l_y"]
        map_r_x = calibration["map_r_x"]
        map_r_y = calibration["map_r_y"]
        Q = calibration["Q"]
        cam_mat_l = calibration["cam_mat_l"]
        cam_mat_r = calibration["cam_mat_r"]
        return image_size, map_l_x, map_l_y, map_r_x, map_r_y, Q, cam_mat_l, cam_mat_r

    def draw_point_cloud(self, points_3d, disparity_map, img):
        """
        visualizes given points with Open3d
        :param points_3d: 3d points matrix related to image
        :param disparity_map: disparity map corresponding given points
        :param img: image to color visualized points with (normally should be left image from stereo camera)
        """
        #logging.info("saving points_3d")
        #np.savez_compressed('point_cloud', points_3d=points_3d, disparity_map=disparity_map, img=img)

        # convert colors from OpenCV format to the one used in Open3d (RGB 0:1 float)
        colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float')/255

        # create a mask to use only valid (detected) points
        mask_map = disparity_map > disparity_map.min()

        # mask points_3d and colors; loses image relation data
        output_points = points_3d[mask_map]
        output_colors = colors[mask_map]

        # create Open3D point cloud, assign given points and visualize
        logging.info("creating point cloud file")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(output_points)
        pcd.colors = o3d.Vector3dVector(output_colors)
        o3d.visualization.draw_geometries([pcd])

    def mask_3d(self, mask, points_3d, disparity_map, img, show_3d_model=False):
        """
        masks points_3d, disparity_map and img with given 2d mask
        """
        def remove_outliers(disparity_map):
            """
            sets single object disparity map outliers invalid using z-score threshold
            """
            # the higher value the less restrictive threshold is; 3 means that in normal distribution 99,7% points would remain
            z_threshold = 2
            disparity_invalid = disparity_map.min()
            # analyze only valid disparities
            disparity_map_valid = disparity_map[disparity_map > disparity_invalid]
            mean = np.mean(disparity_map_valid)
            std = np.std(disparity_map_valid)
            for i in range(disparity_map.shape[0]):
                for j in range(disparity_map.shape[1]):
                    if disparity_map[i][j] > disparity_invalid:
                        z = (disparity_map[i][j] - mean) / std
                        if abs(z) > z_threshold:
                            disparity_map[i][j] = disparity_invalid
            return disparity_map

        # invalidate masked points
        disparity_map = np.where(mask, disparity_map, disparity_map.min())
        # remove disparity outliers
        disparity_map = remove_outliers(disparity_map)
        # update mask to not contain detected outliers
        mask_updated = np.where(disparity_map > disparity_map.min(), True, False)
        # extend mask from 2d to 3d
        mask_3d = mask_updated[:, :, np.newaxis]
        # mask 3d points to detected object without outlying points
        points_3d = np.where(mask_3d, points_3d, np.nan)

        if show_3d_model:
            self.draw_point_cloud(points_3d, disparity_map, img)
        return points_3d

    @staticmethod
    def get_object_position(object_3d):
        # get non Nan points, array flattens
        object_3d = object_3d[~np.isnan(object_3d)]
        # reshape array to 2 dimensions to separate point coordinate data
        object_3d = np.reshape(object_3d, (int(len(object_3d)/3), 3))
        # get mean position of object points
        position = np.mean(object_3d, axis=0)
        return position
