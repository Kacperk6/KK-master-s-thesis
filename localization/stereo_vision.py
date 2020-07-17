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
            self.calibrate_camera()
        self.image_size, self.map_l_x, self.map_l_y, self.map_r_x, self.map_r_y, self.Q = self.load_calibration_data()
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

        cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)
        if self.calibrate_stereo_matching:
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
        img_l_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_r_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disparity_map = self.stereo.compute(img_l_gray, img_r_gray)
        disparity_map = disparity_map.astype('float32')/16
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
            # disparity_visualiztion = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            cv2.imshow("depth", disparity_visualiztion)
            if key == ord('p'):
                self.make_point_cloud(disparity, img_l_color)
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

    def split_stereo_image(self, stereo_image, height, width):
        """
        splits image from stereo camera to 2 separate images
        :param stereo_image: image to split
        :param height: input image height
        :param width: input image width
        :return: frame_left, frame_right: split input image
        """
        logging.debug("splitting stereo image")
        frame_left = stereo_image[0:height, 0:int(width / 2)]
        frame_right = stereo_image[0:height, int(width / 2): width]
        return frame_left, frame_right

    def preprocess_stereo_image(self, stereo_image, height, width):
        """
        splits stereo image and undistorts it
        :param stereo_image: image to split
        :param height: input image height
        :param width: input image width
        :return: frame_left, frame_right: preprocessed input image
        """
        frame_left, frame_right = self.split_stereo_image(stereo_image, height, width)
        frame_left, frame_right = self.undistort(frame_left, frame_right)
        return frame_left, frame_right

    def undistort(self, img_l, img_r):
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
        return image_size, map_l_x, map_l_y, map_r_x, map_r_y, Q

    def calibrate_camera(self):
        """
        Creates stereo camera calibration data and saves it in file
        """
        # vertices, not squares
        columns = 6
        rows = 8
        square_size = 25  # [mm]

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints_l = []  # 2d points in image plane
        imgpoints_r = []

        cap = cv2.VideoCapture('utils/output.avi')
        WIDTH = int(cap.get(3))
        HEIGHT = int(cap.get(4))
        FPS = int(cap.get(5))

        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)

        # sample counter; just for user information
        i = 0

        while cap.isOpened():
            ret, img_double = cap.read()
            if ret:
                img_double_gray = cv2.cvtColor(img_double, cv2.COLOR_BGR2GRAY)
                cv2.imshow("img", img_double_gray)
                key = cv2.waitKey(int(1000 / FPS))
                if key == ord(' '):
                    img_l, img_r = self.split_stereo_image(img_double_gray, HEIGHT, WIDTH)
                    # Find the chess board corners
                    ret_l, corners_l = cv2.findChessboardCorners(img_l, (rows, columns), None)
                    ret_r, corners_r = cv2.findChessboardCorners(img_r, (rows, columns), None)
                    if ret_l & ret_r:
                        corners2_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
                        corners2_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)
                        while True:
                            # Draw and display the corners
                            img_corners_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
                            img_corners_l = cv2.drawChessboardCorners(img_corners_l, (rows, columns), corners2_l, ret_l)
                            img_corners_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
                            img_corners_r = cv2.drawChessboardCorners(img_corners_r, (rows, columns), corners2_r, ret_r)

                            cv2.namedWindow('corners_l', cv2.WINDOW_AUTOSIZE)
                            cv2.imshow('corners_l', img_corners_l)
                            cv2.namedWindow('corners_r', cv2.WINDOW_AUTOSIZE)
                            cv2.moveWindow('corners_r', 1000, 0)
                            cv2.imshow('corners_r', img_corners_r)
                            key = cv2.waitKey()
                            if key == ord('y'):
                                # accept found corners and add them to imgpoints arrays
                                objpoints.append(objp)
                                imgpoints_l.append(corners2_l)
                                imgpoints_r.append(corners2_r)
                                i += 1
                                print("number of samples: ", i)
                                break
                            elif key == ord('s'):
                                # change detected corners order for one of images, to match second image
                                corners2_r = np.flip(corners2_r, 0)
                            else:
                                break
                        cv2.destroyWindow('corners_l')
                        cv2.destroyWindow('corners_r')
                elif key == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        logging.info("calibrating cameras")
        # yes, i know it's not safe :/
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, img_l.shape[::-1], None,
                                                                     None)
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, img_r.shape[::-1], None,
                                                                     None)

        (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            mtx_l, dist_l,
            mtx_r, dist_r,
            (int(WIDTH / 2), HEIGHT), None, None, None, None,
            cv2.CALIB_FIX_INTRINSIC, criteria)

        (leftRectification, rightRectification, leftProjection, rightProjection,
         dispartityToDepthMap, _, _) = cv2.stereoRectify(
            mtx_l, dist_l,
            mtx_r, dist_r,
            (int(WIDTH / 2), HEIGHT), rotationMatrix, translationVector,
            None, None, None, None, None,
            cv2.CALIB_ZERO_DISPARITY, 0)

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, leftRectification,
            leftProjection, (int(WIDTH / 2), HEIGHT), cv2.CV_32FC1)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, rightRectification,
            rightProjection, (int(WIDTH / 2), HEIGHT), cv2.CV_32FC1)

        logging.info("saving calibration data")
        path = 'config/camera_calibration/'
        np.savez_compressed(path + 'stereo', image_size=(int(WIDTH / 2), HEIGHT), map_l_x=leftMapX, map_l_y=leftMapY,
                            map_r_x=rightMapX, map_r_y=rightMapY, Q=dispartityToDepthMap)
        logging.info("camera calibration finished")
        return True

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
        # extend mask from 2d to 3d
        mask_3d = mask[:, :, np.newaxis]

        # invalidate masked points
        points_3d = np.where(mask_3d, points_3d, np.nan)
        disparity_map = np.where(mask, disparity_map, disparity_map.min())
        img = np.where(mask_3d, img, 0)

        if show_3d_model:
            self.draw_point_cloud(points_3d, disparity_map, img)
        return points_3d, disparity_map, img
