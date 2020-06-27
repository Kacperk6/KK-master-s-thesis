import numpy as np
import cv2
import logging


DEPTH_VISUALIZATION_SCALE = 2048


def nothing(x):
    """
    just for calibration trackbars
    """
    pass


class Stereoscopy:
    def __init__(self, calibrate_camera=False, calibrate_stereo_matching=False):
        self.calibrate_stereo_matching = calibrate_stereo_matching
        if calibrate_camera:
            self.calibrate_stereo()
        self.image_size, self.map_l_x, self.map_l_y, self.map_r_x, self.map_r_y = self.load_calibration_data()
        self.stereo = cv2.StereoBM_create()
        self.stereo.setMinDisparity(4)
        self.stereo.setNumDisparities(128)
        self.stereo.setBlockSize(21)
        self.stereo.setSpeckleRange(16)
        self.stereo.setSpeckleWindowSize(45)
        cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)
        if self.calibrate_stereo_matching:
            pass
            #cv2.createTrackbar("edge minVal", "processed img", EDGE_MIN_VAL, 255, nothing)

    def run(self, img):
        img_l, img_r = self.preprocess_stereo_image(img, img.shape[0], img.shape[1])
        cv2.imshow("img_l", img_l)
        cv2.imshow("img_r", img_r)

        disparity = self.stereo.compute(img_l, img_r)
        cv2.imshow("depth", disparity/DEPTH_VISUALIZATION_SCALE)
        return disparity

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
        splits stereo image, converts to greyscale and undistorts it
        :param stereo_image: image to split
        :param height: input image height
        :param width: input image width
        :return: frame_left, frame_right: preprocessed input image
        """
        frame_left, frame_right = self.split_stereo_image(stereo_image, height, width)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
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
        return image_size, map_l_x, map_l_y, map_r_x, map_r_y

    def calibrate_stereo(self):
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
        # np.save(path + 'objpoints', objpoints)
        np.savez_compressed(path + 'stereo', image_size=(int(WIDTH / 2), HEIGHT), map_l_x=leftMapX, map_l_y=leftMapY,
                            map_r_x=rightMapX, map_r_y=rightMapY)
        '''
        path_l = path + 'left/'
        np.save(path_l + 'mtx', mtx_l)
        np.save(path_l + 'dist', dist_l)
        np.save(path_l + 'rvecs', rvecs_l)
        np.save(path_l + 'tvecs', tvecs_l)
        np.save(path_l + 'newcameramtx', newcameramtx_l)
        np.save(path_l + 'imgpoints', imgpoints_l)

        path_r = path + 'right/'
        np.save(path_r + 'mtx', mtx_r)
        np.save(path_r + 'dist', dist_r)
        np.save(path_r + 'rvecs', rvecs_r)
        np.save(path_r + 'tvecs', tvecs_r)
        np.save(path_r + 'newcameramtx', newcameramtx_r)
        np.save(path_r + 'imgpoints', imgpoints_r)
        '''
        logging.info("camera calibration finished")
        return True
