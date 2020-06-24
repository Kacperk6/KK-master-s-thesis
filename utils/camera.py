import numpy as np
import cv2
import logging

from utils.import_image import get_all_images

WIDTH = 1280
HEIGHT = 480
FPS = 30


def get_video_live():
    logging.info("getting video")
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(5, FPS)
    logging.info("video parameters: resolution: {}x{}; FPS: {}".format(WIDTH, HEIGHT, FPS))
    return cap


def get_video_from_file(file_name):
    logging.info("getting video")
    return cv2.VideoCapture('../../AI_filmy_zdjecia/' + file_name)


def get_image(file_path):
    logging.info("getting single image")
    return cv2.imread(file_path)


def split_stereo_image(stereo_image, height, width):
    """
    splits image from stereo camera to 2 separate images
    :param stereo_image: image to split
    :param height: input image height
    :param width: input image width
    :return: frame_left, frame_right: split input image
    """
    logging.debug("splitting stereo image")
    frame_left = stereo_image[0:height, 0:int(width/2)]
    frame_right = stereo_image[0:height, int(width / 2): width]
    return frame_left, frame_right


def preprocess_stereo_image(stereo_image, height, width):
    """
    splits stereo image and undistorts it
    :param stereo_image: image to split
    :param height: input image height
    :param width: input image width
    :return: frame_left, frame_right: preprocessed input image
    """
    frame_left, frame_right = split_stereo_image(stereo_image, height, width)
    frame_left = undistort(frame_left, 'left')
    frame_right = undistort(frame_right, 'right')
    return frame_left, frame_right


def save_video():
    """
    saves unedited video from camera.
    Designed to be used independently
    """
    cap = get_video_live()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, FPS, (WIDTH, HEIGHT))
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("img", frame)
            out.write(frame)
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def calibrate(camera_name='left'):
    """
    Creates camera calibration data and saves it in files
    :param camera_name: name of a camera to calibrate; "left" or "right"
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
    imgpoints = []  # 2d points in image plane

    cap = cv2.VideoCapture('output.avi')
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    # sample counter; just for user information
    i = 0

    while cap.isOpened():
        ret, img_double = cap.read()
        if ret:
            img_double_gray = cv2.cvtColor(img_double, cv2.COLOR_BGR2GRAY)
            cv2.imshow("img", img_double_gray)
            key = cv2.waitKey(int(1000 / FPS))
            if key == ord(' '):
                img_l, img_r = split_stereo_image(img_double_gray, img_double_gray.shape[0], img_double_gray.shape[1])
                if camera_name == 'left':
                    img_gray = img_l
                elif camera_name == 'right':
                    img_gray = img_r
                else:
                    logging.info("invalid camera name")
                    return False
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(img_gray, (rows, columns), None)
                if ret:
                    corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                    # Draw and display the corners
                    img_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                    img_corners = cv2.drawChessboardCorners(img_corners, (rows, columns), corners2, ret)

                    cv2.namedWindow('corners', cv2.WINDOW_NORMAL)
                    cv2.imshow('corners', img_corners)
                    key = cv2.waitKey()
                    if key == ord('y'):
                        objpoints.append(objp)
                        imgpoints.append(corners2)
                        i += 1
                        print("number of samples: ", i)
                cv2.destroyWindow('corners')
            if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

    logging.info("calibrating camera")
    # yes, i know it's not safe :/
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (int(WIDTH/2), HEIGHT), 0, (int(WIDTH/2), HEIGHT))
    logging.info("saving calibration data")
    path = '../config/camera_calibration/{}/'.format(camera_name)
    np.save(path + 'mtx', mtx)
    np.save(path + 'dist', dist)
    np.save(path + 'rvecs', rvecs)
    np.save(path + 'tvecs', tvecs)
    np.save(path + 'newcameramtx', newcameramtx)
    logging.info("camera calibration finished")
    return True


def load_calibration_data(camera_name):
    path = 'config/camera_calibration/{}/'.format(camera_name)
    mtx = np.load(path + 'mtx.npy')
    dist = np.load(path + 'dist.npy')
    rvecs = np.load(path + 'rvecs.npy')
    tvecs = np.load(path + 'tvecs.npy')
    newcameramtx = np.load(path + 'newcameramtx.npy')
    return mtx, dist, rvecs, tvecs, newcameramtx


def undistort(img, camera_name='left'):
    mtx, dist, rvecs, tvecs, newcameramtx = load_calibration_data(camera_name)
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return img
