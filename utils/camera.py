import numpy as np
import cv2
import logging

from utils.import_image import get_all_images

# większa rozdzielczość źle działa
WIDTH = 1280
HEIGHT = 480
FPS = 30

# loaded from file
# ŁADUJE TE WARTOŚCI ZA KAŻDYM WYWOŁANIEM camera.py
IMAGE_SIZE = 0
MAP_L_X = 0
MAP_L_Y = 0
MAP_R_X = 0
MAP_R_Y = 0


def get_video_live():
    logging.info("getting video")
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(5, FPS)
    logging.info("video parameters: resolution: {}x{}; FPS: {}".format(cap.get(3), cap.get(4), cap.get(5)))
    return cap


def get_video_from_file(file_name):
    logging.info("getting video")
    return cv2.VideoCapture('../../AI_filmy_zdjecia/' + file_name)


def get_image(file_path):
    logging.info("getting single image")
    return cv2.imread(file_path)


def save_video():
    """
    saves unedited video from camera.
    Designed to be used independently
    """
    cap = get_video_live()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, FPS, (WIDTH, HEIGHT))
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
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
