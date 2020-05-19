import cv2


def get_video_live():
    return cv2.VideoCapture(0)


def get_video_from_file(file_name):
    return cv2.VideoCapture('../../../AI_filmy_zdjecia/' + file_name)


def get_image(file_path):
    return cv2.imread(file_path)
