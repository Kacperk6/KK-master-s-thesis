import cv2


def get_video_live():
    return cv2.VideoCapture(0)


def get_video_from_file(file_name):
    return cv2.VideoCapture('../../AI_filmy_zdjecia/' + file_name)


def get_image(file_path):
    return cv2.imread(file_path)


def split_stereo_image(stereo_image, height, width):
    """
    splits image from stereo camera to 2 separate images
    :param stereo_image: image to split
    :param height: input image height
    :param width: input image width
    :return: [frame_left, frame_right]: split input image
    """
    frame_left = stereo_image[0:height, 0:int(width/2)]
    frame_right = stereo_image[0:height, int(width / 2): width]
    return [frame_left, frame_right]
