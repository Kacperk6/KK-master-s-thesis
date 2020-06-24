import cv2
import os


def get_all_images(folder_path):
    """
    loads all images from a given directory
    :param folder_path: path to folder containing loaded images, relative to main project folder
                        eg. 'images/lifeLines'
    :return: python list of images in opencv (numpy) format
    """
    images = []
    for filename in os.listdir(folder_path):
        image = cv2.imread(f'{folder_path}/{filename}')
        if image is not None:
            images.append(image)
    return images
