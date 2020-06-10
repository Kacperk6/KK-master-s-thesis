import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from utils import camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


def run():
    logging.info("program started")
    cap = camera.get_video_from_file('moje_640.avi')
    yolact = YolactFacade()
    while True:
        _, img = cap.read()
        cv2.imshow("img", img)
        mask = yolact.run(img, 'cup')
        if mask is None:
            logging.info("mask not found")
        else:
            yolact.draw_object(img, mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


run()
