import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereoscopy import Stereoscopy
from utils import camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


def run():
    logging.info("program started")
    #cap = camera.get_video_from_file('AI_data2/str_01.avi')
    cap = camera.get_video_live()
    #yolact = YolactFacade()
    stereoscopy = Stereoscopy(False, True)
    while True:
        _, img = cap.read()
        #cv2.imshow("img", img)
        depth_map = stereoscopy.run(img)
        '''
        mask = yolact.run(img, 'person')
        if mask is None:
            logging.info("mask not found")
        else:
            yolact.draw_object(img, mask)
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


run()
