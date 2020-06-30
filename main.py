import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereoscopy import Stereoscopy
from utils import camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


def run():
    logging.info("program started")
    #cap = camera.get_video_from_file('AI_data2/str_01.avi')
    stereoscopy = Stereoscopy()
    cap = camera.get_video_live()
    #yolact = YolactFacade()
    while True:
        _, img_double = cap.read()
        img_l, img_r = stereoscopy.preprocess_stereo_image(img_double, img_double.shape[0], img_double.shape[1])
        cv2.imshow("img_l", img_l)
        depth_map = stereoscopy.run(img_l, img_r)
        depth_map_img = depth_map/2048
        cv2.imshow("depth", depth_map_img)
        '''
        mask = yolact.run(img, 'person')
        if mask is None:
            logging.info("mask not found")
        else:
            yolact.draw_object(img, mask)
        '''
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.imwrite('img_l.png', img_l)
            cv2.imwrite('depth.png', depth_map_img)
    cap.release()
    cv2.destroyAllWindows()


run()
