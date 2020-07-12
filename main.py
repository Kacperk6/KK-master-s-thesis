import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereoscopy import Stereoscopy
from utils import camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


def run():
    logging.info("program started")

    #yolact = YolactFacade()
    stereoscopy = Stereoscopy()

    cap = camera.get_video_live()
    #cap = camera.get_video_from_file('output1.avi')
    while True:
        _, img_double = cap.read()
        img_l_dist, img_r_dist = stereoscopy.split_stereo_image(img_double, img_double.shape[0], img_double.shape[1])
        img_l, img_r = stereoscopy.preprocess_stereo_image(img_double, img_double.shape[0], img_double.shape[1])
        cv2.imshow("img_l", img_l)
        cv2.imshow("img_r", img_r)
        depth_map = stereoscopy.run(img_l, img_r)
        depth_map_img = depth_map/2048
        cv2.imshow("depth", depth_map_img)
        '''
        mask = yolact.run(img_l, 'person')
        if mask is None:
            logging.info("mask not found")
        else:
            yolact.draw_object(img_l, mask)
        '''
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.imwrite('img_l.png', img_l)
            cv2.imwrite("img_r.png", img_r)
    cap.release()
    cv2.destroyAllWindows()


run()
