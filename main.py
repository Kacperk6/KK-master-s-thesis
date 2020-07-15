import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereo_vision import StereoVision
from utils import camera

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


def run():
    logging.info("program started")

    yolact = YolactFacade()
    stereoscopy = StereoVision(calibrate_stereo_matching=True)

    cap = camera.get_video_live()
    while True:
        _, img_double = cap.read()
        img_l, img_r = stereoscopy.preprocess_stereo_image(img_double, img_double.shape[0], img_double.shape[1])
        cv2.imshow("img_l", img_l)

        mask = yolact.run(img_l, 'person')
        mask = None
        if mask is None:
            logging.info("mask not found")
        else:
            yolact.draw_object(img_l, mask)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.imwrite('img_l.png', img_l)
            cv2.imwrite("img_r.png", img_r)
        elif key == ord('d'):
            depth_map, points_3d = stereoscopy.run(img_l, img_r)
            if mask is not None:
                points_3d, depth_map, img_l = stereoscopy.mask_3d(mask, points_3d, depth_map, img_l)
            depth_map_img = depth_map / 128
            cv2.imshow("depth", depth_map_img)
            stereoscopy.draw_point_cloud(points_3d, depth_map, img_l)

    cap.release()
    cv2.destroyAllWindows()


run()
