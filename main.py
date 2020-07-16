import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereo_vision import StereoVision
from utils import camera


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


class UI:
    def __init__(self):
        logging.info("initializing user interface")

        logging.info("initializing YOLACT")
        self.yolact = YolactFacade()

        logging.info("initializing stereo vision")
        self.stereo_vision = StereoVision()

        # how many left pixels to cut from interface image to fit stereo vision output data
        self.img_cut_size = self.stereo_vision.minDisparity + self.stereo_vision.numDisparities

        cv2.namedWindow("interface")
        cv2.setMouseCallback("interface", self.get_mouse_position)
        self.mouse_x, self.mouse_y = 0, 0

        # action to perform
        # 0 - just image transmission
        # 1 - detect objects
        # 2 - move to point
        self.mode = 0

    def run(self):
        #global mouse_action
        cap = camera.get_video_live()
        while cap.isOpened():
            ret, img_double = cap.read()
            if ret:
                img_l, img_r = self.stereo_vision.preprocess_stereo_image(img_double, img_double.shape[0],
                                                                          img_double.shape[1])
                img_interface = self.cut_image(img_l)
                cv2.imshow("interface", img_interface)

                if self.mode == 1:
                    img_detected = self.detect_objects(img_interface)
                    cv2.imshow("interface_2", img_detected)
                '''
                mask = self.yolact.run(img_l, 'person')
                if mask is None:
                    logging.info("mask not found")
                else:
                    self.yolact.draw_object(img_l, mask)
                '''
                key = cv2.waitKey(1)
                if key in (ord('0'), ord('1'), ord('2')):
                    self.mode = int(chr(key))
                    logging.info("mode changed to {}".format(self.mode))
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.imwrite('img_l.png', img_l)
                    cv2.imwrite("img_r.png", img_r)
                '''
                elif key == ord('d'):
                    depth_map, points_3d = self.stereo_vision.run(img_l, img_r)
                    if mask is not None:
                        points_3d, depth_map, img_l = self.stereo_vision.mask_3d(mask, points_3d, depth_map, img_l)
                    depth_map_img = depth_map / 128
                    cv2.imshow("depth", depth_map_img)
                    self.stereo_vision.draw_point_cloud(points_3d, depth_map, img_l)
                '''
        cap.release()
        cv2.destroyAllWindows()

    def cut_image(self, img):
        h, w = img.shape[0], img.shape[1]
        img = img[0:h, self.img_cut_size:w]
        return img

    # mouse callback function
    def get_mouse_position(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.mode == 0:
                pass
            elif self.mode == 1:
                pass
            self.mouse_x, self.mouse_y = x + self.img_cut_size, y

    def detect_objects(self, img):
        img_detected = self.yolact.evaluate_frame(img)
        return img_detected


ui = UI()
ui.run()
