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

        # whether update interface image
        self.update = True

    def run(self):
        cap = camera.get_video_live()
        self.update = True
        while cap.isOpened():
            if self.update:
                ret, img_double = cap.read()
                # handle image read error
                if not ret:
                    logging.error("image read failed. aborting")
                    break
                img_l, img_r = self.stereo_vision.preprocess_stereo_image(img_double, img_double.shape[0],
                                                                          img_double.shape[1])
                img_interface = self.cut_image(img_l)
                cv2.imshow("interface", img_interface)

            key = cv2.waitKey(1)
            if key in (ord('0'), ord('1'), ord('2')):
                self.mode = int(chr(key))
                logging.info("mode changed to {}".format(self.mode))
            elif key == ord('q'):
                break
            elif key == ord(' '):
                cv2.imwrite('img_l.png', img_l)
                cv2.imwrite("img_r.png", img_r)
            # elif key == ord('d'):
            #     depth_map, points_3d = self.stereo_vision.run(img_l, img_r)
            #     if mask is not None:
            #         points_3d, depth_map, img_l = self.stereo_vision.mask_3d(mask, points_3d, depth_map, img_l)
            #     depth_map_img = depth_map / 128
            #     cv2.imshow("depth", depth_map_img)
            #     self.stereo_vision.draw_point_cloud(points_3d, depth_map, img_l)

            if self.mode == 1 and self.update:
                img_detected = self.detect_objects(img_interface)
                cv2.imshow("interface_2", img_detected)
                self.update = False

        logging.info("shutting down")
        cap.release()
        cv2.destroyAllWindows()

    def cut_image(self, img):
        h, w = img.shape[0], img.shape[1]
        img = img[0:h, self.img_cut_size:w]
        return img

    # mouse callback function
    def get_mouse_position(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDBLCLK:
            # compensate cut image
            self.mouse_x, self.mouse_y = x + self.img_cut_size, y
            if self.mode == 0:
                pass
            elif self.mode == 1:
                pass
            elif self.mode == 2:
                pass

    def detect_objects(self, img):
        img_detected = self.yolact.evaluate_frame(img)
        return img_detected


ui = UI()
ui.run()
