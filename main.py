import numpy as np
import cv2
import logging

from yolact1.yolact_facade import YolactFacade
from localization.stereo_vision import StereoVision
from utils import camera
from utils import task_functions
from yolact1.data.config import COCO_CLASSES
from shape_analysis.shape_analysis import ShapeAnalyzer


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s')


class UI:
    def __init__(self):
        logging.info("initializing user interface")

        logging.info("initializing YOLACT")
        self.yolact = YolactFacade()

        logging.info("initializing stereo vision")
        self.stereo_vision = StereoVision()

        logging.info("initializing shape analyzer")
        self.shape_analyzer = ShapeAnalyzer(self.stereo_vision.cam_mat_l)

        # how many left pixels to cut from interface image to fit stereo vision output data
        self.img_cut_size = self.stereo_vision.minDisparity + self.stereo_vision.numDisparities

        # whether to show Open3D scene visualization
        self.show_3d_model = False


        # action to perform
        # 0 - just image transmission
        # 1 - detect objects
        # 2 - move to point
        self.mode = 0

        # whether update interface image
        self.update = True

        cv2.namedWindow("interface")
        cv2.setMouseCallback("interface", self.get_mouse_position)

        # mouse callback parameters holders
        self.is_mouse_called = False
        self.mouse_x, self.mouse_y = 0, 0

    def run(self):
        cap = camera.get_video_live()
        self.update = True
        while cap.isOpened():
            if self.mode == 0:
                self.update = True
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

            # 1 - detect objects
            if self.mode == 1:
                if self.update:
                    logging.info("detecting objects")
                    img_detected, cats, _, bboxes, masks = self.yolact.evaluate_frame(img_interface)
                    if img_detected is None:
                        logging.info("no objects detected")
                    else:
                        cv2.imshow("interface", img_detected)
                    self.update = False
                if self.is_mouse_called and not self.update:
                    self.is_mouse_called = False
                    object_idx = task_functions.choose_object_at_point(self.mouse_x, self.mouse_y, cats, bboxes, masks)
                    if object_idx is not None:
                        logging.info("chosen object: {}".format(COCO_CLASSES[cats[object_idx]]))
                        scene_3d, disparity_map = self.stereo_vision.get_3d_scene(img_l, img_r, self.show_3d_model)
                        # disparity_map_img = disparity_map / 128
                        # cv2.imshow("disparity", disparity_map_img)
                        # cv2.imshow("image", img_l)
                        mask = self.resize_mask(masks[object_idx], (img_l.shape[0], img_l.shape[1]))
                        object_3d = self.stereo_vision.mask_3d(mask, scene_3d, disparity_map, img_l,
                                                               self.show_3d_model)
                        position = self.stereo_vision.get_object_position(object_3d)
                        print("poistion: ", position)
                        grasp_parameters = self.shape_analyzer.analyze_shape(mask, position)
                        print("grasp_parameters: ", grasp_parameters)
                    else:
                        logging.info("no detected object at given point")
            elif self.mode == 2:
                if self.update:
                    logging.info("mapping scene")
                    scene_3d, disparity_map = self.stereo_vision.get_3d_scene(img_l, img_r, self.show_3d_model)
                    # disparity_map_img = disparity_map / 128
                    # cv2.imshow("disparity", disparity_map_img)
                    self.update = False
                if self.is_mouse_called and not self.update:
                    self.is_mouse_called = False
                    #np.savez_compressed("points_3d_ground", scene_3d=scene_3d, disparity_map=disparity_map, img=img_l)
                    point = scene_3d[self.mouse_y][self.mouse_x + + self.img_cut_size]
                    print("point: ", point)
                    print(task_functions.is_point_reachable(scene_3d, point))


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
            self.is_mouse_called = True
            self.mouse_x, self.mouse_y = x, y

    def resize_mask(self, mask, size):
        mask_resized = np.zeros(size, np.bool)
        mask_resized[0:size[0], self.img_cut_size:size[1]] = mask
        return mask_resized


ui = UI()
ui.run()
