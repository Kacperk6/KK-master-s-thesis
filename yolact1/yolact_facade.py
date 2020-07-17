import torch
import cv2
import numpy as np
import logging
import random

from .yolact import Yolact
from .data import set_cfg
from .utils.augmentations import FastBaseTransform
from .layers.output_utils import postprocess

from yolact1.data.config import COCO_CLASSES


class YolactFacade:
    def __init__(self):
        with torch.no_grad():
            # CUDA setup, I should really just replace all Tensor initializations but I'm in too deep at this point
            torch.backends.cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # Use whichever config you want here.
            set_cfg('yolact_base_config')

            self.net = Yolact().cuda()
            self.net.load_weights('yolact1/weights/yolact_base_54_800000.pth')
            self.net.eval()

            self.transform = FastBaseTransform()

            self.location = [0, 0]
            self.is_first_detection = True

    def run(self, frame, cat_name):
        """
        szuka obiektu podanej klasy na obrazie; główna metoda, do wywoływania z zewnątrz
        :param frame: analizowany obraz
        :param cat_name: string; nazwa klasy do wykrycia
        :return: np.array; maska wykrytego obiektu;
                 False; w przypadku braku wykrycia obiektu lub błędu
        """
        #cat_index = get_cat_index(cat_name)
        #if cat_index is None:
        #    logging.info("Aborting")
        #    return None

        cats, scores, boxes, masks = self.predict(frame)
        self.log_classes(cats)
        if cats is False:
            logging.info("No object found. Aborting")
            return None
        #masks_cat_index = np.where(cats == cat_index)
        #if len(masks_cat_index[0]) == 0:
        #    logging.info("Given category instance not found. Aborting")
        #    return None
        #masks_cat = masks[masks_cat_index]
        #boxes_cat = boxes[masks_cat_index]
        #if self.is_first_detection:
        #    mask = self.select_mask_score(masks_cat, boxes_cat)
        #    self.is_first_detection = False
        #else:
        #    mask = self.select_mask_location(masks_cat, boxes_cat)
        logging.info("mask found")
        return cats, scores, boxes, masks

    def predict(self, img):
        """
        analizuje jeden obraz
        :param img: np.array; analizowany obraz
        :return: cats: np.array([], int64)
                 scores: np.array([], float32)
                 boxes: np.array([[x1, y1, x2, y2]], int64)
                 masks: np.array([[[]]], uint8)
        """
        with torch.no_grad():
            h, w, _ = img.shape

            batch = self.transform(torch.from_numpy(img).cuda().float()[None, ...])
            preds = self.net(batch)

            # You can modify the score threshold to your liking
            cats, scores, boxes, masks = postprocess(preds, w, h, score_threshold=0.7)
            if len(cats.size()) == 0:
                return False
            # changed predicted data to cpu numpy array type to further process
            cats = cats.cpu().numpy().astype('int64')
            scores = scores.cpu().numpy().astype('float32')
            boxes = boxes.cpu().numpy().astype('int64')
            masks = masks.cpu().numpy().astype('bool')
            return cats, scores, boxes, masks

    def select_mask_score(self, masks, boxes):
        """
        Zwraca jedną maskę ze zbioru
        :param masks: np.array; zbiór masek
        :return: np.array; jedna maska
        """
        self.location = [int((boxes[0][0] + boxes[0][2]) / 2),
                         (int(boxes[0][1] + boxes[0][3]) / 2)]
        return masks[0]

    def select_mask_size(self, masks, boxes):
        size = 0
        size_largest = 0
        idx_largest = 0
        # przejście przez wyszystkie maski
        for i in range(len(masks)):
            # przejście przez wszystkie kolumny i wiersze bboxa odpowiadającego masce
            for j in range(boxes[i][0], boxes[i][2]):
                for k in range(boxes[i][1], boxes[i][3]):
                    # jeśli maska jest w tym miejscu, zwiększ rozmiar
                    if masks[i][k][j] == 1:
                        size += 1
            if size > size_largest:
                size_largest = size
                idx_largest = i
            size = 0
        self.update_object_location(boxes[idx_largest])
        return masks[idx_largest]

    def select_mask_location(self, masks, boxes):
        distance_closest = 1000  # więcej niż się da na obrazie
        idx_closest = 0

        # przejście przez wyszystkie maski
        for i in range(len(boxes)):
            distance = (((boxes[i][0] + boxes[i][2]) / 2 - self.location[0])**2 +
                        ((boxes[i][1] + boxes[i][3]) / 2 - self.location[1])**2)**0.5
            if distance < distance_closest:
                distance_closest = distance
                idx_closest = i
        self.update_object_location(boxes[idx_closest])
        return masks[idx_closest]

    def update_object_location(self, box):
        self.location = [int((box[0] + box[2]) / 2),
                         (int(box[1] + box[3]) / 2)]

    @staticmethod
    def log_classes(cats):
        predicted_classes_str = ''
        for cat in cats:
            predicted_classes_str += COCO_CLASSES[cat] + ', '
        logging.debug("predicted classes:\n{}".format(predicted_classes_str))

    @staticmethod
    def draw_object(frame, mask):
        frame_mask = np.zeros_like(frame)
        frame_mask[:, :, 0] += frame[:, :, 0] * mask
        frame_mask[:, :, 1] += frame[:, :, 1] * mask
        frame_mask[:, :, 2] += frame[:, :, 2] * mask
        cv2.imshow("chosen object", frame_mask)

    @staticmethod
    def color_object(frame, mask, color):
        """
        colors object based on given mask
        :param frame: image to edit
        :param mask: boolean mask of an object
        :param color: BGR color scalar
        :return: colored image
        """
        alpha = 0.5
        frame_mask = frame.copy()
        frame_mask[mask] = color
        frame = cv2.addWeighted(frame_mask, alpha, frame, 1-alpha, 0, frame)

    @staticmethod
    def sign_object(frame, cat, bbox, color):
        point = (bbox[0], bbox[1])
        text = COCO_CLASSES[cat]
        cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def evaluate_frame(self, frame):
        """
        detects objects on frame and marks them
        :param frame: image to detect objects on
        :return: image with marked detected objects
                 categories, scores, bounding boxes and masks of detected objects
        """
        cats, scores, boxes, masks = self.predict(frame)
        frame = frame.copy()
        colors = self.make_random_colors(len(cats))
        for i in range(len(masks)):
            self.color_object(frame, masks[i], colors[i])
            self.sign_object(frame, cats[i], boxes[i], colors[i])
        return frame, cats, scores, boxes, masks

    @staticmethod
    def make_random_colors(number_of_colors):
        """
        returns a list of colors with granted color diversity
        """
        # minial difference between compared values of color channel, depends on number of colors
        min_difference = int(255/number_of_colors)

        def make_random_color():
            """
            returns a random BGR color tuple
            """
            B = random.randint(0, 255)
            G = random.randint(0, 255 - B)
            R = random.randint(0, 255 - B - G)
            color = (B, G, R)
            return color

        def compare_colors(color_1, color_2):
            """
            if any color channel difference is greater or equal min_difference, return True, otherwise return False
            """
            # could make it some better-looking, really
            difference = (abs(color_1[0] - color_2[0]), abs(color_1[1] - color_2[1]), abs(color_1[2] - color_2[2]))
            for channel in difference:
                if channel >= min_difference:
                    return True
            return False

        colors = []
        while len(colors) < number_of_colors:
            color_new = make_random_color()
            is_color_valid = True
            # compare new color with all existing ones
            for color in colors:
                if not compare_colors(color_new, color):
                    is_color_valid = False
                    break
            if is_color_valid:
                colors.append(color_new)
        return colors


def get_cat_index(cat_name):
    try:
        return COCO_CLASSES.index(cat_name)
    except:
        logging.error("Invalid class name")
        return None
