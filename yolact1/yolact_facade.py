import torch
import cv2
import numpy as np
import logging

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
        cat_index = get_cat_index(cat_name)
        if cat_index is None:
            logging.info("Aborting")
            return None

        cats, scores, boxes, masks = self.predict(frame)
        self.log_classes(cats)
        if cats is False:
            logging.info("No object found. Aborting")
            return None
        masks_cat_index = np.where(cats == cat_index)
        if len(masks_cat_index[0]) == 0:
            logging.info("Given category instance not found. Aborting")
            return None
        masks_cat = masks[masks_cat_index]
        boxes_cat = boxes[masks_cat_index]
        if self.is_first_detection:
            mask = self.select_mask_score(masks_cat, boxes_cat)
            self.is_first_detection = False
        else:
            mask = self.select_mask_location(masks_cat, boxes_cat)
        logging.info("mask found")
        return mask

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
            cats, scores, boxes, masks = postprocess(preds, w, h, score_threshold=0.5)
            if len(cats.size()) == 0:
                return False
            # changed predicted data to cpu numpy array type to further process
            cats = cats.cpu().numpy().astype('int64')
            scores = scores.cpu().numpy().astype('float32')
            boxes = boxes.cpu().numpy().astype('int64')
            masks = masks.cpu().numpy().astype('uint8')
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
            predicted_classes_str += COCO_CLASSES[cat] + '\n'
        logging.debug("predicted classes:\n{}".format(predicted_classes_str))

    @staticmethod
    def draw_object(frame, mask):
        frame_mask = np.zeros_like(frame)
        frame_mask[:, :, 0] += frame[:, :, 0] * mask
        frame_mask[:, :, 1] += frame[:, :, 1] * mask
        frame_mask[:, :, 2] += frame[:, :, 2] * mask
        cv2.imshow("chosen object", frame_mask)


def get_cat_index(cat_name):
    try:
        return COCO_CLASSES.index(cat_name)
    except:
        logging.error("Invalid class name.")
        return None
