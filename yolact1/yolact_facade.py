import torch
import cv2
import numpy as np

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
            print("aborting")
            return None

        cats, scores, boxes, masks = self.predict(frame)
        if cats is False:
            print("No object found.\naborting")
            return None
        masks_cat_index = np.where(cats == cat_index)
        if len(masks_cat_index[0]) == 0:
            print("Item not found.\naborting")
            return None
        masks_cat = masks[masks_cat_index]
        mask = self.select_mask(masks_cat)
        return mask

    def predict(self, img):
        """
        analizuje jeden obraz
        :param img: np.array; analizowany obraz
        :return: cats: np.array([], int64)
                 scores: np.array([], float32)
                 boxes: np.array([[x, y, w, h]], int64)
                 masks: np.array([[[]]], uint8)
        """
        with torch.no_grad():
            h, w, _ = img.shape

            batch = self.transform(torch.from_numpy(img).cuda().float()[None, ...])
            preds = self.net(batch)

            # You can modify the score threshold to your liking
            cats, scores, boxes, masks = postprocess(preds, w, h, score_threshold=0.15)
            if len(cats.size()) == 0:
                return False
            # changed predicted data to cpu numpy array type to further process
            cats = cats.cpu().numpy().astype('int64')
            scores = scores.cpu().numpy().astype('float32')
            boxes = boxes.cpu().numpy().astype('int64')
            masks = masks.cpu().numpy().astype('uint8')
            return cats, scores, boxes, masks

    @staticmethod
    def select_mask(masks):
        """
        Zwraca jedną maskę ze zbioru
        :param masks: np.array; zbiór masek
        :return: np.array; jedna maska
        """
        return masks[0]

    @staticmethod
    def print_classes(cats):
        print("predicted classes:")
        for cat in cats:
            print(COCO_CLASSES[cat])

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
        print("ERROR: invalid class name.")
        return None
