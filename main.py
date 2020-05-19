import cv2
from yolact1.yolact_facade import YolactFacade
from utils import camera


def run():
    cap = camera.get_video_live()
    yolact = YolactFacade()
    while True:
        _, img = cap.read()
        cv2.imshow("img", img)
        mask = yolact.run(img, 'person')
        if mask is None:
            print("yolact.run: mask not found")
        else:
            print("yolact.run: mask found!")
            yolact.draw_object(img, mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


run()
