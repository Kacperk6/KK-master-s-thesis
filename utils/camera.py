import cv2
import logging

WIDTH = 1280
HEIGHT = 480
FPS = 30


def get_video_live():
    logging.info("getting video")
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(5, FPS)
    logging.info("video parameters: resolution: {}x{}; FPS: {}".format(WIDTH, HEIGHT, FPS))
    return cap


def get_video_from_file(file_name):
    logging.info("getting video")
    return cv2.VideoCapture('../../AI_filmy_zdjecia/' + file_name)


def get_image(file_path):
    logging.info("getting single image")
    return cv2.imread(file_path)


def split_stereo_image(stereo_image, height, width):
    """
    splits image from stereo camera to 2 separate images
    :param stereo_image: image to split
    :param height: input image height
    :param width: input image width
    :return: [frame_left, frame_right]: split input image
    """
    logging.debug("splitting stereo image")
    frame_left = stereo_image[0:height, 0:int(width/2)]
    frame_right = stereo_image[0:height, int(width / 2): width]
    return frame_left, frame_right


def save_video():
    """
    saves unedited video from camera.
    Designed to be used independently
    """
    cap = get_video_live()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, FPS, (WIDTH, HEIGHT))
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("img", frame)
            out.write(frame)
            if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def cap_frames():
    """
    saves frames from a video
    """
    i = 0

    cap = cv2.VideoCapture('output.avi')
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("img", img)
            key = cv2.waitKey(int(1000 / FPS))
            if key == ord(' '):
                logging.info("saving image")
                cv2.imwrite("{}.png".format(i), img)
                img_l, img_r = split_stereo_image(img, img.shape[0], img.shape[1])
                cv2.imwrite("../../../kamera_kalibracja/lewa/{}.png".format(i), img_l)
                cv2.imwrite("../../../kamera_kalibracja/prawa/{}.png".format(i), img_r)
                i += 1
            if key == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
