# import dependencies
import cv2
#import numpy as np
#import matplotlib.pyplot as plt

import video_reader
import platesrecognizer
import config
from utils import crop, box_to_coords

# TODO move deepsort to object tracker module
# deep sort imports

# create frame reader, objects detector, objects tracker, plates detector, plates recognizer, database keeper, trackable objects manager
frame_reader = video_reader.Reader(config.VIDEOSOURCE)
plts_recogn = platesrecognizer.PlatesRecognizer()


DEBUG = config.DEBUG
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE

def test_image_detection():
    import objectsdetector
    obj_detector = objectsdetector.ObjectsDetector()

    IMAGE = 'data/images/my_extras/russia_car_plate.jpg'
    im_name = 'ussia_car_plate'
    im_reader = video_reader.Reader(IMAGE)

    ret, frame = im_reader.read()

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    bboxes, scores, names = obj_detector.get_boxes(frame)

    image_h, image_w, _ = frame.shape

    i = 0

    for bbox in bboxes:
        i += 1
        frame_copy = frame.copy()
        coor = bbox.copy()
        # coor[0] = int(coor[0] * image_h)
        # coor[2] = int(coor[2] * image_h)
        # coor[1] = int(coor[1] * image_w)
        # coor[3] = int(coor[3] * image_w)

        y0, x0, y1, x1 = coor[0], coor[1], coor[2], coor[3]
        c0 = (x0, y0)
        c1 = (x1, y1)

        cv2.rectangle(frame, c0, c1, (241, 255, 51), 2)
        cv2.circle(frame, c0, 5, (0, 0, 255), 1)
        cv2.circle(frame, c1, 10, (0, 0, 255), 1)

        bbox = [y0, x0, y1 - y0, x1 - x0, ]

        curr_item_image_crop = crop(frame_copy, bbox)

        cv2.imshow('curr_item_image_crop', curr_item_image_crop)
        cv2.waitKey(DEBUG_IMG_PAUSE)

        cv2.imwrite(im_name + str(i) + '.jpg', curr_item_image_crop)


def plates_detection_test():
    import platesdetector
    plts_detector = platesdetector.PlatesDetector()

    IMAGE = 'data/images/my_extras/one_car_plates2.jpg'
    im_name = 'one_car_plates2'
    im_reader = video_reader.Reader(IMAGE)


    ret, frame = im_reader.read()

    image_h, image_w, _ = frame.shape

    if image_h > 1080:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    plates_detection_successful, plates_box, confidence = \
        plts_detector.detect_plates(frame)

    print(plates_box)

    plates_crop = crop(frame, plates_box)

    cv2.imshow('[INFO, platesdetector]: plates_image'
               , plates_crop)

    cv2.waitKeyEx(0)

    i = 0
    cv2.imwrite(im_name + str(i) + '.jpg', plates_crop)



image = 'data/images/my_extras/one_car_plates20.jpg'

im_reader = video_reader.Reader(image)

ret, frame = im_reader.read()
plates_text = plts_recogn.recoginze_plates(frame)

print(plates_text)

def plates_OCR_test():
    pass