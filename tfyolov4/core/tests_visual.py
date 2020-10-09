import objectsdetector
import cv2
import config


def test_obj_detector():

    obj_detector = objectsdetector.ObjectsDetector()
    image = 'data/images/my_extras/example.png'
    image = cv2.imread(image)
    bboxes, scores, names = obj_detector.get_boxes(image)

    for bbox in bboxes:

        image_w, image_h = image.shape

        coor = bbox.copy()
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        y0, y1, x0, x1 = coor[0], coor[2], coor[1], coor[3]
        c0 = (x0, y0)
        c1 = (x1, y1)

        cv2.rectangle(image, c0, c1, (241, 255, 51), 2)
        cv2.circle(image, c0, 5, (0, 0, 255), 1)
        cv2.circle(image, c1, 10, (0, 0, 255), 1)
        #                cv2.resize(original_image, (800, 600))
        cv2.imshow('detect: image with boxes and circles',image)
        cv2.waitKey(0)


test_obj_detector()