### messing with os.path to make this package able to load its submodules
### TODO in future refactor or make use virtualenv
import os, sys, inspect
# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import config

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS

from tfyolov4.core import utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
#
# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
# flags.DEFINE_string('output', './detections/', 'path to output folder')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.25, 'score threshold')


class Detector:
    #/ def main(_argv):

    def __init__(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True

        #print('config: ', config)

        #session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.input_size = 416  # /FLAGS.size
        # /images = FLAGS.images

        # load model
        # /    if FLAGS.framework == 'tflite':
        # /            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        # /    else:
        self.saved_model_loaded = tf.saved_model.load('tfyolov4/checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        self.CLASSES = 'tfyolov4/data/classes/coco.names'
        self.infer = self.saved_model_loaded.signatures['serving_default']

    def detect(self, image):
        '''recieves an images and returns array of coordinates
        of detected objetcs in format [[y1, y2, x1, x2], [y1, y2, x1, x2]...]
        Class of detected objects is defined inside of function'''

        original_image = image.copy() #/cv2.imread(image_path)
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (self.input_size, self.input_size))

        print('[DEBUG, detect.py] resized image shape: ', image_data.shape)

        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        batch_data = tf.constant(images_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45, #FLAGS.iou,
            score_threshold=0.25  #FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height


        print('[DEBUG, detect.py] bboxes: ', bboxes)

        original_h, original_w, _ = image.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        # converts to xmin, ymin, xmax, ymax


        print('[DEBUG, detect.py] bboxes after format: ', bboxes)


        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(self.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        ##----

        return bboxes, scores, names


        #print('len(boxes): ', len(boxes), boxes)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        print('[INFO, detect] valid_detections:', valid_detections)

        # converting variables to numpy arrays
        boxes, scores, classes = boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0]

        # get box coords for selected classes

        selected_class = 2  # 2 is a 'car' class in coco dataset
        draw_box = True

        coords_to_return = []

        for i in range(len(boxes)):

            obj_clas = classes[i]
            obj_score = scores[i]
            box = boxes[i]

            # skip classes that are not selected
            if obj_clas != selected_class: continue

            print('[INFO, detect]: clas, score, box: ', obj_clas, obj_score, box)

            image_h, image_w, _ = original_image.shape
    #            if int(obj_clas) < 0 or obj_clas > num_classes: continue

            # print('[INFO, detect]: '
            #       'x0: ', coor[0],
            #       'x1: ', coor[2],
            #       'y0: ', coor[1],
            #       'y1: ', coor[3]
            #      )

            coords_to_return.append(box)

            if draw_box:

                coor = box.copy()
                coor[0] = int(coor[0] * image_h)
                coor[2] = int(coor[2] * image_h)
                coor[1] = int(coor[1] * image_w)
                coor[3] = int(coor[3] * image_w)

                y0, y1, x0, x1 = coor[0], coor[2], coor[1], coor[3]
                c0 = (x0, y0)
                c1 = (x1, y1)

                cv2.rectangle(original_image, c0, c1, (241, 255,51 ), 2)
                cv2.circle(original_image, c0, 5, (0, 0, 255), 1)
                cv2.circle(original_image, c1, 10, (0, 0, 255), 1)
#                cv2.resize(original_image, (800, 600))
                cv2.imshow('detect: image with boxes and circles', original_image)

        print('[INFO, detect]: ', len(coords_to_return), 'boxes  detected')

    #     if draw_box:
    #         # print(item)
    #         image = utils.draw_bbox(original_image, pred_bbox)
    #         # image = utils.draw_bbox(image_data*255, pred_bbox)
    #         image = Image.fromarray(image.astype(np.uint8))
    # #        image.show()
    #         image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #         image = cv2.resize(image, (800, 600))
    #         cv2.imshow('detector', image)
    #         cv2.waitKey(0)
    #     #    cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

        return coords_to_return
#
# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
