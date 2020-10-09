from tfyolov4 import detect
import config

DEBUG = False # config.DEBUG

class ObjectsDetector:

    def __init__(self):
        # instatinate submodule class
        self.detector = detect.Detector()

        if DEBUG:
            print('[DEBUG]: objectsdetector module loaded')

    def get_boxes(self, frame):
        bboxes, scores, names = self.detector.detect(frame)
        return bboxes, scores, names


