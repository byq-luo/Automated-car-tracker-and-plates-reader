from tfyolov4 import detect_plates
from utils import crop
import config
import cv2

DEBUG = config.DEBUG

class PlatesDetector:

    def __init__(self):

        # instationate submodule class
        self.detector = detect_plates.Detector()

        if DEBUG:
            print('[DEBUG]: platesdetector module loaded')

    def detect_plates(self, frame):
        if DEBUG:
            print('[DEBUG, platesdetector.detect_plates: recieved frame for detecting')
        box = self.detector.detect(frame)
        if len(box) != 1:
            print( len(box), "plates boxes detected, can't proceed")
            return False, [], -1

        if DEBUG:
            print('[DEBUG, platesdetector.detect_plates]: box:', box[0])

        # TODO make platesdetector get confidence of detection
        confidence = -1

        # current detector return array of boxes of detected items, so returning first one
        return True, box[0], confidence

