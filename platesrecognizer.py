import my_plates_ocr
import config

DEBUG = config.DEBUG

class PlatesRecognizer:

    def __init__(self):
        pass

    def recoginze_plates(self, plates_image):

        plates_text, recognition_successful = my_plates_ocr.recognize_plates(plates_image)

        if True: #DEBUG:
            print('[INFO, platesrecognizer]: plates recognized: ', plates_text)

        if len(plates_text) == 0:
            recognition_successful = False
        else:
            recognition_successful = True

        # TODO implement confidence
        confidence = -1

        return recognition_successful, plates_text, confidence

