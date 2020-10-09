# import nesessary packages
import cv2
import config

DEBUG = config.DEBUG

class Reader:

    def __init__(self, source):
        if DEBUG:
            print('[INFO, reader]: reader module loaded')
        # if source:
        self.vs = None
        self.set_source(source)
        # else:
        #     print('[INFO, reader]: videosource not defined, using camera')
        #     self.vs = cv2.VideoCapture(0)
        self.start_frame_number = 0

    def set_start_frame_no(self, frame_no):
        self.vs.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def set_source(self, source):
        # set SOURCE_VID file as source otherwise use camera
        if source:
            self.vs = cv2.VideoCapture(source)
            if DEBUG:
                print('[INFO, reader]: videofile ' + source + ' succesfully opened')
        else:
            print('[ERR, reader]: no source file provided, using camera as source')

    def read(self):
        ret, frame = self.vs.read()
        return ret, frame