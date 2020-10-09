import config
import datetime

DEBUG = config.DEBUG

class Timegetter:
    def __init__(self):
        if DEBUG:
            print('[INFO, timegetter]: timegetter initialized')

    def get_frame_time(self, frame):

        # TODO make timegetter get time from frame
        return datetime.datetime.now()