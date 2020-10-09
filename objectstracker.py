import config

DEBUG = config.DEBUG

class ObjectsTracker:

    def __init__(self):
        self.trackableobjects = []
        if DEBUG:
            print('[DEBUG]: objectstracker module loaded')





