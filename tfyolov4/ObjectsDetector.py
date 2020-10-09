from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

class ObjectsDetector:
    def __init__(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)


    def get_boxes(self, frame):
        pass


