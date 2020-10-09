import config
import cv2

DEBUG = config.DEBUG

class Locationgetter:
    def __init__(self, frame):
        self.split_point = 0
        self.location_set = False
        self.frame_height, self.frame_width = frame.shape[:2]
        if DEBUG:
            print('[INFO, locationgetter]: locationgetter initialized, '
                  'frame shape, width, height: ', frame.shape
                  , self.frame_width, self.frame_height)

    def set_locations(self):
        # TODO think over implementing location by mask
        if DEBUG:
            print('[DEBUG, locationgetter]: frame width, height: ', self.frame_width, self.frame_height)
        self.split_point = int(self.frame_width/2)
        if DEBUG:
            print('[INFO, locationgetter]: set vertical line split, left is outside')
        self.location_set = True

    def is_object_inside(self, box):

        if not self.location_set:
            print('[INFO, locationgetter]: trying to get object location before '
                  'locations setting. Initializing default locations')
            self.set_locations()

        if DEBUG:
            print(box)
        y0, y1, x0, x1 = box[0], box[2], box[1], box[3]

        horiz_position = int((x0 + x1) / 2)
        vert_position = int((y0 + y1) / 2)

        if horiz_position > self.split_point:
            object_is_inside = False
        else:
            object_is_inside = True

        if DEBUG:
            print('[DEBUG, locationgetter]: object horizontal pozition: '
                  , horiz_position, ', object is inside: ', object_is_inside)

        return object_is_inside


    def get_frame_with_split(self, frame):
        # TODO check if passed frame width and height
        # TODO equals self.frame_width, self.frame_height otherwise resize
        frame_to_return = frame.copy()
        cv2.line(frame_to_return,
                 (self.split_point, 0),
                 (self.split_point, self.frame_height),
                 color=(255, 0, 0),
                 thickness=2)

        cv2.putText(frame_to_return, 'inside'
                , (self.split_point - 70, self.frame_height - 5)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame_to_return, 'outside'
                , (self.split_point + 10, self.frame_height - 5)
                , cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        return frame_to_return