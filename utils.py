import cv2
import config

DEBUG = config.DEBUG
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE

def crop(frame, box):

    if DEBUG:
        print('[INFO, utils]: box:', box)
    box = [int(i) for i in box]
    print('[ERR DEBUG, utils.crop]: box:', box)
    print('[ERR DEBUG, utils.crop]: frame.shape:', frame.shape)

    y0, y1, x0, x1 = box[0], box[2], box[1], box[3]

    # check for x, y values do not exceed frame shape
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    if x0 < 1: x0 = 1
    if y0 < 1: y0 = 1
    if x1 > frame_width: x1 = frame_width
    if y1 > frame_height: y1 = frame_height
    if DEBUG:
        print('[ERR DEBUG, utils.crop]: x0, x1, y0, y1:', y0, x0, y1, x1)

    ymin = min(y0, y1)
    ymax = max(y0, y1)
    xmin = min(x0, x1)
    xmax = max(x0, x1)

    cropped_frame = frame[ymin:ymax, xmin:xmax]

    if DEBUG:
        frame_copy = frame.copy()
        cv2.circle(frame_copy, (x0, y0), 5, (0,0,255), 3)
        cv2.circle(frame_copy, (x1, y1), 10, (0,0,255), 3)
        cv2.imshow('utils.crop: frame with circles', frame_copy)
        cv2.imshow('utils.crop: cropped frame', cropped_frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    return cropped_frame

def box_to_coords(frame, box):

    image_h, image_w, _ = frame.shape

    box[0] = int(box[0] * image_h)
    box[2] = int(box[2] * image_h)
    box[1] = int(box[1] * image_w)
    box[3] = int(box[3] * image_w)

    y0, y1, x0, x1 = box[0], box[2], box[1], box[3]

    return box
