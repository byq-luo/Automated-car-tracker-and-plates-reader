# project config file
import time
# data source settings
video_filename = 'VIDEO0170.mp4'
video_source_folder = 'data/video_source/'
VIDEOSOURCE = video_source_folder + video_filename
VIDEOSOURCE_STARTING_FRAME = 0

DATABASE = False

# general debug settings
DEBUG = True

# to switch DEBUG for specific module
DEBUG_MY_PLATES_OCR = False
DEBUG_IMG_PAUSE = 5 # 0 for endless

# Video output settings
IMG_RESIZE_COEF = 0.9
SHOW_OUTPUT_FRAME = True
SHOW_OUTPUT_FRAME_PAUSE = 1
SHOW_OBJECT_LOCATION = True

# Video writing config
WRITE_OUTPUT = True
OUTPUT_FILENAME = 'output_' + video_filename + '_' \
        + time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime()) + '.avi'

# Database settings
DB_RECIEVER_COLUMNS = [
      'videosource_filename'
    , 'frame_idx'
    , 'frame_time'
    , 'track_id'
    , 'class_name'
    , 'box'
    , 'confidence'
    , 'skipped_plates_detection'
    , 'plates_detection_successful'
    , 'plates_car_relative_box'
    , 'plates_detection_confidence'
    , 'plates_OCR_successful'
    , 'plates_text_candidate'
    , 'plates_text_finalized'
    , 'plates_OCR_confidence'
    , 'object_is_inside' ]

DB_FOLDER = 'output/database'
DB_FILENAME = 'db_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.csv'
