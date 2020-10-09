# import dependencies
import video_reader
import objectsdetector
import objectstracker
import databasekeeper
import platesdetector
import platesrecognizer
import trackableobjectsmanager
import locationgetter
import timegetter
import config
from utils import crop, box_to_coords
import cv2

DEBUG = config.DEBUG

# create frame reader, objects detector, objects tracker, plates detector, plates recognizer, database keeper, trackable objects manager
frame_reader = video_reader.Reader(config.VIDEOSOURCE)
obj_detector = objectsdetector.ObjectsDetector()
obj_tracker = objectstracker.ObjectsTracker()
plts_detector = platesdetector.PlatesDetector()
plts_recogn = platesrecognizer.PlatesRecognizer()
time_getter = timegetter.Timegetter()
db_kpr = databasekeeper.DatabaseKeeper(config.DATABASE)
to_manager = trackableobjectsmanager.TrackableObjectsManager()
location_getter = locationgetter.Locationgetter()

# my video source firs 30 frames are corrupted, so I skip them
frame_reader.set_start_frame_no(30)

# loop while can read frame:
while True:

    # with frame reader read frame
    ret, frame = frame_reader.read()
    if not ret:
        print("[INFO, myplatescounter]: failed to grab frame. program termineated")
        break

    # get time of frame
    frame_time = time_getter.get_frame_time(frame)

    # current frame detections holder
    current_set = {}


    # pass frame to objects detector and get boxes of detected objects (DO)
    boxes = obj_detector.get_boxes(frame)
    if DEBUG:
        print('[INFO, myplatescounter]: detected' , len(boxes), ' boxes')
        print('[INFO, myplatescounter]: boxes:')
        for no, box in enumerate(boxes):
            print(no, box)


    # save boxes to current_set
    for no, box in enumerate(boxes):
        current_set[no] = {'box': box}
    if DEBUG:
        print('[INFO, myplatescounter]: current_set: \n')
        for key, value in current_set.items():
            print(key, ': ', value, '\n')


    # pass each box to plates detector and save results in current_set
    for item in current_set:
        if DEBUG:
            print('[ERR DEBUG, myplatescounter]: ', item)
            print('[ERR DEBUG, myplatescounter]: ', current_set[item])

        # crop current item from the original image
#        frame_copy = frame.copy()
        # box is saved in decimal size of height and width, need to convert to
        # curren frame coordinates in pixels
        curr_item_coord = box_to_coords(frame, current_set[item]['box'])
        curr_item_image_crop = crop(frame, curr_item_coord)
        if DEBUG:
            cv2.imshow('[INFO, myplatesdetector]: image_crop of car', curr_item_image_crop)
            cv2.waitKey(20)

        # pass cropped image to plates detector and save coordinates of plates boxes
        plates_detection_successful, plates_box = plts_detector.detect_plates(curr_item_image_crop)
        if plates_detection_successful:
            current_set[item]['plates_box'] = plates_box
            if DEBUG:
                print('[DEBUG, myplatescounter]: plates detected for item:', item, ' box: ', plates_box)
                cv2.imshow('[INFO, platesdetector]: plates_image'
                           , crop(curr_item_image_crop, box_to_coords(plates_box)))
                cv2.waitKeyEx(20)
                # TODO think to introduce convertion of plates box to frame box
        else:
            current_set[item]['plates_box'] = 'not_detected'
            if DEBUG:
                print('[DEBUG, myplatescounter]: no plates detected for item:', item)


    # loop through items in current set and recognize plates for each detected plates box
    for item in current_set:
        if current_set[item]['plates_box'] == 'not_detected':
            current_set[item]['plates'] = 'not_detected'
            if DEBUG:
                print('[DEBUG, myplatescounter]: plates for item:', item, 'not detected')

        else:
            # get current item image from whole frame
            curr_item_box = current_set[item]['box']
            curr_item_image = crop(frame, curr_item_box)

            # get plates image from current item image
            curr_item_plates_box = current_set[item]['plates_box']
            plates_image = crop(curr_item_image, curr_item_plates_box)
            if DEBUG:
                print('[INFO, myplatescounter]: item: ', item)
                print('[INFO, myplatescounter]: current_set[item]: ', current_set[item])
                cv2.imshow('myplatescounter: plates_image', plates_image)
                cv2.waitKey(20)

            # recognize plates by plates recognizer
            recognition_successful, recognized_plates = plts_recogn.recoginze_plates(plates_image)
            if DEBUG:
                print('[INFO, myplatescounter]: recognition successful: ', recognition_successful)
                print('[INFO, myplatescounter]: recognized plates: ', recognized_plates)

            # if nothing recognized pass a note in 'recognized_plates'
            if not recognition_successful:
                recognized_plates = 'not_recognized'

            # save recognition result to current set
            current_set[item]['plates'] = recognized_plates

        # small pause for cv2 to draw all windws

        cv2.waitKey(20)
        print('skipping to new frame')
        continue

    # pass each box to location
    # for item in current_set:
    #     item_location = location_getter.get_item_location(frame, current_set[item]['box'])
    #     current_set[item]['location'] = item_location
    #     if DEBUG:
    #         print('[INFO, myplatescounter]: item, itemlocation: ', item, item_location)

    # with database keeper update info in database with info on TO:
    # while len(current_set) > 0:
    #     item, properties = current_set.popitem()
    #     plates = properties['plates']
    #     time = frame_time
    #     location = properties['location']
    #     item_id = 0   # currently not in use
    #
    #     db_kpr.update_db(item_id = id
    #                      , time = frame_time
    #                      , location = location
    #                      , plates = plates)

