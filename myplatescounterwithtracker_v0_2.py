# import dependencies
import cv2
import numpy as np
from pprint import pprint

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

# TODO move deepsort to object tracker module
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

DEBUG = config.DEBUG
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE
VIDEOSOURCE_STARTING_FRAME = config.VIDEOSOURCE_STARTING_FRAME # skipping first 30 frames because of corrupted videofile
SHOW_OUTPUT_FRAME = config.SHOW_OUTPUT_FRAME
SHOW_OBJECT_LOCATION = config.SHOW_OBJECT_LOCATION
IMG_RESIZE_COEF = config.IMG_RESIZE_COEF
SHOW_OUTPUT_FRAME_PAUSE = config.SHOW_OUTPUT_FRAME_PAUSE
WRITE_OUTPUT = config.WRITE_OUTPUT
OUTPUT_FILENAME = config.OUTPUT_FILENAME
DB_FILENAME = config.DB_FILENAME
DB_FOLDER = config.DB_FOLDER

# create frame reader, objects detector, objects tracker, plates detector, plates recognizer, database keeper, trackable objects manager
frame_reader = video_reader.Reader(config.VIDEOSOURCE)
obj_detector = objectsdetector.ObjectsDetector()
obj_tracker = objectstracker.ObjectsTracker()
plts_detector = platesdetector.PlatesDetector()
plts_recogn = platesrecognizer.PlatesRecognizer()
time_getter = timegetter.Timegetter()
db_kpr = databasekeeper.DatabaseKeeper(config.DATABASE)
to_manager = trackableobjectsmanager.TrackableObjectsManager()

# to initialize location getter first frame required
ret, frame = frame_reader.read()
# ret is True if frame sucesfully returned
if ret:
    # if configured to resize image, perform resize
    if IMG_RESIZE_COEF != 1:
        frame = cv2.resize(frame, None, fx=IMG_RESIZE_COEF, fy=IMG_RESIZE_COEF)
        location_getter = locationgetter.Locationgetter(frame)
# if ret is False, frame is not read and program should be terminated
else:
    print('[ERR, myplatescounter]: cant read frame. Program terminated')
    quit()

# my video source first 30 frames are corrupted, so I skip them
frame_reader.set_start_frame_no(VIDEOSOURCE_STARTING_FRAME)

# initialize video_writer if configured to write output
if WRITE_OUTPUT:
    # import ffmpeg
    #
    # frame_reader.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
    # frame_reader.set(cv2.CAP_PROP_CONVERT_RGB, False)
    # ff_proc = (
    #     ffmpeg
    #         .input('pipe:', format='rawvideo', pix_fmt='gray16le',
    #                s='%sx%s' % (int(frame_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                             , int(frame_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    #                r='60')
    #         .output(OUTPUT_FILENAME, vcodec='ffv1', an=None)
    #         .run_async(pipe_stdin=True)
    # )

    # frame_reader.vs.set(cv2.CV_CAP_PROP_FOURCC)
    frame_height, frame_width = frame.shape[:2]
    video_writer = cv2.VideoWriter(OUTPUT_FILENAME, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (frame_width, frame_height))

# Definition of the parameters for Deepsort
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0

# initialize deep sort
model_filename = 'deep_sort/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

# frames counter
current_frame_idx = VIDEOSOURCE_STARTING_FRAME

# loop while can read frame:
while True:

    # with frame reader read frame
    frame_returned, frame = frame_reader.read()

    # if frame not returned:
    if not frame_returned:
        print("[INFO, myplatescounter]: failed to grab frame. program terminated")

        # from to_manager get all existing track_ids
        all_tracks_ids = to_manager.get_all_track_ids()

        for track_id in list(all_tracks_ids):
            to_manager.choose_best_plates_ocr(track_id)
            data_packed_for_db = to_manager.get_data_packed_for_db(track_id)
            # send data to db_keeper and receive confirmation
            data_received_confirmation = db_kpr.recieve_data(data_packed_for_db)
#            if data_received_confirmation:
            to_manager.delete_object(track_id)

        # database keeper write all data to disk
        # db_filename = DB_FOLDER + DB_FILENAME
        db_kpr.save_db_to_csv(DB_FILENAME)

        break

    # if frame returned
    if DEBUG:
        print('[INFO, myplatescounter]: frame size: ', frame.shape)

    # resize frame if configured
    if IMG_RESIZE_COEF != 1:
        frame = cv2.resize(frame, None, fx=IMG_RESIZE_COEF, fy=IMG_RESIZE_COEF)

    # get time of frame
    frame_time = time_getter.get_frame_time(frame)

    # pass frame to objects detector and
    # get boxes, confidence, class_names of detected objects (DO)
    # boxes returned in format (x0, y0, width, height)
    bboxes, scores, names = obj_detector.get_boxes(frame)
    if DEBUG:
        print('[INFO, myplatescounter]: detected' , len(bboxes), ' boxes')
        print('[INFO, myplatescounter]: boxes:')
        for no, box in enumerate(bboxes):
            print(no, box)

        frame_copy = frame.copy()
        print('[INFO, myplatescounter]: frame size: ', frame_copy.shape)
        image_h, image_w, _ = frame_copy.shape
        print('[INFO, myplatescounter]: frame width ', image_w)

    for bbox in bboxes:
        coor = bbox.copy()

        y0, x0, y1, x1 = coor[0], coor[1], coor[2], coor[3]
        c0 = (x0, y0)
        c1 = (x1, y1)

        if DEBUG:
            print('y0, y1, x0, x1:', y0, y1, x0, x1)

            cv2.rectangle(frame_copy, c0, c1, (241, 255, 51), 2)
            cv2.circle(frame_copy, c0, 5, (0, 0, 255), 1)
            cv2.circle(frame_copy, c1, 10, (0, 0, 255), 1)
            #                cv2.resize(original_image, (800, 600))
            cv2.imshow('myplatescounter: image with boxes and circles', frame_copy)
            cv2.waitKey(DEBUG_IMG_PAUSE)


    # encoding features of detections for DeepSORT
    features = encoder(frame, bboxes)

    # zipping detections for DeepSORT tracker
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)

    if DEBUG:
        print('[DEBUG, myplatescounter] detections before non_max_suppression:', detections)

    detections = [detections[i] for i in indices]

    if DEBUG:
        print('[[DEBUG, myplatescounter] detections: after non_max_suppression', detections)

    # Call the tracker
    tracker.predict()

    # update tracker with current detections and
    # get matched track_id with detections in format dict {detection_idx : track_id}
    detections_with_track_ids = tracker.update(detections)

    if DEBUG:
        print('[DEBUG, myplatescounter]: detections_with_track_ids', detections_with_track_ids)

    # container to store info on objects (detection, track_id etc) in current frame
    viewable_tracks = {} # format dict {track_id : {property : value})
    # packing info about detections in container
    for idx, detection in enumerate(detections):
        viewable_tracks[detections_with_track_ids[idx]] = {
                'box'       : detection.tlwh,
                'confidence': detection.confidence,
                'class'     : detection.class_name
                }

        if DEBUG:
            print('[DEBUG, myplatescounter], added detection: ', idx
                  , 'track: ',  detections_with_track_ids[idx], 'with properties: '
                  , detection.tlwh, detection.confidence, detection.class_name )

    if DEBUG:
        print('[DEBUG, myplatescounter]: viewable_tracks:')
        pprint(viewable_tracks)

    # TODO decide who should make decision when to delete missing object:
    # TODO tracker or trackingobjectsmanager

    if DEBUG:
        # print out tracker's current tracks that are viewable on current frame
        tracker_viewable_tracks = tracker.get_viewable_tracks()
        print('[DEBUG, myplatescounter]: tracker_viewable_tracks:')
        pprint(tracker_viewable_tracks)

    # for each currently viewable track proceed to plates
    # detection and recognition
    for track_id, track_props in viewable_tracks.items():

        # ask trackableobjectsmanager if track should be passed
        # to plates detection and recognition (if, for example
        # plates are confidently recognized, no more need for
        # next steps to perform any more)
        if to_manager.skip_plates_detection(track_id):
            if DEBUG:
                print('[INFO, myplatescounter]: track ' +
                      track_id + 'skipped plates detection')
            # TODO maybe write a statement to objecttracker that plates reading skipped?
            viewable_tracks[track_id]['skipped_plates_detection'] = True
            continue

        else:
            viewable_tracks[track_id]['skipped_plates_detection'] = False

        # pass each box to plates detector and save results by trackableobjectsmanager

        # crop object box from whole frame
        curr_item_image_crop = crop(frame, track_props['box'])
        if DEBUG:
            cv2.imshow('[INFO, myplatesdetector]: image_crop of car', curr_item_image_crop)
            cv2.waitKey(DEBUG_IMG_PAUSE)

        # pass cropped image to plates detector and save coordinates of plates boxes
        plates_detection_successful, plates_box, confidence = plts_detector.detect_plates(curr_item_image_crop)

        # if plates detection not successful:
        # 1) make a note to to_manager
        # 2) end detection for current object
        if not plates_detection_successful:
            # to_manager.save_plates_undetected(track_id, frame_time)
            if DEBUG:
                print('[DEBUG, myplatescounter]: no plates detected for track:', track_id)
            viewable_tracks[track_id]['plates_detection_successful'] = False
            viewable_tracks[track_id]['plates_car_relative_box'] = False
            viewable_tracks[track_id]['plates_detection_confidence'] = False
            viewable_tracks[track_id]['plates_OCR_successful'] = False
            viewable_tracks[track_id]['plates_text'] = 'not detected'
            viewable_tracks[track_id]['plates_OCR_confidence'] = -1
            continue

        else:
            viewable_tracks[track_id]['plates_detection_successful'] = True
            viewable_tracks[track_id]['plates_car_relative_box'] = plates_box
            viewable_tracks[track_id]['plates_detection_confidence'] = confidence

        # get crop of plates
        plates_image = crop(curr_item_image_crop, plates_box)
        if DEBUG:
            print('[DEBUG, myplatescounter]: plates detected for track_id:', track_id, ' box: ', plates_box)
            cv2.imshow('[INFO, platesdetector]: plates_image'
                       , plates_image)
            cv2.waitKey(DEBUG_IMG_PAUSE)
            # TODO think to introduce convertion of plates box to frame box

        # perform plates optical character recognition (OCR)
        plts_OCR_successful, plates_text, confidence = plts_recogn.recoginze_plates(plates_image)

        # if ocr successful
        if plts_OCR_successful:
            if DEBUG:
                print('[DEBUG, myplatescounter]: plates recognized for track_id:', track_id, ' plates text: ',
                      plates_text)
            viewable_tracks[track_id]['plates_OCR_successful'] = True
            viewable_tracks[track_id]['plates_text'] = plates_text.rstrip()
            viewable_tracks[track_id]['plates_OCR_confidence'] = confidence

        else:  # if plates not recognized
            if DEBUG:
                print('[INFO, myplatescounter]: plates not recognized fot track_id ', track_id)
            # next line is redundant
            # to_manager.save_plates_not_recognized(track_id)
            viewable_tracks[track_id]['plates_OCR_successful'] = False
            viewable_tracks[track_id]['plates_text'] = 'unrecognizable'
            viewable_tracks[track_id]['plates_OCR_confidence'] = confidence

    # for each currently viewable track get its location,
    # save it in viewable tracks
    for track_id, track_props in viewable_tracks.items():
        # unpack bounding box coordinates
        car_box_coords = [int(value) for value in track_props['box']]

        # pass coordinates to locator and get location in format True/False
        object_is_inside = location_getter.is_object_inside(car_box_coords)
        # save location to objects holder
        viewable_tracks[track_id]['object_is_inside'] = object_is_inside

    # pass info on current detections and tracks to to_manager
    for track_id, track_props in viewable_tracks.items():
        to_manager.save_detection_info(
              current_frame_idx = current_frame_idx
            , frame_time =  frame_time
            , track_id =    track_id
            , class_name =  track_props['class']
            , box =         track_props['box']
            , confidence =  track_props['confidence']
            , skipped_plates_detection =    track_props['skipped_plates_detection']
            , plates_detection_successful = track_props['plates_detection_successful']
            , plates_car_relative_box =     track_props['plates_car_relative_box']
            , plates_detection_confidence = track_props['plates_detection_confidence']
            , plates_OCR_successful =       track_props['plates_OCR_successful']
            , plates_text =                 track_props['plates_text']
            , plates_OCR_confidence =       track_props['plates_OCR_confidence']
            , object_is_inside =            track_props['object_is_inside']
        )

    # get ids of tracks deleted by tracker on last update
    deleted_on_last_update_ids = tracker.get_deleted_on_last_update()
    if DEBUG:
        print('[DEBUG, myplatescounter] deleted_on_last_update: ', deleted_on_last_update_ids)

    # send info on tracks deleted by tracker on last tracker update to to_manager
    for deleted_track_id in deleted_on_last_update_ids:
        to_manager.receive_info_on_deleted_track(deleted_track_id)
        if DEBUG:
            print('[DEBUG, myplatescounter] info on deleted_on_last_update: '
                  , deleted_track_id, 'sent to to_manager')

    # from to_manager get all tracks that are not being tracked,
    # finalize plates candidates,
    # send to db
    # delete from to_manager
    track_ids_not_being_tracked = to_manager.get_track_ids_not_being_tracked()
    for track_id in track_ids_not_being_tracked:
        to_manager.choose_best_plates_ocr(track_id)
        data_packed_for_db = to_manager.get_data_packed_for_db(track_id)
        # send data to db_keeper and receive confirmation
        data_received_confirmation = db_kpr.recieve_data(data_packed_for_db)
        to_manager.delete_object(track_id)

        if DEBUG:
            print('[DEBUG, myplatescounter]: track info on track_id: '
                  , track_ids_not_being_tracked
                  , 'passed to database and deleted from trackableobjectsmanager')

    if DEBUG:
        print('[DEBUG, myplatescounter]: viewable_tracks after all frame processing steps:')
        pprint(viewable_tracks)


    # This block of code output frames with detections and IDs
    if SHOW_OUTPUT_FRAME or WRITE_OUTPUT:
        frame_output = frame.copy()
        for track_id, track_props in viewable_tracks.items():

            # draw car box
            car_box_coords = [int(value) for value in track_props['box']]

            coor = car_box_coords

            y0, x0, y1, x1 = coor[0], coor[1], coor[2], coor[3]
            c0 = (x0, y0)
            c1 = (x1, y1)

            if DEBUG:
                print('[DEBUG, myplatescounter]: car_box_coords: ', car_box_coords)
            cv2.rectangle(frame_output, c0, c1, (0, 255, 0), 2)

            # draw track_id
            car_x_min, car_y_min = track_props['box'][1], track_props['box'][0]
            cv2.putText(frame_output, 'track_id: ' + str(track_id)
                        , (x0, y0 + 20)  # y0 + 15 to make text inside box
                        , cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # if plates detected draw plates box
            if track_props['plates_detection_successful']:
                plates_relative_x_min = track_props['plates_car_relative_box'][1]
                plates_relative_y_min = track_props['plates_car_relative_box'][0]
                plates_relative_x_max = track_props['plates_car_relative_box'][3]
                plates_relative_y_max = track_props['plates_car_relative_box'][2]

                plates_abs_x_min = int(car_x_min + plates_relative_x_min)
                plates_abs_y_min = int(car_y_min + plates_relative_y_min)
                plates_abs_x_max = int(car_x_min + plates_relative_x_max)
                plates_abs_y_max = int(car_y_min + plates_relative_y_max)
                plates_c0 = (plates_abs_x_min, plates_abs_y_min)
                plates_c1 = (plates_abs_x_max, plates_abs_y_max)

                cv2.rectangle(frame_output, plates_c0, plates_c1, (241, 255, 51), 2)
                # if plates recognized put plates text
                if track_props['plates_OCR_successful']:
                    plates_txt = track_props['plates_text'].rstrip()
                    print('DEBUG: ', track_props['plates_text'], plates_txt)
                    cv2.putText(frame_output, plates_txt
                                , (plates_abs_x_min, plates_abs_y_min)
                                , cv2.FONT_HERSHEY_SIMPLEX, 0.75, (241, 255, 51), 2)

            # show object location
            if SHOW_OBJECT_LOCATION:
                obj_center = (
                    int( (x0 + x1) /2 ),
                    int( (y0 + y1) / 2),
                )
                cv2.circle(frame_output, obj_center, 4, (255, 0, 0), thickness=-1)

                location_text = 'inside ' if track_props['object_is_inside'] else 'outside'

                cv2.putText(frame_output, location_text
                    , (x0, (y0  + 40))  # y0 + 30 to location text second line under track_id text
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # add location line

        if SHOW_OBJECT_LOCATION:
            frame_output = location_getter.get_frame_with_split(frame_output)

        cv2.imshow('output frame', frame_output)
        cv2.waitKey(SHOW_OUTPUT_FRAME_PAUSE)

        if WRITE_OUTPUT:
            video_writer.write(frame_output)
            if DEBUG:
                print('[DEBUG, myplatescounter]: frame written to ', OUTPUT_FILENAME)

    # # calculate frames per second of running detections
    # # fps = 1.0 / (time.time() - start_time)
    # # print("FPS: %.2f" % fps)
    # result = np.asarray(frame)
    # #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #

    if DEBUG:
        print('[INFO, myplatescounter]: end of frame #', current_frame_idx, '\n')

    current_frame_idx += 1