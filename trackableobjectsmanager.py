import config
import pandas as pd
DEBUG = config.DEBUG

class TrackableObject:

    def __init__(self, track_id, class_name, location):
        self.track_id = track_id
        self.status = 'tracking'
        self.is_tracking = True
        self.class_name = class_name
        self.is_inside = 'not_set'
        self.current_plates = {'recognized': False,
                       'confidence': 0.0,
                       'current_value': '',
                       'detections_qty': 0
                       }
        self.skip_plates_detection_and_recognition = False
        self.missed_frames = 0   #since_last_detection
        self.detections_qty = 0
        self.timeseries_info = pd.DataFrame(data=None, columns = [
              'frame_idx'
            , 'frame_time'
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
            , 'object_is_inside'
                ])

    # def add_detection(self, bbox, confidence = -1, frame_time = 'n_a'):
    #     self.detections_qty += 1
    #
    #         {'bbox': bbox,
    #         'confidence': confidence,
    #         'frame_time': frame_time}
    #     )
    #
    # def add_plates_detection(self, plates_box, confidence, frame_time):
    #     self.plates['detections'].append(
    #         {'bbox': plates_box,
    #          'confidence': confidence,
    #          'recognized_value': '',
    #          'frame_time': frame_time}
    #     )
    #
    # def save_plates_undetected(self, frame_time):
    #     self.plates['detections'].append(
    #         {'bbox': 'not_detected',
    #          'confidence': 0,
    #          'recognized_value': '',
    #          'frame_time': frame_time}
    #     )
    #
    # def save_plates_text(self, plates_text, confidence, frmae_time):
    #     self.plates['detections']

class TrackableObjectsManager:

    def __init__(self):
        self.trackableobjects = {}
#      self.pd_records = pd.DataFrame()

    def add_to(self, track_id, class_name, object_is_inside):
        # adds new trackable object

        # check if track_id is already registered
        if track_id in self.trackableobjects.keys():
            print("[ERR, trackableobjectsmanager]: can't add track, "
                  "id is already in trackableobjects ")
            print('[ERR, trackableobjectsmanager]: to_id:', track_id, \
                  ' trackableobjects: ', self.trackableobjects)
            return False

        self.trackableobjects[track_id] = TrackableObject(track_id, class_name, object_is_inside)

    # def save_detected_tracks(self, detected_track_id, box, confidence, class_name):
    #     if detected_track_id not in self.trackableobjects.keys():

    def update_statuses(self, viewable_track_id):
        missed = {}
        for to_track_id in self.trackableobjects.keys():
            if to_track_id in to_track_id:
                self.trackableobjects[to_track_id].status = 'tracking'
                self.trackableobjects[to_track_id].missed_frames = 0

            else:
                self.trackableobjects[to_track_id].status = 'missed'
                self.trackableobjects[to_track_id].missed_frames += 1
                missed[to_track_id] = self.trackableobjects[to_track_id].missed_frames

        return missed

    # def add_to_detection(self, to_id, bbox, accuracy, frame_time):
    #
    #     self.trackableobjects[to_id].add_detection(bbox, accuracy, frame_time)
    #
    # def update(self, viewable_tracks, frame_time):
    #     '''
    #
    #         viewable_tracks in format:
    #         track_id = {'box': bbox,        # in format `(min x, miny, max x, max y)`.
    #                     'accuracy': 0.0 }
    #
    #     '''
    #
    #     for track_id, values in viewable_tracks.items():
    #         print('[DEBUG, trackableobjectsmanager] track_id, values:',
    #               track_id, values)
    #
    #         if track_id not in self.trackableobjects:
    #             self.add_to(track_id, values, frame_time)
    #         else:
    #             bbox = values['box']
    #             confidence = values['confidence']
    #             self.add_to_detection(track_id, bbox, confidence, frame_time)
    #
    # def update_deleted(self, deleted_on_last_update_ids):
    #
    #     for deleted_track_id in deleted_on_last_update_ids:
    #         # check if TO in self.trackableobjects
    #         # finalize plates text candidates selection
    #         # save track_id data to database
    #         # delete trackable object from self.trackableobjects
    #
    #         if DEBUG:
    #             print('[INFO, trackableobjectsmanager]: all trackableobjects: '
    #                   , self.trackableobjects)
    #             print('[INFO, trackableobjectsmanager]: list to delete: '
    #                   , deleted_on_last_update_ids)
    #         pass
#
    #     deleted_ids = []
    #     for obj_id in deleted_on_last_update_ids:
    #         if obj_id in self.trackableobjects:
    #             self.delete_object(obj_id)
    #             deleted_ids.append(obj_id)
    #         else:
    #             print('[ERR, trackableobjectsmanager]: cant delete track',
    #                   obj_id, ' as it is not in trackableobjects ')
    #             continue
    #
    #     return deleted_ids


    def delete_object(self, obj_id):
        del self.trackableobjects[obj_id]

    def skip_plates_detection(self, track_id):
        # returns True if plates should not be detected and
        # recognized for the track

        # if detected track is not in trackableobjects return False
        if track_id not in self.trackableobjects.keys():
            if DEBUG:
                print('[DEBUG, trackableobjectsmanager]: track_id ', track_id,
                      ' not in trackableobjects.  skipped plates detection and recognition: ', False )
            return False

        skip = self.trackableobjects[track_id].skip_plates_detection_and_recognition

        if DEBUG:
            print('[DEBUG, trackableobjectsmanager]: track_id ', track_id,
                  ' skipped plates detection and recognition: ', skip )
        return skip

    # def save_plates_bbox(self, track_id, plates_bbox, confidence, frame_time):
    #     self.trackableobjects[track_id].add_plates_detection(
    #         plates_bbox, confidence, frame_time)

    # def save_plates_undetected(self, track_id, frame_time):
    #     self.trackableobjects[track_id].save_plates_undetected(frame_time)

    # def save_plates_text(self, track_id, plates_text, confidence, frame_time):
    #     self.trackableobjects[track_id].save_plates_text(plates_text
    #                                                , frame_time, confidence)

    # def stop_tracking_all_objs(self):
    #     # TODO implement next statements
    #     if DEBUG:
    #         print('stoping tracking all tracks:')
    #     for track_id, track_prop in self.trackableobjects.items():
    #         self.choose_best_plates_ocr(track_id)
    #     if DEBUG:
    #         print('[DEBUG, trackableobjectsmanager]: succesfully stopped tracking track_id:' track_id)
    #         self.trackableobjects[track_id].
    #
        # for each trackable object (TO):
        # define final recognized plates values among all candidates and save
        #  it to 'TO frames records'
        # check if there were frames where TO missed and add these frames
        #  to 'TO frames records'; for such frames calculate location as average
        #  between last known location and first known after missing


    def receive_info_on_deleted_track(self, track_id):
        # mark tracks that were stopped tracking by tracker
        self.trackableobjects[track_id].is_tracking = False
        if DEBUG:
            print('[DEBUG, trackableobjectsmanager]: track_id', track_id
                  , 'is_tracking set to: ', self.trackableobjects[track_id].is_tracking)

    def get_track_ids_not_being_tracked(self):

        track_ids_not_being_tracked = []
        for track_id, track_prop in self.trackableobjects.items():
            if not track_prop.is_tracking:
                track_ids_not_being_tracked.append(track_id)

        if DEBUG:
            print('[DEBUG, trackableobjectsmanager]: track_ids_not_being_tracked:'
                  , track_ids_not_being_tracked)

        return track_ids_not_being_tracked

    # def save_plates_not_recognized(self, track_id):
    #     pass
    #     # TODO implement save_plates_not_recognized
    #     #

    # def match(self, detections):
    #
    #     detections_idxs = detections.keys
    #     to_idxs = [id for self.trackableobjects.id in self.trackableobjects]
    #
    #     unmatched_detections = set(detections_idxs) - set(to_idxs)
    #     unmatched_to = set(to_idxs) - set(detections_idxs)
    #     matched = set(detections_idxs).intersection(to_idxs)
    #
    #     if DEBUG:
    #         print('[INFO, trackableobjectsmanager] match; '
    #           , matched, unmatched_detections, unmatched_to)
    #
    #     return matched, unmatched_detections, unmatched_to
    #
    #
    # def add_to(self, object_id):
    #     self.trackableobjects.append(object_id)

    # def delete_to(self):
    #     pass

    # def set_status(self, trackable_object, status):
    #     self.trackableobjects.status = 'status'



    # def reset_set(self):
    #     self.to_set = {'id': {
    #                         'box': 'unknown',
    #                         'location': 'unknown',
    #                         'plates': 'unknown',
    #                         'detection_time': 'unknown'}
    #                     }
    #
    # def get_set(self):
    #     return self.to_set

    # def save_new_locations(self, boxes):
    #     if len(self.processing_set) > 0:
    #         print("processing set is not empty, can't save new set")
    #         return
    #
    #     for no, box in enumerate(boxes):
    #         self.processing_set[no]['box'] = box
    #         if DEBUG:
    #             print('[INFO trackableobjectsmanager]: new locations saved')
    #
    # def set_plates(self, plates_detector):
    #     for item in self.processing_set:
    #         plates = plates_detector(item.box)


    # TODO consider deletion missed for too long if tracker
    # TODO for some reason hasn't deleted
    # def delete_lost(self, max_days_missed_to_loose):
    #     for to_track_id in self.trackableobjects.keys():

    def save_detection_info(self, current_frame_idx, frame_time
            , track_id, class_name, box, confidence
            , skipped_plates_detection, plates_detection_successful
            , plates_car_relative_box, plates_detection_confidence
            , plates_OCR_successful, plates_text, plates_OCR_confidence
            , object_is_inside):

        if track_id not in self.trackableobjects.keys():
            self.add_to(track_id, class_name, object_is_inside)

        # TODO consider properties values validation
        to_timeseries_columns = [ 'frame_idx'
            , 'frame_time'
            , 'box'
            , 'confidence'
            , 'skipped_plates_detection'
            , 'plates_detection_successful'
            , 'plates_car_relative_box'
            , 'plates_detection_confidence'
            , 'plates_OCR_successful'
            , 'plates_text'
            , 'plates_OCR_confidence'
            , 'object_is_inside']

        track_properties_list = [current_frame_idx
            , frame_time
            , box
            , confidence
            , skipped_plates_detection
            , plates_detection_successful
            , plates_car_relative_box
            , plates_detection_confidence
            , plates_OCR_successful
            , plates_text
            , plates_OCR_confidence
            , object_is_inside]

        for col, val in zip(to_timeseries_columns, track_properties_list):
            self.trackableobjects[track_id].timeseries_info.loc[current_frame_idx, col] = val

        self.trackableobjects[track_id].missied_frames = 0
        self.trackableobjects[track_id].detections_qty += 1

    def update_location_and_direction(self, track_id, is_inside):
        location = 'inside' if is_inside else 'outside'
        if self.trackableobjects[track_id].is_inside != is_inside:
            if DEBUG:
                print('[DEBUG, trackableobjectsmanager]: track_id ', track_id
                      , 'changed location to ', location)
            self.trackableobjects[track_id].is_inside = is_inside

        else:
            if DEBUG:
                print('[DEBUG, trackableobjectsmanager]: track_id ', track_id
                      , 'stays at same location: ', location)

        return

    def choose_best_plates_ocr(self, track_id):
        # TODO implement more advanced algorithm for plates ocr choise
        # get timeseries data on current track_id
        timeseries_info = self.trackableobjects[track_id].timeseries_info

        # get plates_candidates, drop nan
        plates_candidates = timeseries_info['plates_text'].dropna()

        # drop
        print('[ERR, trackableobjectsmanager]: plates_candidates except NaN: \n', plates_candidates)

        # drop values that shouldn't be selected as best ocr
        values_to_drop = ['not detected', 'unrecognizable', '']
        plates_candidates = plates_candidates[~plates_candidates.isin(values_to_drop)]

        if DEBUG:
            print('[DEBUG, trackableobjectsmanager]: plates_candidates ', values_to_drop, ' dropped. Left: ', plates_candidates)

        # if less than 3 recognition simply return last one
        if len(plates_candidates) == 0:
            best_plates_ocr = 'never_recognized'

        elif len(plates_candidates) < 3:
            if DEBUG:
                print(plates_candidates)
            last_value = plates_candidates.iloc[-1]
            best_plates_ocr = last_value

        else:
            # getting most frequent value
            most_freq = plates_candidates.mode().values
            if len(most_freq) > 1:
                # TODO consider using random or other methods e.g. longest
                most_freq = most_freq[0]
            else:
                most_freq = most_freq[0]
            best_plates_ocr = most_freq

        self.trackableobjects[track_id].current_plates['current_value'] = best_plates_ocr

        if DEBUG:
            print('[DEBUG, trackableobjectsmanager]: best plates ocr chosen for track_id '
                  , track_id, 'are: ', best_plates_ocr)

        return best_plates_ocr

    def get_data_packed_for_db(self, track_id):

        # TODO implement next statements
        # select all finished tracking objects (TO)
        # pack all selected TO in container
        # return container

        packed_data = self.trackableobjects[track_id].timeseries_info
        track_id_object = self.trackableobjects[track_id]

        packed_data = packed_data.join(pd.DataFrame(
            {
        'track_id'         : track_id,
        'class_name'       : track_id_object.class_name,
        'best_plates_ocr'  : track_id_object.current_plates['current_value'],
        'best_plates_conf' : track_id_object.current_plates['confidence'],
        'detections_qty'   : track_id_object.detections_qty
            }, index = packed_data.index
        ))

        # TODO implement checking format to fit db receiver method

        if DEBUG:
            print('DEBUG, trackableobjectsmanager], packed data:')
            print(packed_data)

        return packed_data

    def get_all_track_ids(self):
        return self.trackableobjects.keys()
