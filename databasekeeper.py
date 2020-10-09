import pandas as pd
import config

DEBUG = config.DEBUG

class DatabaseKeeper:

    def __init__(self, database=False):
        if database:
            self.database = database
        else:
            self.database = pd.DataFrame(
                columns=[
                    'track_id'
                    , 'class_name'
                    , 'best_plates_ocr'
                    , 'best_plates_conf'
                    , 'detections_qty'
                    , 'frame_idx'
                    , 'frame_time'
                    , 'box'
                    , 'confidence'
                    , 'skipped_plates_detection'
                    , 'plates_detection_successful'
                    , 'plates_car_relative_box'
                    , 'plates_detection_confidence'
                    , 'plates_OCR_successful'
                    , 'plates_text'
                    , 'plates_text_candidate'
                    , 'plates_text_finalized'
                    , 'plates_OCR_confidence'
                    , 'object_is_inside'
                ]
            )

    def update_db(self, item_id=0, time=False, location='unknown', plates='unknown'):

        df = self.database
        row = 0 if pd.isnull(df.index.max()) else df.index.max() + 1
        df.loc[row, 'id'] = item_id
        df.loc[row, 'plates'] = plates
        df.loc[row, 'time'] = time
        df.loc[row, 'location'] = location

        if DEBUG:
            print('[INFO, databasekeeper]: appended row: ', df.loc[row])

    def get_last_location(self, plates):
        db = self.database
        last_loc = db[db['plates']==plates]['time'].max()

        return last_loc

    def make_records(self, data_for_dbkeeper):
        # TODO implement make_records function
        # unpack data
        # record to database
        if DEBUG:
            print('[INFO, databasekeeper]:, records saved to database')
        pass

    def recieve_data(self, data_packed_for_db):
        #check if reseived columns same as db
        db_columns = self.database.columns
        recieved_columns = data_packed_for_db.columns

        # initially 'data_succesfully_received' set to True
        data_succesfully_received = True

        # if columns not match print message and
        # set 'data_succesfully_received' set to False
        if set(db_columns) != set(recieved_columns):
            print('[ERR, databasekeeper]:' +
            'columns, not present in db_columns: '
                , set(recieved_columns).difference(set(db_columns)),
            'columns, not present in received_columns: '
                , set(db_columns).difference(set(recieved_columns)))
            print('[ERR, databasekeeper]: records not saved to db')
            # TODO implement actions when columns not match
            data_succesfully_received = False



        self.database = self.database.append(data_packed_for_db, ignore_index=True)

        if DEBUG:
            print('[INFO, databasekeeper]:, records saved to database')

        return data_succesfully_received

    def save_db_to_csv(self, filename):
        self.database.to_csv(filename)

        if DEBUG:
            print('[INFO, databasekeeper]: db_succefully saved')
