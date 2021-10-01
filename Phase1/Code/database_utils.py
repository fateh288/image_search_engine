import json

import numpy as np
import shelve
import feature_extractor
from tqdm.auto import tqdm
import constants

class ShelveDB:
    def __init__(self, name):
        self.dbname = constants.temp_dir+name

    def clear_db(self):
        with shelve.open(self.dbname) as db:
            db.clear()

    def add_object(self, key,feature_list):
        assert type(feature_list) == feature_extractor.FeatureList
        with shelve.open(self.dbname) as db:
            db[key] = feature_list

    def print_db(self):
        with shelve.open(self.dbname) as db:
            dkeys = list(db.keys())
            dkeys.sort()
            for key in dkeys:
                print("imageID=",key,"feature_list="+str(db[key]))

    def write_to_file(self, file_name):
        with open(constants.output_dir + file_name, 'w') as file:
            with shelve.open(self.dbname) as db:
                dkeys = list(db.keys())
                dkeys.sort()
                for key in tqdm(dkeys):
                    file.write(repr(db[key]))

    def select(self, id_list=list()):
        result = []
        assert type(id_list) == list
        keys = id_list
        with shelve.open(self.dbname) as db:
            if len(id_list) == 0:
                keys = list(db.keys())
            for key in keys:
                if db.__contains__(str(key)):
                    result.append(db[str(key)])
                else:
                    result.append(None)
                    print(key," not found in db, appending None in results")
        return result