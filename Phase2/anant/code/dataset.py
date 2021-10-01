import glob
import ntpath
import json

import constants

from features import get_all_features

from skimage.io import imread
from numpyencoder import NumpyEncoder


# Function to load the images with image ids from the inputs folder
def load_dataset_from_folder():
    dataset = []
    folder_path = "../inputs/" + constants.FOLDER + "/*.png"
    for path in glob.glob(folder_path):
        id = ntpath.splitext(ntpath.basename(path))[0]
        image = imread(path)
        dataset.append({'id': id, 'image': image/255})
    return dataset

def load_dataset_using_Y():
    dataset = []
    folder_path = "../inputs/" + constants.FOLDER + "/" + constants.IMAGE + "-*-" + constants.Y + "-*.png"
    for path in glob.glob(folder_path):
        id = ntpath.splitext(ntpath.basename(path))[0]
        image = imread(path)
        dataset.append({'id': id, 'image': image/255})
    return dataset

# Saves JSON file the given directory
def save_json(data, filename):
    with open(filename, "w") as file:
        jsonString = json.dumps(data, cls=NumpyEncoder)
        file.write(jsonString)

# Opens JSON file from given directory
def open_json(filename):
    with open(filename, 'r') as file:
        data = json.loads(file.readlines()[0])
    return data

# Procesess data after recieving from dataset. Fetches features from the functions
def process_data():
    dataset = []
    # dataset = load_dataset_from_folder()
    if(constants.TASK == "task2"):
        dataset = load_dataset_using_Y()

    data = {}

    for file in dataset:
        id = file['id']
        features = get_all_features(file['image'])
        payload = {
            "image": file['image'],
            "features": {
                "mean": features['mean'],
                "sd": features['sd'],
                "skew": features['skew'],
                "cm": features['mean'] + features['sd'] + features['skew'],
                "elbp": features['elbp'],
                "hog": features['hog']
            }
        }

        data[id] = payload

    save_json(data, constants.FOLDER + "_database.json")

    print("Data Processed.")
