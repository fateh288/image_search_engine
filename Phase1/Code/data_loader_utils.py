import numpy as np
from sklearn import datasets
from skimage.io import imread_collection, imread
from skimage.color import rgb2gray
import glob
import os
import ntpath
import constants

def load_olivetti_dataset_images():
    dataset = datasets.fetch_olivetti_faces(data_home=".\\Datasets\\olivetti_dataset", shuffle=False, random_state=7, download_if_missing=True,
                                          return_X_y=False)
    return dataset['images']


# Place dataset folder in Datasets folder
def load_dataset_images_from_folder(folder_name, type='.jpg',rgb=False):
    col_dir = folder_name+'\\*'+type
    print(col_dir)
    collection = imread_collection(col_dir)
    print(collection[0])
    if rgb is False:
        collection = [rgb2gray(c) for c in collection]
    return collection

def load_dataset_from_folder(folder_name, type='.jpg', as_grey=True):
    dataset = {}
    dataset['images'] = []
    dataset['ids'] = []
    col_dir = constants.dataset_dir+folder_name + '\\*' + type
    print(col_dir)
    for img_path in glob.glob(col_dir):
        image_id = ntpath.splitext(ntpath.basename(img_path))[0]
        img = imread(img_path,as_gray=as_grey)/255.0
        dataset['images'].append(img)
        dataset['ids'].append(image_id)
    dataset['images'] = np.asarray(dataset['images'])
    dataset['ids'] = np.asarray(dataset['ids'])
    return dataset