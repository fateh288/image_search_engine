import sys
import numpy as np
import data_loader_utils
import tasks

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':

    folder_name = ['smooth-set2']

    write = True
    if write is True:
        for folder in folder_name:
            dataset = data_loader_utils.load_dataset_from_folder(folder, '.png')
            tasks.store_image_features_in_database(folder, dataset['images'], dataset['ids'])
            tasks.write_database_to_file(folder, folder+'_db.txt')

    queries = [('smooth-set2','image-0','all', 4)]

    for query in queries:
        tasks.task_similarity(query[0], query[1], query[2],query[3])
