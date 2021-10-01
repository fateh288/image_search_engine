import sys
import numpy as np

import feature_extractor
import database_utils
from tqdm.auto import tqdm
import similarity_measures
from collections import OrderedDict
import similarity_utils
import utils
import display_utils

np.set_printoptions(threshold=sys.maxsize)

def store_image_features_in_database(dataset_name, images, ids):
    print("Extracting features from dataset and storing in shelve database")
    shelve_db = database_utils.ShelveDB(dataset_name)
    shelve_db.clear_db()
    for id,image in tqdm(zip(ids,images)):
        feature = feature_extractor.FeatureList(image=image, id=id)
        feature.add_default_features()
        shelve_db.add_object(id,feature)
    print("Images and features stored in Shelve database")

def write_database_to_file(dataset_name,file_name):
    print("Writing database to file")
    shelve_db = database_utils.ShelveDB(dataset_name)
    shelve_db.write_to_file(file_name)
    print("Check file in Outputs folder")

def print_feature(dataset_name, id, feature_type):

    shelve_db = database_utils.ShelveDB(dataset_name)
    feature = shelve_db.select([id])[0]
    if feature_type == 'color':
        print("color moment flattened feature =",np.array([feature.color_moment_1,feature.color_moment_2,feature.color_moment_3]).flatten())
    elif feature_type == 'hog':
        print("hog flattened feature =",np.array([feature.hog_feature]).flatten())
    elif feature_type == 'elbp':
        print("elbp flattened feature =", np.array([feature.elbp_feature]).flatten())
    else:
        print("feature_type=",feature_type," not found in list")

def get_closest(dataset_name, query_id, feature_type='all',k=4):

    assert feature_type=='color' or feature_type=='hog' or feature_type=='elbp' or feature_type=='all'

    shelve_db = database_utils.ShelveDB(dataset_name)
    query_feature = shelve_db.select([query_id])[0]
    db_features = shelve_db.select()

    distance = OrderedDict()
    score = OrderedDict()
    score_color = OrderedDict()
    score_hog = OrderedDict()
    score_elbp = OrderedDict()
    distance_color = OrderedDict()
    distance_hog = OrderedDict()
    distance_elbp = OrderedDict()

    if feature_type == 'color' or feature_type == 'all':
        distance = OrderedDict()
        score = OrderedDict()
        for block_id in np.arange(db_features[0].hog_feature.shape[0]):
            dist_list = [(db_feature.id,similarity_measures.color_moment_feature_similarity(db_feature.color_moment_1[block_id], query_feature.color_moment_1[block_id],
                                                                        db_feature.color_moment_2[block_id], query_feature.color_moment_2[block_id],
                                                                        db_feature.color_moment_3[block_id], query_feature.color_moment_3[block_id]))
                                                                        for db_feature in db_features]

            distance[block_id] = sorted(dist_list, key=lambda x: x[1])[0:k + 1]
            for i, (image_id, _) in enumerate(distance[block_id]):
                if score.get(image_id) is None:
                    score[image_id] = [k - i + 1]
                else:
                    score[image_id].append(k - i + 1)
        score_color = score
        distance_color =distance

    if feature_type == 'hog' or feature_type == 'all':
        distance = OrderedDict()
        score = OrderedDict()
        query_hog = query_feature.hog_feature
        for block_id in np.arange(db_features[0].hog_feature.shape[0]):
            dist_list = [(db_feature.id,similarity_measures.l_norm_similarity(db_feature.hog_feature[block_id], query_hog[block_id],li=1)) for db_feature in db_features]
            distance[block_id] = sorted(dist_list,key=lambda x:x[1])[0:k+1]
            for i,(image_id,_) in enumerate(distance[block_id]):
                if score.get(image_id) is None:
                    score[image_id] = [k-i+1]
                else:
                    score[image_id].append(k-i+1)
        score_hog = score
        distance_hog = distance

    if feature_type == 'elbp' or feature_type == 'all':
        distance = OrderedDict()
        score = OrderedDict()

        query_elbp = query_feature.elbp_feature

        for block_id in np.arange(db_features[0].hog_feature.shape[0]):
            dist_list = [(db_feature.id,
                          similarity_measures.chi_square(db_feature.elbp_feature[block_id], query_elbp[block_id]))
                         for db_feature in db_features]
            distance[block_id] = sorted(dist_list, key=lambda x: x[1])[0 : k + 1]
            for i, (image_id, _) in enumerate(distance[block_id]):
                if score.get(image_id) is None:
                    score[image_id] = [k - i + 1]
                else:
                    score[image_id].append( k - i + 1)
        score_elbp = score
        distance_elbp = distance

    if feature_type =='all':
        score = utils.combine_dicts_by_key(utils.combine_dicts_by_key(score_color,score_hog),score_elbp)
        combined_dict = utils.combine_dicts_by_key(utils.combine_dicts_by_key(distance_color,distance_hog),distance_elbp)
        distance = utils.filter_dict_sum_tuple_values(combined_dict)

    image_ids_k, matched_blocks = similarity_utils.get_matching_blocks_dict(distance, score)

    return image_ids_k, np.round(similarity_utils.get_percenatage_similarity_list(matched_blocks),3), matched_blocks

def draw_and_display(dataset_name, query_id, image_ids_k, percentage_matches, matched_blocks, save_file_name):
    to_draw_images = list()
    shelve_db = database_utils.ShelveDB(dataset_name)
    query_image = shelve_db.select([query_id])[0].image
    features = shelve_db.select(image_ids_k)
    colors = np.array(list(range(0,255,int(255.0/len(features)))))/255.0
    #print("colors = ",colors)
    for index,feature in enumerate(features):
        to_draw_image = feature.image
        for block_id,_ in matched_blocks[feature.id][0:4]:
            start,end = utils.get_coord_from_block_id(block_id,8,8)
            to_draw_image = display_utils.draw(to_draw_image,start_coord=start, end_coord=end,val=colors[index])
            query_image = display_utils.draw(query_image,start_coord=start, end_coord=end,val=colors[index])
        to_draw_images.append(to_draw_image)
    display_utils.display_similar_images(query_image=query_image,similar_images=to_draw_images,query_image_data=query_id,
                                         similar_images_data=list(zip(image_ids_k,percentage_matches)), save_file_name=save_file_name)
    #display_utils.display_images(to_draw_images)

def task_similarity(folder,query_file_name,dist_type,k):
    save_file_name = "query("+str(folder)+","+str(query_file_name)+","+str(dist_type)+","+str(k)+").png"
    image_ids_k,percentage,matched_blocks = get_closest(folder,query_file_name,dist_type,k=k)

    print((folder,query_file_name,dist_type),"  -->  results = ",list(zip(image_ids_k,percentage)))

    query_img = database_utils.ShelveDB(folder).select([query_file_name])[0].image
    draw_and_display(folder,query_file_name,image_ids_k,percentage,matched_blocks, save_file_name)
