from time import time

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import dataset


top_features_count=10

type_id_mapping = ['cc', 'con', 'detail', 'emboss', 'jitter', 'neg', 'noise01', 'noise02', 'original', 'poster', 'rot', 'smooth', 'stipple']

def get_type_from_filename(filename):
    return filename.split('-')[1]

json = dataset.open_json('all_database.json')
image_ids = json.keys()
print(image_ids)
# for id in image_ids:
#     print(get_type_from_filename(id))
print(len(image_ids))
n_components = [1,5,20,len(image_ids)-1]

object_feature_mapping = np.array([json[id]['features']['hog'] for id in image_ids])
print(object_feature_mapping.shape)

type_feature_mapping = np.zeros((len(type_id_mapping), object_feature_mapping.shape[1]))
prob_feature_given_type = np.zeros((len(type_id_mapping), object_feature_mapping.shape[1]))
prob_type = np.zeros(len(type_id_mapping))

for id in image_ids:
    type_id = type_id_mapping.index(id.split('-')[1])
    type_feature_mapping[type_id] += json[id]['features']['hog']
    prob_type[type_id]+=1

print("type_sum=",prob_type)
prob_type/=np.sum(prob_type)
print("prob_type=",prob_type,"sum=",np.sum(prob_type)," len=",len(prob_type))

for type_id, type_arr in enumerate(type_feature_mapping):
    for feature_id, feature_arr in enumerate(type_arr):
        prob_feature_given_type[type_id][feature_id] = type_feature_mapping[type_id][feature_id] / np.sum(type_feature_mapping[type_id])

print("type_feature_probs=", prob_feature_given_type)


for num_com in n_components:
    print("LDA for n_components=",num_com)
    lda = LatentDirichletAllocation(n_components=num_com, max_iter=100,
                                    learning_method='online',
                                    learning_offset=10.,
                                    random_state=7,
                                    n_jobs=-1,
                                    evaluate_every=10,
                                    perp_tol=1000,
                                    verbose=1)
    lda.fit(object_feature_mapping)

    print("components=",lda.components_)
    print("components shape=",lda.components_.shape)

    prob_feature_in_topic = np.array([np.sum(lda.components_[:, feature_id]) for feature_id, _ in enumerate(lda.components_[0])])
    prob_feature_in_topic/=np.sum(prob_feature_in_topic)
    print("feature probabilities in component=", prob_feature_in_topic, "sum=", np.sum(prob_feature_in_topic))

    prob_type_given_feature_in_topic = np.zeros((num_com, object_feature_mapping.shape[1],len(type_feature_mapping)))
    for component_id, component in enumerate(lda.components_):
        for feature_id,feature in enumerate(component):
            prob_type_given_feature_in_topic[component_id][feature_id] = [prob_feature_given_type[type_id][feature_id] * prob_type[type_id] / prob_feature_in_topic[feature_id]
                                                                          for type_id, type in enumerate(type_id_mapping)]

    prob_lda_topics = np.sum(lda.components_, axis=1) / np.sum(lda.components_)
    print("topic probabilities=", prob_lda_topics, "sum=", np.sum(prob_lda_topics))
    prob_lda_components_ = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    print("components_probs=", prob_lda_components_," sum=",np.sum(prob_lda_components_,axis=1))

    component_by_type = np.zeros((lda.components_.shape[0],len(type_id_mapping)))
    for component_id, component in enumerate(lda.components_):
        for type_id,type in enumerate(type_id_mapping):
            component_by_type[component_id][type_id] = prob_lda_topics[component_id]*np.sum(prob_lda_components_[component_id]*prob_type_given_feature_in_topic[component_id][:,type_id])
    component_by_type = np.nan_to_num(component_by_type)
    print("component_by_type=",component_by_type,"shape=",component_by_type.shape,"topic scores=",np.sum(component_by_type,axis=1),np.sum(component_by_type))
    ranked_components_by_type = [topic.argsort()[::-1] for topic_idx, topic in enumerate(component_by_type)]
    print("ranked_components_by_type",ranked_components_by_type)


# top_features_ind = [sorted(topic.argsort()[:-top_features_count-1:-1]) for topic_idx, topic in enumerate(lda.components_)]
# print("top_features_ind=",top_features_ind)
# print("inference objects shape=",np.array(object_feature_mapping[0:110:3]).shape)
# transform = lda.transform(object_feature_mapping[0:110:3])
# print("params=",lda.get_params())
# print("transform=",transform)
# print("transform shape=",np.array(transform).shape)
# print("done in %0.3fs." % (time() - t0))