from collections import OrderedDict
import numpy as np
import utils
def get_percenatage_similarity_list(matched_blocks: OrderedDict):
    return [(1-np.average(np.array(scores)[:,1]))*100 for image_id, scores in matched_blocks.items()]

def get_matching_blocks_dict(distance:OrderedDict, score: OrderedDict, k=4):
    descending_score = OrderedDict()
    for key, val in score.items():
        descending_score[key] = np.sum(val)

    descending_score = dict(sorted(descending_score.items(), key=lambda item: item[1], reverse=True))

    image_ids_k = []
    top_k = OrderedDict()
    for i, (key, val) in enumerate(descending_score.items()):
        image_ids_k.append(key)
        top_k[key] = score[key]
        if i == k:
            break

    # explanation of top_k
    matched_blocks = OrderedDict()
    for image_id in image_ids_k:
        matched_blocks[image_id] = []

    for image_id in image_ids_k:
        for block_id, dist_list in distance.items():
            for id, dist in dist_list:
                if id == image_id:
                    # print(image_id, block_id, dist_list)
                    matched_blocks[image_id].append((block_id, dist))
    for key,val in matched_blocks.items():
        matched_blocks[key] = sorted(matched_blocks[key],key=lambda item: item[1])
    #matched_blocks = dict(sorted(matched_blocks.items(), key=lambda item: item[1]))
    #print("matched_blocks=",matched_blocks)
    return image_ids_k, matched_blocks
