import numpy as np
from collections import OrderedDict

def array2d_to_blocks(array, shape):
    arr_shape = np.shape(array)
    xcut = np.linspace(0,arr_shape[0],shape[0]+1).astype(np.int)
    ycut = np.linspace(0,arr_shape[1],shape[1]+1).astype(np.int)

    blocks = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            blocks.append(array[xcut[i]:xcut[i+1],ycut[j]:ycut[j+1]])

    return np.array(blocks)

def print_dict(dictionary : OrderedDict):
    for key,val in dictionary.items():
        print(key,val)

def combine_dicts_by_key(dict1 : OrderedDict, dict2: OrderedDict):
    new_dict = OrderedDict(dict1)
    for key,val in dict2.items():
        if new_dict.__contains__(key):
            new_dict[key].extend(val)
        else:
            new_dict[key] = [val]
    return new_dict


def filter_dict_sum_tuple_values(dict:OrderedDict):
    for key,val in dict.items():
        list_dict = OrderedDict()
        for id, dist in val:
            list_dict[id] = list_dict[id]+dist if id in list_dict else dist
        dict[key]=list(list_dict.items())
    return dict

def get_coord_from_block_id(block_id, num_blocks_row, window_size):
    row = int(block_id/num_blocks_row)
    col = block_id%num_blocks_row
    start = (window_size*row,window_size * col)
    end = (window_size*row+window_size-1, window_size*col+window_size-1)
    return start, end