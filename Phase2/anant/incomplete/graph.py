import json

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


with open('matrix.csv', 'r') as file:
    SIMILARITY_MATRIX = json.load(file)
SUBJECTS = ['Subject-1', 'Subject-2', 'Subject-3']
N = 4
M = 4


def get_subjects(shape):
    subjects = [f"Subject-{x+1}" for x in range(shape)]

    return subjects


def get_corr_matrix(dataframe, subjects):
    corr = []

    for item in subjects:
        temp = []

        for i in range(len(dataframe[item])):
            if(item != f'Subject-{i+1}'):
                temp.append(
                    [item.split("-")[-1], f'{i+1}', dataframe[item][i]]
                    # [item.split("-")[-1], f'Subject-{i+1}', dataframe[item][i]]
                )

        temp.sort(key=lambda a: a[2])

        for val in temp[:N]:
            corr.append(val)

    return corr


def create_graph(nodes, matrix):
    x = nx.DiGraph()

    for node in nodes:
        x.add_node(node.split("-")[-1])

    for edge in matrix:
        x.add_edge(edge[0], edge[1], weight=edge[2])

    return x


subject_list = get_subjects(np.shape(SIMILARITY_MATRIX)[0])
df = pd.DataFrame(SIMILARITY_MATRIX, columns=subject_list, index=subject_list)
corr = get_corr_matrix(df, subject_list)
G = create_graph(subject_list, corr)
nx.draw(G, with_labels=True)
plt.savefig('../outputs/task-9.png')
plt.show()
