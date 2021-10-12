import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_random_matrix(shape):
    a = np.random.rand(shape, shape)
    s = [f"Subject-{_}" for _ in range(shape)]

    for _ in range(np.shape(a)[0]):
        a[_][_] = 1.0

    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1] - 1):
            a[i][j] = a[j][i]

    return a, s


def get_corr_matrix(matrix, df):
    a = matrix
    corr = []

    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            corr.append([f"Subject-{i}", f"Subject-{j}",
                         1000**(df[f"Subject-{i}"][f"Subject-{j}"])])

    return corr


def build_graph(w, lev):
    if (lev > 5):
        return
    for z in corr:
        ind = -1
        if z[0] == w:
            ind = 0
            ind1 = 1
        if z[1] == w:
            ind == 1
            ind1 = 0
        if ind == 0 or ind == 1:
            if str(w) + "_" + str(corr[ind1]) not in existing_edges:
                G.add_node(str(corr[ind]))
                existing_edges[str(w) + "_" + str(corr[ind1])] = 1
                G.add_edge(w, str(corr[ind1]))
                build_graph(corr[ind1], lev+1)


def build_graph_for_all():
    count = 0
    for d in corr:
        if (count > 40):
            return
        if d[0] not in existing_edges:
            G.add_node(str(d[0]))
        if d[1] not in existing_edges:
            G.add_node(str(d[1]))
        G.add_weighted_edges_from([[str(d[0]), str(d[1]), d[2]]])
        count = count + 1




matrix, subject = get_random_matrix(40)

df = pd.DataFrame(matrix, columns=subject, index=subject)

corr = get_corr_matrix(matrix, df)

G = nx.Graph()

existing_edges = {}
existing_nodes = {}

build_graph_for_all()

pos = nx.spring_layout(G, weight='weight')
nx.draw(G, pos=pos, width=2, with_labels=True)

plt.savefig("graph.png")
