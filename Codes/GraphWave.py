import networkx as nx

import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from karateclub import GraphWave
from karateclub import Role2Vec
import time

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def plot_embeddings(embeddings,g,labels=None):


    X=np.array(g.nodes())

    emb_list = []

    for k in X:

        emb_list.append(embeddings[k])

    emb_list = np.array(emb_list)



    model = TSNE(n_components=2)

    node_pos = model.fit_transform(emb_list)

    plt.scatter(node_pos[X, 0], node_pos[X, 1], label=labels)

    # color_idx = {}
    #
    # for i in range(len(X)):
    #
    #     color_idx.setdefault(Y[i][0], [])
    #
    #     color_idx[Y[i][0]].append(i)


    plt.legend()

    plt.show()

    return node_pos


timestar=time.time()
idx_features_labels = np.genfromtxt('./fujian-nodes-2019.txt',dtype=np.dtype(str), skip_header=1)
edges_unordered = np.genfromtxt('./fujian-Spatial graph-2019.txt',dtype=np.int32, skip_header=1)[:,:2]
labels=idx_features_labels[:, -1]
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)

# nodes=np.array(nodes[:,1],dtype=np.int32)
# edges=np.array(edges[:,2:4],dtype=np.int32)
g=nx.Graph()
b=idx_map.values()
g.add_nodes_from(idx_map.values())
g.add_edges_from(edges)

model = GraphWave()
model.fit(g)
embedding = model.get_embedding()


timeend=time.time()
print("用时：",timeend-timestar,"s")



embedding2d=plot_embeddings(embedding,g)

# np.savetxt('./data/fujian/embedding.txt',embedding)
# np.savetxt('./data/fujian/embedding2d.txt',embedding2d)
