from __future__ import division

import argparse
import random

import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D #to make scatter plots in 3D
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from numpy import linalg as LA
from scipy.cluster.vq import kmeans as scipy_kmeans

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help="PATH_OF_THE_FILE")
parser.add_argument('--custom', default=False, type=lambda x: (str(x).lower() == 'true'), help="CUSTOM_K_MEANS_BOOLEAN")
parser.add_argument('--random', default=True, type=lambda x: (str(x).lower() == 'true'), help="RANDOM_CENTERS_BOOLEAN")
parser.add_argument('--normalizeLaplacian', default=True, type=lambda x: (str(x).lower() == 'true'), help="NORMALIZED_LAPLACIAN_BOOLEAN")
parser.add_argument('--k', type=int, default=5, help="NUMBER_OF_CLUSTERS")
args = parser.parse_args()


# Draw the eigenvectors embedding to a 2D plane or a 3D plane if it was 3 eigenvectors
def draw_eigenvectors(data, y_hat):
    # Dimension Reduction TSNE technique when the data is multidimensional
    if args.k > 3:
        tsne = TSNE(n_components=2, random_state=0) # n_components= number of dimensions
        data = tsne.fit_transform(data)

    colormap = np.array(['coral', 'lightblue', 'r', 'g','b'])
    if args.k == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d') # for 3d
        for i,y in enumerate(data3d): 
            ax.scatter(y[0], y[1], y[2], color=colormap[y_hat[i]])
        plt.show()
    else: 
        fig, ax = plt.subplots()
        for i,y in enumerate(data): 
            ax.scatter(y[0], y[1], color=colormap[y_hat[i]])
        plt.show()


def score_clustering_graph(G, y_hat):
    print('Starting to calculate score')
    total = 0
    for i in range(args.k):
        v_isize = len([x for x in y_hat.items() if x[1] == i])  # size of the cluster i
        print('Starting to check cluster {} of size {}'.format(i, v_isize))
        count = 0
        for key,value in y_hat.items():
             if value == i:
             # it means we are in a vertex that is inside the cluster i, let's check the number of edges to another clusters
                for k in G.neighbors(key):
                    if y_hat[k] != i:
                        count += 1
        total += count/v_isize
    return total 


# Score our partitions using Adjacency Matrix
def score_clustering(A, y_hat):
    print('Starting to calculate score')
    total = 0
    for i in range(args.k):
        v_isize = (y_hat == i).sum() # size of the cluster i
        print('Starting to check cluster {} of size {}'.format(i, v_isize))
        count = 0
        for index,j in enumerate(y_hat):
            if j == i:
                # it means we are in a vertex that is inside the cluster i, let's check the number of edges to another clusters
                for k in range(A.shape[1]):
                    if A.item((index,k)) and  y_hat[k] != i: # if there is an edge and those clusters are different
                        count += 1
        total += count/v_isize
    return total


# Faster and more customizable kmeans using pyclustering
def custom_kmeans(data, tolerance= 0.01, ccore=True):
    if args.random:
        centers = [ [ random.random() for _ in range(args.k) ] for _ in range(args.k) ] #Random center points
    else:
        centers = kmeans_plusplus_initializer(data, args.k).initialize()
    print("number centers", len(centers))
    dimension = len(data[0])
    metric = distance_metric(type_metric.MINKOWSKI, degree=50) # WE CAN USE OUR DEFINED METRIC TOO
    #metric = distance_metric(type_metric.CHEBYSHEV) # WE CAN USE OUR DEFINED METRIC TOO
    #metric = distance_metric(type_metric.EUCLIDEAN) # WE CAN USE OUR DEFINED METRIC TOO
    observer = kmeans_observer()
    kmeans_instance = kmeans(data, centers, ccore, tolerance, observer=observer, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    type_repr = kmeans_instance.get_cluster_encoding();
    encoder = cluster_encoder(type_repr, clusters, data);
    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING);
    clusters = encoder.get_clusters()
    print(clusters)
    return clusters


# Spectral clustering algorithm using K-means 
def spectral_clustering(G):
    n = len(G)
    eigVal, eigVec = get_eig_laplacian(G)

    #eigVec = np.delete(eigVec, 0, axis=1) # maybe it makes sense to delete the first eigenvector which is trivial
    
    rows_norm = LA.norm(eigVec, axis=1, ord=2)
    Y = (eigVec.T /rows_norm).T


    if args.custom:
        print('Running custom kmeans')
        y_hat = dict()
        labels = custom_kmeans(Y) 
        for i,node in enumerate(G.nodes()):
            y_hat[node]= labels[i]
        return y_hat
    else:
        print('Running euclidean kmeans')
        centroids, distortion = scipy_kmeans(Y,args.k) 

        # creating output label vector
        #y_hat = np.zeros(n, dtype=int)

        # creating output dictionary label vector
        y_hat = dict()
        for i, node in enumerate(G.nodes()):
            dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(args.k)])
            y_hat[node] = np.argmin(dists)
        return y_hat


# Drawing the graph. CAUTION: it takes too much time to execute
def draw(G, y_hat):
    print('Starting to draw')
    colors = ['c','m','y','b','w','r','v']
    color_map= []
    for i,node in enumerate(G):
        color_map.append(colors[y_hat[i]])
    
    # Circular graph
    plt.figure()
    nx.draw_circular(G, with_labels=False, node_size=2, node_color=color_map, linewidth=0.1, alpha=0.1)
    plt.savefig(args.file[:-4]+'_circular_graph_colormap.pdf')
    plt.close()
    plt.figure()

    # Kamada Kawai
    nx.draw_kamada_kawai(G, with_labels=False,node_color=color_map, node_size=1, linewidth=0, alpha=1)
    plt.savefig(args.file[:-4]+'_kamada_kawai_graph_colormap.pdf')
    plt.close()
    
    # Spring w/o edges
    plt.figure()
    nx.draw_networkx_nodes(G,pos=nx.spring_layout(G), alpha=1, node_color=color_map,with_labels=False, node_size=1)
    plt.savefig(args.file[:-4]+'_spring_only_nodes_graph.pdf')
    plt.close()
    
    # Spring
    plt.figure(figsize=(20.6,11.6))
    nx.draw_spring(G, with_labels=False, node_color='blue', node_size=1, lidewidth=0.1, alpha=0.1)
    plt.savefig(args.file[:-4]+'_spring_only_nodes_graph.png', dpi=900)
    plt.close()



# Returns the symmetric normalized Laplacian matrix of a given graph
def laplacian_matrix(A):
    n = np.shape(A)[0]
    if args.normalizeLaplacian:
        D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    else:
        D = np.diag(1/ np.ravel(A.sum(axis=0)))
    return  np.identity(n) - D.dot(A).dot(D) 



# Computes the two(k) smallest(SM) eigenvalues and eigenvectors if we want to do largest magnitude (LM) 
def get_eig_laplacian(G):
    return eigsh(nx.normalized_laplacian_matrix(G), k=args.k, which='SM')


# Writes the result to a file TO BE COMPLETED
def write_result(G, labels):
    with open(args.file+'_result.txt','w') as f:
        for node in G.nodes(): #prova
            f.write(f"{node}:{labels[node]}\n")


# Main function
def main():
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()

    #A = nx.to_numpy_matrix(G) #adjacency matrix

    print('Starting the algorithm')
    y_hat = spectral_clustering(G)
    #score = score_clustering(G,y_hat)
    score = score_clustering_graph(G, y_hat)

    print('Score of the partition: {}'.format(score))
    write_result(G,y_hat)


if __name__ == '__main__':
    main()
