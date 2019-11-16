from __future__ import division
from graph_tool.all import *


import argparse
import random

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
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


# Reads the graph
def read_graph(G):
    num_lines = sum([1 if line[0] != '#' else 0  for line in open(args.file)])
    v = set()
    print('Reading graph')
    with open(args.file) as f:
        for line in tqdm(f, total=num_lines):
            if line[0] == '#':
                #It is a comment, skipping line
                pass
            else:
                values = line.split()
                (u, v) = tuple(values)
                G.add_edge(u, v) 
    print('Finished reading graph')


# Score our partitions
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
def custom_kmeans(data, tolerance=0.25, ccore=False):
    if args.random:
        centers = [ [ random.random() for _ in range(args.k) ] for _ in range(args.k) ] #Random center points
    else:
        centers = kmeans_plusplus_initializer(data, args.k).initialize()
    dimension = len(data[0])
    metric = distance_metric(type_metric.EUCLIDEAN) # WE CAN USE OUR DEFINED METRIC TOO
    observer = kmeans_observer()
    kmeans_instance = kmeans(data, centers, tolerance, ccore, observer=observer, metric=metric)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    type_repr = kmeans_instance.get_cluster_encoding();
    encoder = cluster_encoder(type_repr, clusters, data);
    # change representation from index list to label list
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING);
    clusters = encoder.get_clusters()
    return clusters


# Spectral clustering algorithm using K-means 
def spectral_clustering(A):
    n = np.shape(A)[0]
    eigVal, eigVec = get_eig_laplacian(A)

    #Y = np.delete(eigVec, 0, axis=1) # maybe it makes sense to delete the first eigenvector 
    
    rows_norm = LA.norm(eigVec, axis=1, ord=2)
    Y = (eigVec.T /rows_norm).T

    if args.custom:
        print('Running custom kmeans')
        return custom_kmeans(Y) 
    else:
        print('Running euclidean kmeans')
        centroids, distortion = scipy_kmeans(Y,args.k) 

        # creating output label vector
        y_hat = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(args.k)])
            y_hat[i] = np.argmin(dists)
        return y_hat



# Computes the two(k) smallest(SM) eigenvalues and eigenvectors 
def get_eig_laplacian(G):
    return eigsh(laplacian(G, normalized=args.normalizedLaplacian).todense(), k=args.k, which='SM')



# Writes the result to a file TO BE COMPLETED
def write_result(labels):
    print('Results: {}'.format(labels))
    with open(args.file+'_result.txt','w') as f:
        for i,l in enumerate(labels):
            f.write('\t'+str(l)) 
    


# Main function
def main():
    G = Graph(directed=False)
    read_graph(G)
    
    print('Starting the algorithm')
    y_hat = spectral_clustering(G)
    score = score_clustering(adjacency(G).todense(),y_hat)
    
    print('Score of the partition: {}'.format(score))
    write_result(y_hat)


if __name__ == '__main__':
    main()
