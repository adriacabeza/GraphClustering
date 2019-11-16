from __future__ import division

import argparse
import random
import snap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.sparse.csgraph import laplacian
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
parser.add_argument('--normalizedLaplacian', default=True, type=lambda x: (str(x).lower() == 'true'), help="NORMALIZED_LAPLACIAN_BOOLEAN")
parser.add_argument('--k', type=int, default=5, help="NUMBER_OF_CLUSTERS")
args = parser.parse_args()


# Score our partitions
def score_clustering(A, y_hat):
    print('Starting to calculate score')
    total = 0
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
def spectral_clustering(G):
    n = G.GetNodes()
    eigVec = get_eigenvectors(G)
    eigVectors = np.array([float(x) for x in eigVec])
    #Y = np.delete(eigVec, 0, axis=1) # maybe it makes sense to delete the first eigenvector 
    print(eigVectors) 
    rows_norm = LA.norm(eigVectors, ord=2)
    Y = (eigVectors.T /rows_norm).T

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


# Computes eigenvectors
def get_eigenvectors(G):
    EigVec = snap.TFltV()
    snap.GetEigVec(G, EigVec)
    return EigVec


# Writes the result to a file
def write_result(G, labels):
    i = 0
    with open(args.file[:-4]+'_result.txt','w') as f:
        for node in G.nodes():
            f.write(str(node.GetId()) +'\t'+str(labels[i]))
            i += 1


# Computes modularity in several ways
def modularity(G):
    print('Clauset-Newman-Moore community detection method for large networks')
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityCNM(G, CmtyV)
    for Cmty in CmtyV:
        print('Community:')
        for NI in Cmty:
            print(NI)
    print('Clauset-Newman-Moore community modularity of the network: {}'.format(modularity))
    
    print('Girvan-Newman community detection algorithm based on betweeness on Graph')
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(G, CmtyV)
    for Cmty in CmtyV:
        print('Community:')
        for NI in Cmty:
            print(NI)
    print('Girvan-Newman community modularity of the network: {}'.format(modularity))



##################################
#  LET'S PRINT SOME INFORMATION  #
##################################
def print_info(G):
    print('Clustering coefficient: {}'.format(snap.GetClustCf(G)))
    print('Degree histogram')
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(G, DegToCntV)
    fig, ax = plt.subplots()
    for i,item in enumerate(DegToCntV):
        ax.bar(int(item.GetVal2()),int(item.GetVal1()), width=30, color='orange')
        if i < 5 or i % 10 == 0:
            ax.text(int(item.GetVal2()),int(item.GetVal1()),str(item.GetVal1()))
    ax.set_yscale('log')
    plt.title('Degree distribution')
    plt.show()
    snap.PrintInfo(G,"Information of {}".format(args.file[:-4]))




# Main function
def main():
    G = snap.LoadEdgeList(snap.PUNGraph, args.file ,0,1)
    print('Starting the algorithm')
    y_hat = spectral_clustering(G) 
    write_result(G,y_hat)

if __name__ == '__main__':
    main()
