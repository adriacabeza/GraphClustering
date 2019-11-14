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
        #plt.title('Scatter plot of the eigenvector embeddings')
        plt.show()


# Score our partitions
def score_clustering(A, y_hat):
    total = 0
    n = A.shape[1]
    count = 0
    e = []
    for x in np.nditer(A):
        if not count % n and len(e):
            v_i = len(e)
            sum_e = sum(e)
            e = []
            total += sum_e/v_i
        if x and y_hat[count % n] != y_hat[count//n]:
            e.append(x)  # vertices that are connected to node and their cluster is different
        count += 1
    return total

# Faster and more customizable kmeans using pyclustering
def custom_kmeans(data, tolerance=0.25, ccore=False):
    if args.random:
        centers = [ [ random.random() for _ in range(args.k) ] for _ in range(args.k) ] #Random center points
    else:
        centers = kmeans_plusplus_initializer(data, args.k).initialize()
    dimension = len(data[0])
    metric = distance_metric(type_metric.EUCLIDEAN)
    #metric = distance_metric(type_metric.USER_DEFINED, func=) 
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

    #print('First {} eigenvalues:{}'.format(args.k+1, eigVal))
    #print('First {} eigenvectors:{}'.format(args.k+1, eigVec))
    
    Y = np.delete(eigVec, 0, axis=1) # deleting the first eigenvector which is trivial
    
    rows_norm = LA.norm(Y, axis=1, ord=2)
    Y = (Y.T /rows_norm).T

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


# Drawing the graph. CAUTION: it takes too much time to execute
def draw(G,y_hat):
    print('Starting to draw')
    colors = ['c','m','y','b','w']
    color_map= []
    for i,node in enumerate(G):
        color_map.append(colors[y_hat[i]])
    """
    plt.figure()
    nx.draw_circular(G, with_labels=False, node_size=2, node_color=color_map, linewidth=0.1, alpha=0.1)
    plt.savefig(args.file[:-4]+'_circular_graph_colormap.pdf')
    plt.close()
    """
    plt.figure()
    nx.draw_kamada_kawai(G, with_labels=False,node_color=color_map, node_size=2, linewidth=0.05, alpha=0.1)
    plt.savefig(args.file[:-4]+'_kamada_kawai_graph_colormap.pdf')
    plt.close()


# Prints information about the graph
def print_info(G):
    print('Number of nodes: {}'.format(len(G)))
    print('Number of edges: {}'.format(G.size()))



# Returns the symmetric normalized Laplacian matrix of a given graph
def laplacian_matrix(A):
    n = np.shape(A)[0]
    if args.normalizeLaplacian:
        D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    else:
        D = np.diag(1/ np.ravel(A.sum(axis=0)))
    return  np.identity(n) - D.dot(A).dot(D) 



# Computes the two(k) smallest(SM) eigenvalues and eigenvectors 
def get_eig_laplacian(A):
    return eigsh(laplacian_matrix(A), k=args.k+1, which='SM')



# Writes the result to a file TO BE COMPLETED
def write_result(labels):
    print('Results: {}'.format(labels))
    with open(args.file+'_result.txt','w') as f:
        for i,l in enumerate(labels):
            f.write('\t'+str(l)) 
    

# Main function
def main():
    G = nx.Graph()
    read_graph(G)
    print_info(G)
    A = nx.to_numpy_matrix(G) #adjacency matrix
    print('Starting the algorithm')
    y_hat = spectral_clustering(A)
    score = score_clustering(A,y_hat)
    print('Score of the partition: {}'.format(score))
    #write_result(y_hat)
    #draw(G, y_hat)


if __name__ == '__main__':
    main()
