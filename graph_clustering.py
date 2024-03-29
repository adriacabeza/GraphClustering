import os
import time
import os.path
import argparse
import pickle
import random
import resource
import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import eye
from numpy import linalg as LA
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans as scipy_kmeans
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--iterations', type=int, default=0, help='Number of iterations with different seed.')
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help='Path of the input graph file.')
parser.add_argument('--outputs_path', type=str, default='./results/', help='Path to save the outputs.')
parser.add_argument('--clustering', default='kmeans', type=str, help='Use "kmeans", "custom_kmeans", "kmeans_sklearn", "xmeans" or "agglomerative".')
parser.add_argument('--random_centroids', default=True, type=lambda x: (str(x).lower()=='true'), help='Random centroids initialization for "custom_kmeans".')
parser.add_argument('--distance_metric', default='EUCLIDEAN', type=str, help='Distance metric for "custom_kmeans": "MINKOWSKI", "CHEBYSHEV", "EUCLIDEAN".')
parser.add_argument('--compute_eig', type=lambda x: (str(x).lower()=='true'), default=True, help='Compute eigenvectors or load them.')
parser.add_argument('--k', type=int, default=5, help='Number of desired clusters.')
parser.add_argument('--networkx', type=lambda x: (str(x).lower()=='true'), default=True, help='Use networkx library for Laplacian.')
parser.add_argument('--eig_kept', type=int, default=None, help='Number of eigen vectors kept.')
parser.add_argument('--normalize_laplacian', type=lambda x: (str(x).lower()=='true'), default=True, help='Normalize Laplacian.')
parser.add_argument('--invert_laplacian', type=lambda x: (str(x).lower()=='true'), default=False, help='Invert Laplacian.')
parser.add_argument('--second', type=lambda x: (str(x).lower()=='true'), default=None, help='Using only second smallest eigenvector.')
parser.add_argument('--eig_normalization', type=str, default='vertex', help='Normalization of eigen vectors by "vertex", "eig" or "None".')
args = parser.parse_args()


# If eig_kept not defined set as k
if args.eig_kept is None:
    args.eig_kept = args.k


# Faster and more customizable kmeans using pyclustering
def custom_kmeans(data, k, tolerance=0.1, ccore=True):
    # Centroids initialization
    if args.random_centroids:
        random.seed(args.seed)
        centers = [[random.random() for _ in range(data.shape[1])] for _ in range(k)]
    else:
        centers = kmeans_plusplus_initializer(data, k).initialize()

    # Distance metric definition
    if args.distance_metric=='MINKOWSKI':
        metric = distance_metric(type_metric.MINKOWSKI, degree=50)
    if args.distance_metric=='CHEBYSHEV':
        metric = distance_metric(type_metric.CHEBYSHEV)
    if args.distance_metric=='EUCLIDEAN':
        metric = distance_metric(type_metric.EUCLIDEAN)

    # Clustering
    observer = kmeans_observer()
    kmeans_instance = kmeans(data, centers, ccore, tolerance, observer=observer, metric=metric, seed=args.seed)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    type_repr = kmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    print('Custom k-means finalized')
    return clusters


# Hierarchical agglomerative clustering
def agglomerative_hierarchical(data, k, ccore=True):
    # Clustering
    agglomerative_instance = agglomerative(data, k, type_link.SINGLE_LINK, ccore)
    agglomerative_instance.process()
    clusters = agglomerative_instance.get_clusters()
    type_repr = agglomerative_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    print('Agglomerative clustering finalized')
    return clusters


# Xmeans clustering
def xmeans_clustering(data, k, ccore=True):
    # Prepare initial centers (amount of initial centers defines amount of clusters from which X-Means will start analysis).
    random.seed(args.seed)
    initial_centers = kmeans_plusplus_initializer(data, k).initialize()
    xmeans_instance = xmeans(data, initial_centers, k, ccore=ccore) # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum number of clusters that can be allocated is 20.
    xmeans_instance.process()

    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    type_repr = xmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    print('Xmeans clustering took finalized')
    return clusters


# Computes the k smallest(SM) eigenvalues and eigenvectors
def get_eig_laplacian_networkx(G):
    if args.normalize_laplacian:
        normalized_laplacian = nx.normalized_laplacian_matrix(G).astype(float)
        eigVec = eigsh((-1 if args.invert_laplacian else 1)*normalized_laplacian, k=args.eig_kept, which='LM' if args.invert_laplacian else 'SM')
        return eigVec
    else:
        laplacian_matrix = nx.laplacian_matrix(G).astype(float)
        eigVec = eigsh((-1 if args.invert_laplacian else 1)*laplacian_matrix, k=100, which='LM' if args.invert_laplacian else 'SM')
        return eigVec


def get_eig_laplacian(L):
    print('Computing eigenvectors')
    eigVec = eigsh(L.real, k=args.eig_kept, which='SM')
    return eigVec


# Read graph file and create adjacency matrix
def read_graph(file_name):
    with open(file_name, 'r') as f:
        info = f.readline().split()
        n_vertices = int(info[2])

        # Empty adjacency matrix
        A = lil_matrix((n_vertices,n_vertices))

        # Add edges to the adjacency matrix
        edges = []
        for edge in f:
            vertices = edge.split()
            A[int(vertices[0]),int(vertices[1])] = 1
            A[int(vertices[1]),int(vertices[0])] = 1
            edges.append(tuple([int(vertices[1]),int(vertices[0])]))
    return A, n_vertices, edges


# Spectral clustering algorithm using args.clustering method
def spectral_clustering(G):
    # Compute the eigenvectors of the graph
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    file_output = graphID+'_normalized_laplacian'+str(args.normalize_laplacian)+'_invert_'+str(args.invert_laplacian)+'.pickle'
    print('Starting spectral clustering')
    if args.compute_eig and not os.path.exists(file_output):
        print('Starting to compute eigenvectors')
        if args.networkx:
            print("Let's get the eigenvectors")
            eigVal, eigVec = get_eig_laplacian_networkx(G)
        else:
            A, n_vertices, edges = read_graph(args.file)
            D = lil_matrix((n_vertices,n_vertices))
            aux_sum = A.sum(axis=1)
            for i in range(n_vertices):
                D[i,i] = aux_sum[i]
            L = D-A
            eigVal, eigVec = get_eig_laplacian(L)
        print('Eigenvectors calculated, saving in {}'.format(file_output))
        with open(file_output, 'wb') as f:
            pickle.dump(eigVec, f)
    else:
        print('Eigenvectors already calculated')
        with open(file_output, 'rb') as f:
            eigVec = pickle.load(f)

    # Normalize the eigenvectors
    if args.eig_normalization=='vertex':
        vertex_norm = LA.norm(eigVec, axis=1, ord=2)
        Y = (eigVec.T/vertex_norm).T
    elif args.eig_normalization=='eig':
        eig_norm = LA.norm(eigVec, axis=0, ord=2)
        Y = eigVec/eig_norm
    elif args.eig_normalization=='None':
        Y = eigVec

    # Subselect eig_kept eigenvectors
    Y = Y[:, 0:args.eig_kept]
    if args.second:
        Y = Y[:,[1]]

    # Cluster the eigenvectors of the graph
    if args.clustering=='agglomerative':
        print('Running agglomerative hierarchical clustering.')
        clusters = agglomerative_hierarchical(Y, args.k)
    elif args.clustering=='custom_kmeans':
        print('Running customized KMeans clustering.')
        clusters = custom_kmeans(Y, args.k)
    elif args.clustering=='kmeans':
        print('Running KMeans Euclidean clustering.')
        centroids, distortion = scipy_kmeans(Y, args.k)
    elif args.clustering=='kmeans_sklearn':
        print('Running KMeans Sklearn.')
        kmeans = KMeans(n_clusters=args.k, random_state=args.seed)
        kmeans.fit(Y)
        clusters = kmeans.predict(Y)
    elif args.clustering=='xmeans':
        print('Running XMeans clustering.')
        clusters = xmeans_clustering(Y, args.k)

    # Create output dictionary label vector
    y_hat = dict()
    for i, vertex_ID in enumerate(G.nodes()):
        if args.clustering=='kmeans':
            y_hat[vertex_ID] = np.argmin(np.array([np.linalg.norm(Y[i]-centroids[c]) for c in range(args.k)]))
        else:
            y_hat[vertex_ID] = clusters[i]
    return y_hat


# Writes the result to an output file
def save_result(G, y_hat, score):
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    edges = {'ca-GrQc':13428,'Oregon-1':22002,'soc-Epinions1':405739,'web-NotreDame':1117563,'roadNet-CA':2760388}
    extra = '_random_centroids_'+str(args.random_centroids)+'_distance_metric_'+args.distance_metric if args.clustering=='custom_kmeans' else ''
    file_output = args.outputs_path+graphID+'_'+str(args.clustering)+extra+'_k_'+str(args.k)+'_eig_kept_'+str(args.eig_kept)+'_eig_norm'+args.eig_normalization+'_score_'+str(round(score, 4))+'_unique_'+str(np.unique(list(y_hat.values())).shape[0])+'_second_'+str(args.second)+'_invert_laplacian_'+str(args.invert_laplacian)+'_seed_'+str(args.seed)+'.output'
    with open(file_output, 'w') as f:
        f.write('# '+str(graphID)+' '+str(len(G))+' '+str(edges[graphID])+' '+str(args.k)+'\n')
        for vertex_ID in np.sort([int(x) for x in G.nodes()]):
            f.write(f'{vertex_ID} {y_hat[str(vertex_ID)]}\n')
    print('Results saved in '+file_output)
    return file_output


# Score our partitions using a graph and its cluster
def score_clustering_graph(G, y_hat):
    print('Starting to calculate score.')
    total = 0
    for i in range(args.k):
        v_isize = len([x for x in y_hat.items() if x[1]==i]) # Size of the cluster i
        print('Starting to check cluster {} of size {}'.format(i, v_isize))
        if v_isize==0:
            continue
        count = 0
        for vertex_ID, cluster in y_hat.items():
            if cluster==i: # It means we are in a vertex that is inside the cluster i, let's check the number of edges to another clusters
                for neighbor_vertex_ID in G.neighbors(vertex_ID):
                    if y_hat[neighbor_vertex_ID]!=i:
                        count += 1
        total += count/v_isize
    return total


# Main function
def main():
    print()
    # Read graph file
    global_time = time.time()
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()

    # Algorithm
    print('\nStarting the algorithm.')
    y_hat = spectral_clustering(G)
    time_aux = time.time()
    print('All the algorithm took: {:.3f}'.format(time_aux-global_time))
    if np.unique(list(y_hat.values())).shape[0]<args.k:
        best_file, best_score = '', np.inf
    else:
        best_score = score_clustering_graph(G, y_hat)
        best_file = save_result(G, y_hat, best_score)

    if args.random_centroids:
        # Repeat algorithm with different seeds
        for j in range(1, args.iterations):
            args.seed = j
            print('\nStarting the algorithm with seed {}'.format(j))
            y_hat = spectral_clustering(G)
            print(np.unique(list(y_hat.values())).shape[0])
            if np.unique(list(y_hat.values())).shape[0]<args.k:
                pass
            else:
                score = score_clustering_graph(G, y_hat)
                if score<best_score:
                    print('Score of the clustering: {}'.format(score))
                    best_score = score
                    if best_file!='':
                        os.remove(best_file)
                    best_file = save_result(G, y_hat, score)
                    #break


if __name__=='__main__':
    main()
