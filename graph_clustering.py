import os
import os.path
import argparse
import pickle
import random
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA;
from scipy.cluster.vq import kmeans as scipy_kmeans;
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help='Path of the input graph file.')
parser.add_argument('--iterations', type=int, default=20, help='Number of iterations.')
parser.add_argument('--outputs_path', type=str, default='./results/', help='Path of the outputs.')
parser.add_argument('--clustering', default='kmeans', type= str , help='Use "kmeans", "custom_kmeans", "kmeans_sklearn", "xmeans"  or "agglomerative".')
parser.add_argument('--random_centroids', default=True, type=lambda x: (str(x).lower() == 'true'), help='Random KMeans centroids initialization.')
parser.add_argument('--distance_metric', default='EUCLIDEAN', type=str , help='Distance metric: "MINKOWSKI", "CHEBYSHEV", "EUCLIDEAN".')
parser.add_argument('--subset', type=int, default=2, help='Subset of the eigenvectors that we use.')
parser.add_argument('--k', type=int, default=5, help='Number of desired clusters.')
parser.add_argument('--eig_kept', type=int, default=None, help='Number of eigen vectors kept.')
parser.add_argument('--invert_laplacian', type= lambda x: (str(x).lower() == 'true'), default=False, help='Invert Laplacian')
parser.add_argument('--second', type=lambda x: (str(x).lower() == 'true'), default=False, help='Using only second smallest eigenvector.')
parser.add_argument('--compute_eig', type= lambda x: (str(x).lower() == 'true'), default= True, help='Compute Eigenvectors')
parser.add_argument('--networkx', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use networkx library')
parser.add_argument('--eig_normalization', type=str, default='None', help='Normalization of eigen vectors by "vertex", "eig" or "None".')
args = parser.parse_args()


if args.eig_kept is None:
    args.eig_kept = args.k

# Score our partitions using a graph and its cluster using networkx
def score_clustering_graph(y_hat, G=None):
    if G is None:
        f = open(args.file, 'rb')
        G = nx.read_edgelist(f)
        f.close()
    print('[*] Starting to calculate score.')
    total = 0
    for i in range(args.k):
        v_isize = len([x for x in y_hat.items() if x[1]==i])  # Size of the cluster i
        print('[*] Starting to check cluster {} of size {}'.format(i, v_isize))
        if v_isize==0:
            continue
        count = 0
        for vertex_ID, cluster in y_hat.items():
            if not args.networkx:
                vertex_ID2 = str(vertex_ID)
            else:
                vertex_ID2 = vertex_ID
            if cluster==i: # It means we are in a vertex that is inside the cluster i, let's check the number of edges to another clusters
                for neighbor_vertex_ID in G.neighbors(vertex_ID2):
                    if not args.networkx:
                        neighbor_vertex_ID = int(neighbor_vertex_ID)
                    if y_hat[neighbor_vertex_ID]!=i:
                        count += 1
        total += count/v_isize
    return total


# Faster and more customizable kmeans using pyclustering
def custom_kmeans(data, k, tolerance=0.0001, ccore=True):
    # Centroids initalization
    if args.random_centroids:
        centers = [[random.random() for _ in range(data.shape[1])] for _ in range(k)]
    else:
        centers = kmeans_plusplus_initializer(data, k).initialize()

    # Distance metric definition
    if args.distance_metric=='MINKOWSKI':
        metric = distance_metric(type_metric.MINKOWSKI, degree=1000)
    if args.distance_metric=='CHEBYSHEV':
        metric = distance_metric(type_metric.CHEBYSHEV)
    if args.distance_metric=='EUCLIDEAN':
        metric = distance_metric(type_metric.EUCLIDEAN)

    # Clustering
    observer = kmeans_observer()
    kmeans_instance = kmeans(data, centers, ccore, tolerance, observer=observer, metric=metric) # Create instance of the algorithm
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    type_repr = kmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    return clusters


# Computes the two(k) smallest(SM) eigenvalues and eigenvectors, if we want to do largest magnitude (LM) 
def get_eig_laplacian_networkx(G):
    print('[*] Computing eigenvectors')
    return eigsh(nx.normalized_laplacian_matrix(G).astype(float), k= args.eig_kept, which='LM', sigma=0)


def get_eig_laplacian(L):
    print('[*] Computing eigenvectors')
    return eigsh(L.real, k= args.eig_kept, which='LM', sigma=0)


def read_graph(file_name):
    verts = 0
    with open(file_name, "r") as f:
        info = f.readline().split()
        n_vertices = int(info[2])

        # empty matrix
        A = lil_matrix((n_vertices, n_vertices))
        edges = []

        # Add edges to the adjacency matrix
        for edge in f:
            vertices = edge.split()
            A[int(vertices[0]), int(vertices[1])] = 1
            A[int(vertices[1]), int(vertices[0])] = 1
            edges.append(tuple([int(vertices[1]), int(vertices[0])]))
    return A, n_vertices,edges


# Spectral clustering algorithm using args.clustering method
def spectral_clustering():
    best_score = 100000
    best_file = ''
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    file_output = graphID+'_'+'_eig_norm_'+ str(args.eig_normalization) +'_invert_' + str(args.invert_laplacian) +'.pickle'
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()        

    # We can use networkx or not
    if args.compute_eig:
        if args.networkx:
            # Compute the eigen vectors of the graph
            eigVal, eigVec = get_eig_laplacian_networkx(G)
        else:
            A, n_vertices, edges = read_graph(args.file)
            D = lil_matrix((n_vertices, n_vertices))
            aux_sum = A.sum(axis=1)
            for i in range(n_vertices):
                D[i,i]= aux_sum[i]
            L = D - A
            eigVal, eigVec = get_eig_laplacian(L)
            print(eigVal[:3])
            
        sorted_eigVal = eigVal.argsort()
        eigVec = eigVec[:,sorted_eigVal[::]]
        with open(file_output, 'wb') as f:
            pickle.dump(eigVec,f)  

    else: 
        with open(file_output, 'rb') as f:
            eigVec = pickle.load(f)
    
    for k in range(args.iterations):
        #selected = np.random.choice(a=[False, True], size=(args.eig_kept-args.subset), p=[0.5, 0.5])
        #selected = np.concatenate(([True]*args.subset, selected))
        selected= [True,True,True,False,True,True,False,False, True, True,True,False, False,  True,  True]  
        eigVec2 = eigVec[:,selected]
        print('Eigenvectors after', eigVec2) 

        if args.eig_normalization=='vertex':
            vertex_norm = LA.norm(eigVec2, axis=1, ord=2)
            Y = (eigVec2.T/vertex_norm).T
        elif args.eig_normalization=='eig':
            eig_norm = LA.norm(eigVec2, axis=0, ord=2)
            Y = eigVec2/eig_norm
        elif args.eig_normalization=='None':
            print('no normalization')
            Y = eigVec2

        if args.second:
            Y = Y[:,[1]]


        # Cluster the eigen vectors of the graph
        if args.clustering=='custom_kmeans':
            print('[*] Running customized KMeans clustering.')
            clusters = custom_kmeans(Y, args.k) 
        elif args.clustering=='kmeans':
            print('[*] Running KMeans Euclidean clustering.')
            centroids, distortion = scipy_kmeans(Y,args.k) 
        elif args.clustering=='kmeans_sklearn':
            print('[*] Running KMeans Sklearn.')
            #clusters = KMeans(n_clusters= args.k, init='k-means++').fit_predict(Y)
            Y = Y.real
            print('Eigenvectors real', Y)
            kmeans = KMeans(n_clusters=args.k)
            kmeans.fit(Y)
            clusters = kmeans.labels_ 
            np.savetxt('elmeu.out', clusters)
        # Creating output dictionary label vector
        if args.networkx:
            y_hat = dict()
            for i, vertex_ID in enumerate(G.nodes()):
                if args.clustering=='kmeans':
                    y_hat[vertex_ID] = np.argmin(np.array([np.linalg.norm(Y[i]-centroids[c]) for c in range(args.k)]))
            score = score_clustering_graph(y_hat)
        else:
            y_hat = dict()
            for index,community in enumerate(kmeans.labels_):
                y_hat[index] =community
            score = score_clustering_graph(y_hat)
                          
        if score < best_score:
            best_score = score
            print("Selected eigenvectors: {}".format(selected))
            print('Score of the clustering: {}'.format(best_score))
            if best_file != '':
                os.remove(best_file)
            if args.networkx:   best_file = save_result(y_hat, best_score)
            else:
                graphID = args.file.split('/')[-1].split('.txt')[-2]
                edges = {'ca-GrQc':13428,'Oregon-1':22002,'soc-Epinions1':405739,'web-NotreDame':1117563,'roadNet-CA':2760388}
                extra = '_random_centroids_'+str(args.random_centroids)+'_distance_metric_'+args.distance_metric
                file_output = args.outputs_path+graphID+'_'+str(args.clustering)+extra+'_k_'+str(args.k)+'_eig_kept_'+str(args.eig_kept)+'_eig_norm'+args.eig_normalization+'_score_'+str(round(score, 4)) + "_unique_" + str(np.unique(list(y_hat.values())).shape[0]) + '_second_'+ str(args.second) + '_invert_laplacian_'+ str(args.invert_laplacian) +'.output'

                with open(file_output, 'w') as f:
                    aux = []
                    for index, community in enumerate(kmeans.labels_):
                        aux.append([index, community])
                        aux.sort(key= lambda x: x[0])
                    for x in aux:
                        f.write(str(x[0]) + " " + str(x[1]) + "\n")

        



# Writes the result to a file using networkx
def save_result(y_hat, score):
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    edges = {'ca-GrQc':13428,'Oregon-1':22002,'soc-Epinions1':405739,'web-NotreDame':1117563,'roadNet-CA':2760388}
    extra = '_random_centroids_'+str(args.random_centroids)+'_distance_metric_'+args.distance_metric
    file_output = args.outputs_path+graphID+'_'+str(args.clustering)+extra+'_k_'+str(args.k)+'_eig_kept_'+str(args.eig_kept)+'_eig_norm'+args.eig_normalization+'_score_'+str(round(score, 4)) + "_unique_" + str(np.unique(list(y_hat.values())).shape[0]) + '_second_'+ str(args.second) + '_invert_laplacian_'+ str(args.invert_laplacian) +'.output'
    with open(file_output, 'w') as f:
        f.write('# '+str(graphID)+' '+str(len(G))+' '+str(edges[graphID])+' '+str(args.k)+'\n')
        for vertex_ID in np.sort([int(x) for x in G.nodes()]):
            f.write(f'{vertex_ID} {y_hat[str(vertex_ID)]}\n')
    print('Results saved in '+file_output)
    return file_output



# Main function
def main():
    print('\n[*] Starting the algorithm.')
    spectral_clustering()

if __name__ == '__main__':
    main()
