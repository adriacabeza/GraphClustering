import os
import argparse
import random
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA;
from scipy.cluster.vq import kmeans as scipy_kmeans;
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.encoder import type_encoding, cluster_encoder
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.agglomerative import agglomerative, type_link
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--iterations', type=int, default=0, help='Number of iterations.')
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help='Path of the input graph file.')
parser.add_argument('--outputs_path', type=str, default='./results/', help='Path of the outputs.')
parser.add_argument('--clustering', default='kmeans', type= str , help='Use "kmeans", "custom_kmeans", "kmeans_sklearn", "xmeans"  or "agglomerative".')
parser.add_argument('--random_centroids', default=True, type=lambda x: (str(x).lower() == 'true'), help='Random KMeans centroids initialization.')
parser.add_argument('--distance_metric', default='EUCLIDEAN', type=str , help='Distance metric: "MINKOWSKI", "CHEBYSHEV", "EUCLIDEAN".')
parser.add_argument('--k', type=int, default=5, help='Number of desired clusters.')
parser.add_argument('--eig_kept', type=int, default=None, help='Number of eigen vectors kept.')
parser.add_argument('--normalize_laplacian', type= lambda x: (str(x).lower() == 'true'), default= True, help='Normalize Laplacian')
parser.add_argument('--second', type=lambda x: (str(x).lower() == 'true'), default=None, help='Using only second smallest eigenvector.')
parser.add_argument('--eig_normalization', type=str, default='vertex', help='Normalization of eigen vectors by "vertex", "eig" or "None".')
args = parser.parse_args()


if args.eig_kept is None:
    args.eig_kept = args.k


# Draw the eigenvectors embedding to a 2D plane or a 3D plane if it was 3 eigenvectors
def draw_eigenvectors(data, y_hat):
    # Dimension Reduction TSNE technique when the data is multidimensional
    if args.k > 3:
        tsne = TSNE(n_components=2, random_state=args.seed) # n_components= number of dimensions
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


# Score our partitions using a graph and its cluster
def score_clustering_graph(G, y_hat):
    print('[*] Starting to calculate score.')
    total = 0
    for i in range(args.k):
        v_isize = len([x for x in y_hat.items() if x[1]==i])  # Size of the cluster i
        print('[*] Starting to check cluster {} of size {}'.format(i, v_isize))
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


# It changes randomly some nodes in order to see if it improves in a Hill Climbing approach
#def brute_force(G, 


# Faster and more customizable kmeans using pyclustering
def custom_kmeans(data, k, tolerance=0.0001, ccore=True):
    # Centroids initalization
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
    kmeans_instance = kmeans(data, centers, ccore, tolerance, observer=observer, metric=metric, seed=args.seed) # Create instance of the algorithm
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    type_repr = kmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    return clusters


# Hierarchical agglomerative clustering
def agglomerative_hierarchical(data, k, ccore=True):
    # Clustering
    agglomerative_instance = agglomerative(data, k, type_link.SINGLE_LINK, ccore) # Create instance of the algorithm
    agglomerative_instance.process()
    clusters = agglomerative_instance.get_clusters()
    type_repr = agglomerative_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    return encoder.get_clusters()


# Xmeans clustering
def xmeans_clustering(data, k, ccore=True):
    # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
    # start analysis.

    #random.seed(args.seed)
    initial_centers = kmeans_plusplus_initializer(data, k).initialize()
    # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
    # number of clusters that can be allocated is 20.
    xmeans_instance = xmeans(data, initial_centers, k, ccore=ccore)
    xmeans_instance.process()
    # Extract clustering results: clusters and their centers
    clusters = xmeans_instance.get_clusters()
    type_repr = xmeans_instance.get_cluster_encoding()
    encoder = cluster_encoder(type_repr, clusters, data)

    # Change representation from index list to label list and return clusters
    encoder.set_encoding(type_encoding.CLUSTER_INDEX_LABELING)
    clusters = encoder.get_clusters()
    return clusters


# Computes the two(k) smallest(SM) eigenvalues and eigenvectors, if we want to do largest magnitude (LM) 
def get_eig_laplacian(G):
    if args.normalize_laplacian:
        return eigsh(nx.normalized_laplacian_matrix(G), k=args.eig_kept, which='SM')
    else: 
        return eigsh(nx.laplacian_matrix(G).astype(float), k=args.eig_kept, which='SM')


# Spectral clustering algorithm using args.clustering method
def spectral_clustering(G):
    # Compute the eigen vectors of the graph
    print('[*] Computing the eigen vectors.')
    eigVal, eigVec = get_eig_laplacian(G)
    if args.eig_normalization=='vertex':
        vertex_norm = LA.norm(eigVec, axis=1, ord=2)
        Y = (eigVec.T/vertex_norm).T
    elif args.eig_normalization=='eig':
        eig_norm = LA.norm(eigVec, axis=0, ord=2)
        Y = eigVec/eig_norm
    elif args.eig_normalization=='None':
        Y = eigVec
    if args.second:
        Y = Y[:,[1]]



    # Cluster the eigen vectors of the graph
    if args.clustering=='agglomerative':
        print('[*] Running agglomerative hierarchical clustering.')
        clusters = agglomerative_hierarchical(Y, args.k) 
    elif args.clustering=='custom_kmeans':
        print('[*] Running customized KMeans clustering.')
        clusters = custom_kmeans(Y, args.k) 
    elif args.clustering=='kmeans':
        print('[*] Running KMeans Euclidean clustering.')
        random.seed(args.seed)
        centroids, distortion = scipy_kmeans(Y,args.k) 
    elif args.clustering=='kmeans_sklearn':
        print('[*] Running KMeans Sklearn.')
        clusters = KMeans(n_clusters= args.k, init='k-means++').fit_predict(Y) 
    elif args.clustering=='xmeans':
        print('[*] Running XMeans clustering.')
        clusters = xmeans_clustering(Y, args.k) 
    
    # Creating output dictionary label vector
    y_hat = dict()
    for i, vertex_ID in enumerate(G.nodes()):
        if args.clustering=='kmeans':
            y_hat[vertex_ID] = np.argmin(np.array([np.linalg.norm(Y[i]-centroids[c]) for c in range(args.k)]))
        else:
            y_hat[vertex_ID]= clusters[i]
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


# Writes the result to a file TO BE COMPLETED
def save_result(G, y_hat, score):
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    edges = {'ca-GrQc':13428,'Oregon-1':22002,'soc-Epinions1':405739,'web-NotreDame':1117563,'roadNet-CA':2760388}
    extra = '_random_centroids_'+str(args.random_centroids)+'_distance_metric_'+args.distance_metric+'_seed_'+str(args.seed) if args.clustering=='custom_kmeans' else ''
    file_output = args.outputs_path+graphID+'_'+str(args.clustering)+extra+'_k_'+str(args.k)+'_eig_kept_'+str(args.eig_kept)+'_score_'+str(round(score, 4)) + "_unique_" + str(np.unique(list(y_hat.values())).shape[0]) + '_second_'+ str(args.second) +'.output'
    with open(file_output, 'w') as f:
        f.write('# '+str(graphID)+' '+str(len(G))+' '+str(edges[graphID])+' '+str(args.k)+'\n')
        for vertex_ID in np.sort([int(x) for x in G.nodes()]):
            f.write(f'{vertex_ID} {y_hat[str(vertex_ID)]}\n')
    print('Results saved in '+file_output)
    return file_output

# Main function
def main():
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()
    best_file = ""
    best_score = 0
    print('[*] Starting the algorithm.')
    y_hat = spectral_clustering(G)
    if np.unique(list(y_hat.values())).shape[0] < args.k:
        pass
    else:
        best_score = score_clustering_graph(G, y_hat)
        print('Score of the clustering: {}'.format(best_score))
        best_file = save_result(G, y_hat, best_score)

    for j in range(12,13):
        args.eig_kept = j
        for i in range(1, args.iterations):
            args.seed = i
            print('[*] Starting the algorithm with seed {}'.format(i))
            y_hat = spectral_clustering(G)
            if np.unique(list(y_hat.values())).shape[0] < args.k:
                pass
            else:
                score = score_clustering_graph(G, y_hat) 
                if score < best_score:
                    print('Score of the clustering: {}'.format(score))
                    best_score = score
                    os.remove(best_file)
                    best_file = save_result(G, y_hat, score)


if __name__ == '__main__':
    main()
