import argparse

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.cluster.vq import kmeans
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D #to make scatter plots in 3D

from numpy import linalg as LA

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./data/oregon1_010331.txt')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--k', type=int, default=5)
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


# Visualize high-dimensional using tSNE
def draw_tSNE(data, y_hat):
    tsne = TSNE(n_components=3, random_state=0) # n_components= number of dimensions
    data3d = tsne.fit_transform(data)
   
    colormap = np.array(['coral', 'lightblue', 'r', 'g','b'])
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d') # for 3d
    
    #fig, ax = plt.subplots()
    for i,y in enumerate(data3d): 
        ax.scatter(y[0], y[1], y[2], color=colormap[y_hat[i]])
        #ax.scatter(y[0], y[1], color=colormap[y_hat[i]])
    plt.show()


# Spectral clustering algorithm using K-means 
def spectral_clustering(A):
    n = np.shape(A)[0]
    print(np.shape(A))
    eigVal, eigVec = get_eig_laplacian(A)

    print('First {} eigenvalues:{}'.format(args.k, eigVal))
    print('First {} eigenvectors:{}'.format(args.k, eigVec))
   
    # normalize those eigenvectors
    rows_norm = LA.norm(eigVec, axis=1, ord=2)
    Y = (eigVec.T /rows_norm).T
    

    # run K-means
    centroids, distortion = kmeans(Y,args.k)
    
    # creating output label vector
    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(args.k)])
        y_hat[i] = np.argmin(dists)
    draw_tSNE(Y, y_hat)
    return y_hat


# Prints information about the graph
def print_info(G):
    print('Number of nodes: {}'.format(len(G)))
    print('Number of edges: {}'.format(G.size()))


# Returns the symmetric normalized Laplacian matrix of a given graph
def laplacian_matrix(A):
    n = np.shape(A)[0]
    D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    return  np.identity(n) - D.dot(A).dot(D) 


# Computes the two(k) smallest(SM) eigenvalues and eigenvectors 
def get_eig_laplacian(A):
    return eigsh(laplacian_matrix(A), k=args.k, which='SM')


# Writes the result to a file
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
    A = nx.to_numpy_matrix(G)
    print('Starting the algorithm')
    print(spectral_clustering(A))
    

if __name__ == '__main__':
    main()
