import argparse

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from tqdm import tqdm
from numpy import linalg as LA
from scipy.cluster.vq import kmeans
from scipy.sparse.linalg import eigsh 

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


# Spectral clustering algorithm using K-means 
def spectral_clustering(A):
    n = np.shape(A)[0]
    print(np.shape(A))
    eigVal, eigVec = get_eig_laplacian(A)

    print('First two eigenvalues:{}'.format(eigVal))
    print('First two eigenvectors:{}'.format(eigVec))
   
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
    return eigsh(laplacian_matrix(A), k=2, which='SM')


# Writes the result to a file
def write_result(labels):
    print('Results: {}'.format(labels))
    with open(args.file+'_result.txt','w') as f:
        for i,l in enumerate(labels):
            f.write('\t'+l) 
    

# Main function
def main():
    G = nx.Graph()
    read_graph(G)
    print_info(G)
    A = nx.to_numpy_matrix(G)
    print('Starting the algorithm')
    write_result(spectral_clustering(A))


if __name__ == '__main__':
    main()
