#!/usr/bin/env python
# coding: utf-8

# # Spectral-partitioning

# In[ ]:


import numpy as np
import time
import math
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.sparse import lil_matrix


# In[ ]:


def printTime(text):
    global startTime
    timeTaken = time.time() - startTime
    startTime = time.time()
    print("{} in {:.2f}s".format(text, timeTaken))

# Read the graph with given filename into sparse matrix.
# Grapsh are assumed to be located in folder 'graphs_processed'
def read_file_adjacency(file_name):
    global startTime
    startTime = time.time()
    print('File reading started')
    
    graphs_base = 'data/'

    g_file = open(graphs_base + file_name, "r")
    header = g_file.readline().split()
    graphId = header[1]
    verts = header[2]
    edge_amount = header[3]
    k = header[4]
    print("graph " + graphId + " has " + verts + " vertices, " + edge_amount + " edges and k is: " + k)

    int_verts = int(verts)
    # empty matrix
    adj_matrix = lil_matrix((int_verts, int_verts))
    edges = []

    # Add edges to the adjacency matrix
    for edge in g_file:
        vertices = edge.split()
        adj_matrix[int(vertices[0]), int(vertices[1])] = 1
        adj_matrix[int(vertices[1]), int(vertices[0])] = 1
        edges.append(tuple([int(vertices[1]), int(vertices[0])]))

    g_file.close()

    printTime('Adjacency matrix done')
    return (adj_matrix, edges, int_verts, verts, edge_amount, k, graphId)


# In[ ]:


# Construct a diagonal matrix
def diagonal(adj_matrix, int_verts):
    startTime = time.time()

    diagonal_matrix = lil_matrix((int_verts, int_verts))
    adj_sum = adj_matrix.sum(axis=1)
    for i in range(int_verts):
        diagonal_matrix[i, i] = adj_sum[i]

    printTime('Diagonal calculated')
    return diagonal_matrix


# In[ ]:


# Construct a Laplacian matrix
def laplacian(adj_matrix, diagonal_matrix):
    startTime = time.time()

    L = diagonal_matrix - adj_matrix

    printTime('Laplacian calculated')
    return L


# In[ ]:


def eigens(L, k, k_coefficient, is_soc_Epinions):
    startTime = time.time()

    # When using the soc-Epinions1 graph, we need to use 
    # eigenvalue computation without the shift-invert conversion,
    # and with larger tolerance and calculate smallest
    # eigenvalues (which='SA')
    if is_soc_Epinions:
        eigenValues, eigenVectors = eigsh(L.real, 
                                          k=int(k)*k_coefficient,
                                          which='SA',
                                          tol=0.03)
    else:
        eigenValues, eigenVectors = eigsh(L.real, 
                                          which='LM', 
                                          k=int(k)*k_coefficient, 
                                          sigma=0)

    # sort the eigenvectors by eigenvalues
    sorted_eigenValues = eigenValues.argsort()
    eigenVectors = eigenVectors[:,sorted_eigenValues[::]]

    printTime('Eigenvector calculated')
    return (eigenValues, eigenVectors)


# In[ ]:


# Calculate the k-means model
def k_means(eigenVectors, k, is_roadNet):
    startTime = time.time()
    eigenVectors = eigenVectors.real

    # Use larger tolerance and less iterations with roadNet 
    # (otherwise it's very slow)
    if is_roadNet:
        kmeans = KMeans(n_clusters=int(k), tol=0.01)
    else:
        kmeans = KMeans(n_clusters=int(k), n_init=20, max_iter=400)

    kmeans.fit(eigenVectors)

    printTime('K means calculated')
    return kmeans


# In[ ]:


def result(kmeans, int_verts, adj_matrix, k, edges):
    startTime = time.time()

    communities = []
    for i in range(int(k)):
        communities.append([])

    for index, community in enumerate(kmeans.labels_):
        communities[community].append(index)

    res = objective_function(communities, edges)

    printTime('Objective value calculated')
    return res, communities


# In[ ]:


# Optimixed verion of the objective function
def objective_function(communities, edges):
    sum = 0
    
    community_by_vertex = {}
    for communityIndex, community in enumerate(communities):
        for vertex in community:
            community_by_vertex[vertex] = communityIndex
    
    preprocessed_edges = []
    for edge in edges:
        preprocessed_edges.append((community_by_vertex[edge[0]], community_by_vertex[edge[1]]))
    
    for communityIndex, community in enumerate(communities):
        crossCommunityEdgeCount = 0
        for edge in preprocessed_edges:
            if ((edge[0] == communityIndex and edge[1] != communityIndex) or
               (edge[1] == communityIndex and edge[0] != communityIndex)):
                    crossCommunityEdgeCount += 1
        sum += crossCommunityEdgeCount / len(community)
    
    return sum


# In[ ]:


# Create an output file of the communities to folder 'results',
# where community is array of edges
def output(communities, file_name, graphId, verts, edges, k):
    output = open("results/" + file_name,"w+")
    output.write("# " + graphId + " " + verts + " " + edges + " " + k + "\n")
    for index, community in enumerate(communities):
        for vertice in community:
            output.write(str(vertice) + " " + str(index) + "\n")

    output.close()


# In[ ]:


'''
Run the whole pipeline with sparse matrices, k-means.

Writes outputfile with given output_filename.
Parameters is_soc_Epinions and is_roadNet are boolean flags, 
and we use some more approximations for those graphs.
k_coefficient tells how many times k eigenvectors we calculate.
Param keep_first_n_eigens tells how many first eigenvectors we use in all
iterations when calculating the k-means (other eigenvectors we use with
probability 0.50).
'''
def calculate_with_kmeans(input_file_name,
                          output_filename,
                          random_iterations,
                          is_soc_Epinions,
                          is_roadNet,
                          k_coefficient,
                          keep_first_n_eigens):
    total_startTime = time.time()

    A, edges, int_verts, verts, edgs, k, graphId = read_file_adjacency(input_file_name)
    D = diagonal(A, int_verts)
    L = laplacian(A, D)
    eigenValues, eigenVectors = eigens(L, k, k_coefficient, is_soc_Epinions)

    best_res = math.inf
    # Calculate k-means random_iterations times,
    # each time using different boolean filter for selecting eigenvectors
    # (except take always first random_iterations vectors)
    for i in range(random_iterations):
        select_eigens = np.random.choice(a=[False, True], size=(k_coefficient*int(k)-keep_first_n_eigens), p=[0.5, 0.5])
        select_eigens = np.concatenate(([True]*keep_first_n_eigens, select_eigens))
        filter_eigenVectors = eigenVectors[:,select_eigens]

        k_m = k_means(filter_eigenVectors, k, is_roadNet)
        res, communities = result(k_m, int_verts, A, k, edges)
        if res < best_res:
            print("New best communities: ", select_eigens)
            print("Value is: " + str(res))
            best_filter = select_eigens
            best_res = res
            best_communities = communities

    np.savetxt(output_filename+"_eigen_vectors.csv", eigenVectors.real, delimiter=",")

    output(best_communities, output_filename, graphId, verts, edgs, k)

    timeTaken = time.time() - total_startTime
    print("Whole operation took {:.2f}s".format(timeTaken))

    # save filter to file
    np.savetxt(output_filename+"_filter.csv", best_filter, delimiter=",")
    return eigenValues, eigenVectors, res, communities


# In[ ]:


_, _, _, _ = calculate_with_kmeans(input_file_name='ca-GrQc.txt', 
                                   output_filename='ca-GrQc.output', 
                                   random_iterations=10,
                                   is_soc_Epinions=False, 
                                   is_roadNet=False, 
                                   k_coefficient=6,
                                   keep_first_n_eigens=2)


# In[ ]:


_, _, _, _ = calculate_with_kmeans('Oregon-1.txt', 
                                   'Oregon-1.output', 
                                   random_iterations=10,
                                   is_soc_Epinions=False,
                                   is_roadNet=False,
                                   k_coefficient=3,
                                   keep_first_n_eigens=2)


# In[ ]:


_, _, _, _ =  calculate_with_kmeans('soc-Epinions1.txt',
                                    'soc-Epinions1.output',
                                    random_iterations=10,
                                    is_soc_Epinions=True,
                                    is_roadNet=False,
                                    k_coefficient=1,
                                    keep_first_n_eigens=4)


# In[ ]:


_, _, _, _ =  calculate_with_kmeans('web-NotreDame.txt',
                                    'web-NotreDame.output',
                                    random_iterations=8,
                                    is_soc_Epinions=False,
                                    is_roadNet=False,
                                    k_coefficient=3,
                                    keep_first_n_eigens=20)


# In[ ]:


_, _, _, _ =  calculate_with_kmeans('roadNet-CA.txt',
                                    'roadNet-CA.output',
                                    random_iterations=1,
                                    is_soc_Epinions=False,
                                    is_roadNet=True,
                                    k_coefficient=2,
                                    keep_first_n_eigens=50)

