<h1 align="center">:milky_way: GraphClustering: Method to partition a graph</h1>
We will be using the following graphs from the [Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/data/index.html): ca-GrQc, Oregon-1, roadNet-CA, soc-Epinions1, and web-NotreDame

<p float="center">
  <img src="docs/images/ca-GrQc_kamada_kawai_graph_colormap2clusters.png" width="405" />
  <img src="docs/images/ca-GrQcSpectralClustering2D.png" width="390" /> 
</p>

<p align="center">
  Kamada-Kawai graph visualization of the ca-GrQc graph and Clustering using the Spectral Embedding 
</p>


## Dataset Statistics: 

| Graph         | #vertices | #edges  | #clusters |
|---------------|-----------|---------|-----------|
| ca-GrQc       | 4158      | 13428   | 2         |
| Oregon-1      | 10670     | 22002   | 5         |
| soc-Epinions1 | 75877     | 405739  | 10        |
| web-NotreDame | 325729    | 1117563 | 20        |
| roadNet-CA    | 1957027   | 2760388 | 50        |

## Report
-> [https://www.overleaf.com/5514615922jvndkvxytssz](https://www.overleaf.com/5514615922jvndkvxytssz)


### STUFF WE CAN TRY

- Recursive bi-partitioning: use the eigenvector with second smallest eigenvalue (Fiedler vector) to bipartition the graph by finding the best splitting point:
  - Pick a constant value (0, or 0.5).
  - Pick the median value as splitting point.
  - Look for the splitting point that has the minimum Ncut value:
    1. Choose n possible splitting points.
    2. Compute Ncut value.
    3. Pick minimum.
  
- K-way Spectral Algorithm: Take the first k eigenvectors (without the first one) and then use k-means to make clusters using the eigenvectors as features (http://ai.stanford.edu/~ang/papers/nips01-spectral.pdf)


### Requirements

1. Python 3.7+.

### Recommendations
Usage of [virtualenv](https://realpython.com/blog/python/python-virtual-environments-a-primer/) is recommended for package library / runtime isolation.


### Usage

Now we have three different documents:
- Graph Clustering: which uses networkx, a library written in Python with K-way Spectral Algorithm implemented and fancy drawing stuff.
- Graph Clustering faster: which uses graph tool which it is written in C++ so it should be faster and more efficient, with K-way Spectral Algorithm implemented. 
- **Graph Clustering efficent**: after trying to do the largest graphs with the other two versions (failing), I realized that if we want to be able to process web-NotreDame or roadNet-CA we have to use SNAP library prepared to work with large networks, with nothing implemented because I do not even know how to get the Laplacian. To work with this one only the requirements.txt is needed. 


- Install dependencies
	- Install graph-tool (for the faster version)
```bash
pip install -r requirements.txt
```

- Run the clustering algorithm
```bash
[USAGE]
python3 graph_clustering.py [-h] [--file FILE] [--custom CUSTOM]
                           [--random RANDOM]
                           [--normalizeLaplacian NORMALIZELAPLACIAN] [--k K]

arguments:
  --file FILE           PATH_OF_THE_FILE
  --custom CUSTOM       CUSTOM_K_MEANS_BOOLEAN
  --random RANDOM       RANDOM_CENTERS_BOOLEAN
  --normalizeLaplacian  NORMALIZELAPLACIAN
                        NORMALIZED_LAPLACIAN_BOOLEAN
  --k K                 NUMBER_OF_CLUSTERS

```

## Authors

👤 **Alvaro Orgaz and Adrià Cabeza**

-  [@alvarorgaz](https://github.com/alvarorgaz)
- [@adriacabeza](https://github.com/adriacabeza)
