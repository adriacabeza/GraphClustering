<h1 align="center">:milky_way: Graph Clustering into communities </h1>

[![HitCount](http://hits.dwyl.io/adriacabeza/object-cut.svg)](http://hits.dwyl.io/AlbertSuarez/GraphClustering)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/adriacabeza/GraphClustering)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/adriacabeza/GraphClustering.svg)](https://GitHub.com/adriacabeza/GraphClustering/stargazers/)




We will be using the following graphs from the Stanford Network Analysis Project (SNAP): ca-GrQc, Oregon-1, roadNet-CA, soc-Epinions1, and web-NotreDame (http://snap.stanford.edu/data/index.html). Project description in *project.pdf* and final report in *report.pdf*. 

## Initial example visualization and clustering of the graph ca-GrQc
<p float="center">
  <img src="docs/images/ca-GrQc_kamada_kawai_graph_colormap2clusters.png" width="405"/>
  <img src="docs/images/ca-GrQcSpectralClustering2D.png" width="390"/> 
</p>
<p align="center"> Kamada-Kawai graph visualization of the ca-GrQc graph and Clustering using the Spectral Embedding. </p>

## Statistics of graph datasets
| Graph         | #vertices | #edges  | #clusters |
|---------------|-----------|---------|-----------|
| ca-GrQc       | 4158      | 13428   | 2         |
| Oregon-1      | 10670     | 22002   | 5         |
| soc-Epinions1 | 75877     | 405739  | 10        |
| web-NotreDame | 325729    | 1117563 | 20        |
| roadNet-CA    | 1957027   | 2760388 | 50        |
 
## Run it

### Requirements
Python 3 and install dependencies:
```bash
pip install -r requirements.txt
```

### Recommendations
Usage of [virtualenv](https://realpython.com/blog/python/python-virtual-environments-a-primer/) is recommended for package library / runtime isolation.

### Usage
Run the clustering algorithm from the main Python file *graph_clustering.py*. You can read arguments help and find command examples in *EXPERIMENTS.sh*. List of arguments:

- *seed*: Random seed.
- *iterations*: Number of iterations with different seed.
- *file*: Path of the input graph file.
- *outputs_path*: Path to save the outputs.
- *clustering*: Use "kmeans", "custom_kmeans", "kmeans_sklearn", "xmeans" or "agglomerative".
- *random_centroids*: Random centroids initialization for "custom_kmeans".
- *distance_metric*: Distance metric for "custom_kmeans": "MINKOWSKI", "CHEBYSHEV", "EUCLIDEAN".
- *compute_eig*: Compute eigenvectors or load them.
- *k*: Number of desired clusters.
- *networkx*: Use networkx library for Laplacian.
- *eig_kept*: Number of eigen vectors kept.
- *normalize_laplacian*: Normalize Laplacian.
- *invert_laplacian*: Invert Laplacian.
- *second*: Using only second smallest eigenvector.
- *eig_normalization*: Normalization of eigen vectors by "vertex", "eig" or "None".

## Authors

👤 Álvaro Orgaz Expósito ([alvarorgaz](https://github.com/alvarorgaz))

👤 Adrià Cabeza ([adriacabeza](https://github.com/adriacabeza))
