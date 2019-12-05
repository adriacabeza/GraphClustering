# Oregon

python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering kmeans_sklearn --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True

# quan en trobi acabar el programa 
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 100 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric MINKOWSKI --random_centroids True
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 100 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric EUCLIDEAN --random_centroids True
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 100 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True  --distance_metric CHEBYSHEV --random_centroids True


python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric MINKOWSKI --random_centroids False
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric EUCLIDEAN --random_centroids False
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering custom_kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True  --distance_metric CHEBYSHEV --random_centroids False


python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering kmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering xmeans --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True
python3 graph_clustering.py  --file ./data/Oregon-1.txt --iterations 0 --clustering agglomerative --k 5 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True


# ca-Qrc

python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering kmeans_sklearn --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True

# quan en trobi acabar el programa 
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 100 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric MINKOWSKI --random_centroids True
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 100 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric EUCLIDEAN --random_centroids True
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 100 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True  --distance_metric CHEBYSHEV --random_centroids True


python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric MINKOWSKI --random_centroids False
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True --distance_metric EUCLIDEAN --random_centroids False
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering custom_kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True  --distance_metric CHEBYSHEV --random_centroids False


python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering kmeans --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering xmeans --k 2 --eig_kept 5 --second False --eig_normalization None --normalize_laplacian False --networkx True
python3 graph_clustering.py  --file ./data/ca-GrQc.txt --iterations 0 --clustering agglomerative --k 2 --eig_kept 2 --second False --eig_normalization None --normalize_laplacian False --networkx True

