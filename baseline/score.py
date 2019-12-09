import argparse
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help='Path of the input graph file.')
parser.add_argument('--output', type=str,  help='Path of the result file.')
parser.add_argument('--k', type=int, default=5, help='Number of desired clusters.')
args = parser.parse_args()

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

print('Reading Graph')
f = open(args.file, 'rb')
G = nx.read_edgelist(f)
f.close()

print('Reading result')
y_hat = dict()
with open(args.output, 'rb') as f:
        next(f)
        for line in f:
            line_list = line.replace('\n','').split()
            y_hat[line_list[0]] = int(line_list[1])

score = score_clustering_graph(G, y_hat)
print('Score of the baseline: {}'.format(score))
