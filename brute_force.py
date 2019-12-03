import random
import argparse
import networkx as nx
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='./data/Oregon-1.txt', help='Path of the input graph file.')
parser.add_argument('--output_file', type=str, default='./data/Oregon-1.txt', help='Path of the output file.')
parser.add_argument('--k', type=int, default=5, help='Number of desired clusters.')
parser.add_argument('--iterations', type=int, default=10)
args = parser.parse_args()


# Writes the result to a file TO BE COMPLETED
def save_result(G, y_hat, score):
    graphID = args.file.split('/')[-1].split('.txt')[-2]
    edges = {'ca-GrQc':13428,'Oregon-1':22002,'soc-Epinions1':405739,'web-NotreDame':1117563,'roadNet-CA':2760388}
    file_output = output_file.replace('.output', '')+'_brute_force_'+str(round(score, 4))+'.output'
    with open(file_output, 'w') as f:
        f.write('# '+str(graphID)+' '+str(len(G))+' '+str(edges[graphID])+' '+str(args.k)+'\n')
        for vertex_ID in np.sort([int(x) for x in G.nodes()]):
            f.write(f'{vertex_ID} {y_hat[str(vertex_ID)]}\n')
    print('Results saved in '+file_output)


# Score our partitions using a graph and its cluster
def score_clustering_graph(G, y_hat):
    total = 0
    for i in range(args.k):
        v_isize = len([x for x in y_hat.items() if x[1]==i])  # Size of the cluster i
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


def brute_force(G, y_hat):
    for it in range(args.iterations):
        for vertex_ID in y_hat.keys():
            best_score = score_clustering_graph(G, y_hat)
            for cluster in range(args.k):
                y_hat_tmp = y_hat.copy()
                y_hat_tmp[vertex_ID] = cluster
                score = score_clustering_graph(G, y_hat_tmp)
                if score<best_score:
                    print('[*] Best score saved {}'.format(score))
                    best_score = score
                    y_hat = y_hat_tmp.copy()
    return y_hat


def brute_force_2(G, y_hat):
    for it in range(args.iterations):
        print("Iteration ",it)
        for vertex_ID in y_hat.keys():
            best_score = score_clustering_graph(G, y_hat)
            for cluster in range(args.k):
                y_hat_tmp = y_hat.copy()
                y_hat_tmp[vertex_ID] = cluster
                score = score_clustering_graph(G, y_hat_tmp)
                if score<best_score:
                    s = score
                    y_h= y_hat_tmp.copy()
        print('[*] Best score saved {}'.format(s))
        best_score = s
        y_hat = y_h.copy()
    return y_hat


def brute_force_3(G, y_hat):
    for it in range(args.iterations):
        print("Iteration ",it)
        s = 1000
        for j in range(200):
            subset = random.sample(list(y_hat.keys()),30)
            best_score = score_clustering_graph(G, y_hat)

            for cluster in range(args.k):
                y_hat_tmp = y_hat.copy()
                for vertex_ID in subset:
                    y_hat_tmp[vertex_ID] = cluster
                score = score_clustering_graph(G, y_hat_tmp)
                if score<best_score:
                    s = score
                    y_h= y_hat_tmp.copy()
        if s < best_score:
            print('[*] Best score saved {}'.format(s))
            best_score = s
            y_hat = y_h.copy()
    return y_hat




# Main function
def main():
    f = open(args.file, 'rb')
    G = nx.read_edgelist(f)
    f.close()
    y_hat ={}
    with open(args.output_file) as f:
        next(f)
        for line in f:
            line_list = line.replace('\n','').split() 
            y_hat[line_list[0]] = int(line_list[1])
    print(len(y_hat),'with initial score', score_clustering_graph(G, y_hat))
    y_hat_bf = brute_force_3(G, y_hat)
    print('Score of the brute force: {}'.format(score_clustering_graph(G, y_hat_bf)))
    save_result(G, y_hat_bf)

        
if __name__ == '__main__':
    main()
