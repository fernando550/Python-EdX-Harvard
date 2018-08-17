# networkX degrees and component count

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import bernoulli

# random graph generation
def er_graph(n,p):
    g2 = nx.Graph()
    g2.add_nodes_from(range(n))
    for node1 in g2.nodes():
        for node2 in g2.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                g2.add_edge(node1,node2)
    return g2

# degree distribution
def plot_degree_dist(g):
    degree_dict = [d for n,d in g.degree()]
    plt.hist(degree_dict, histtype='step')
    plt.xlabel('degree k')
    plt.ylabel('p(k)')
    plt.title('degree distribution')
    
#g3 = er_graph(500, 0.08)
#plot_degree_dist(g3)

# descriptive statistics of social networks
a1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
a2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

g1 = nx.to_networkx_graph(a1)
g2 = nx.to_networkx_graph(a2)

def basic_net_stats(g):
    print("num of nodes: ", g.number_of_nodes())
    print("num of edges: ", g.number_of_edges())
    degree_sequence = [d for n, d in g.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))
    print("\n")

#basic_net_stats(g1)
#basic_net_stats(g2)
#
#plot_degree_dist(g1)
#plot_degree_dist(g2)


# finding largest connection component
g1_lcc = max(nx.connected_component_subgraphs(g1), key=len)
g2_lcc = max(nx.connected_component_subgraphs(g2), key=len)

plt.figure()
nx.draw(g1_lcc, node_color="red", edge_color="grey", node_size=20)
plt.figure()
nx.draw(g2_lcc, node_color="green", edge_color="grey", node_size=20)
