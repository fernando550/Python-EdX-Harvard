# networkX graph generator and distribution

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import bernoulli

# graph instance
g = nx.Graph()

# adding nodes
g.add_node(1)
g.add_nodes_from([2,3])
g.add_nodes_from(["u","v"])

#get nodes
#print('node list: ', g.nodes())

# add an edge or vertix
g.add_edge(1,2)
g.add_edge("u","v")
g.add_edges_from([(1,3), (1,4),(1,5),(1,6)])
g.add_edge("u","w")

# get edges
#print('edge list: ', g.edges())

# remove nodes
g.remove_node(2)
g.remove_nodes_from([4,5])
#print('nodes after removal: ', g.nodes())

# remove edges
g.remove_edge(1,3)
g.remove_edges_from([(1,2),('u','v')])
#print('edges after removal: ', g.edges())

# number of nodes and edges
#print('num of nodes: ', g.number_of_nodes())
#print('num of edges: ', g.number_of_edges())


# karate club graph
g1 = nx.karate_club_graph()
#nx.draw(g1, with_labels=True, node_color='lightblue', edge_color='gray')

#each graph stores degrees for each node, or how many connections it has
g1.degree() # returns dictionary
g1.degree(33) # returns dictionary
g1.degree([1,2,33,10]) # returns dictionary
g1.degree()[33] # returns value

# random graph generation
def er_graph(n,p):
    g2 = nx.Graph()
    g2.add_nodes_from(range(n))
    for node1 in g2.nodes():
        for node2 in g2.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                g2.add_edge(node1,node2)
    return g2

#nx.draw(er_graph(50,0.08), node_size=40, node_color='gray')

# degree distribution
def plot_degree_dist(g):
    degree_dict = [d for n,d in g.degree()]
    plt.hist(degree_dict, histtype='step')
    plt.xlabel('degree k')
    plt.ylabel('p(k)')
    plt.title('degree distribution')
    
g3 = er_graph(500, 0.08)
#plot_degree_dist(g3)


# descriptive statistics of social networks
np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

