import networkx as nx
import numpy as np
from math import log
from scipy import sparse as sps
from scipy import stats as stats
from utilities import fast_choice, sp_unique, gaussian_pdf_2d, printarr
from mr_calc_and_approx import bipartite_workload_decomposition
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys




def generate_grid_compatability_matrix_with_map(m, d, zeta, structure='grid', prt=True):


    num_of_centers_lamda = int(m**0.5)
    centers_lamda = [((np.random.uniform(0.2*m, 0.8*m), np.random.uniform(0.2*m, 0.8*m)), np.random.uniform(0.2, 1)) for _ in range(num_of_centers_lamda)]
    lamda = gaussian_pdf_2d(m, m, centers_lamda, normalize=True)
    lamda = lamda + (10**-6)*np.ones(lamda.shape)
    lamda = lamda/lamda.sum()

    num_of_centers_mu = int(m**0.5)
    centers_mu = [((np.random.uniform(0.2*m, 0.8*m), np.random.uniform(0.2*m, 0.8*m)), np.random.uniform(0.2, 1)) for _ in range(num_of_centers_mu)]
    mu = gaussian_pdf_2d(m, m, centers_mu, normalize=True)
    mu = mu + (10**-6)*np.ones(mu.shape)
    mu = mu/mu.sum()
    mu = (1 - zeta) * lamda + zeta * mu

    min_val  = min(lamda.min(), mu.min())
    max_val  = max(lamda.max(), mu.max())

    # fig, ((ax1, ax2)) = plt.subplots(1, 2)

    g = nx.empty_graph(0, None)

    # nodes = [(x, y, lamda[x,y], mu[x,y]) for x in range(m) for y in range(m)]
    nodes = [(x, y) for x in range(m) for y in range(m)]
    node_map = np.array([[k, x, y] for k, (x,y) in enumerate(nodes)])
    

    g.add_nodes_from(nodes)

    def dist_mod_k(ix, iy, jx, jy, k,  p=1):

        # print(ix, iy, jx, jy, k)

        dx = min(np.abs(ix - jx), np.abs(ix + k - jx), np.abs(jx + k - ix))
        dy = min(np.abs(iy - jy), np.abs(iy + k - jy), np.abs(jy + k - iy))

        # print(dx, dy)
        # print((dx**p + dy**p)**(1./p))

        return (dx**p + dy**p)**(1./p)

    def dist(ix, iy, jx, jy, k=0,  p=1):

        dx = np.abs(ix - jx)
        dy = np.abs(iy - jy)

        return (dx**p + dy**p)**(1./p)

    dist_func = dist_mod_k if structure == 'tours' else dist

    edges = set()
        # ((ix, iy), (jx, jy)) 
        # for ((ix, iy), (jx, jy)) in product(nodes, nodes)
        # if dist_func(ix, iy, jx, jy, m) <= d)
    print('creating edges')
    for (x_1, y_1) in nodes:
        for x_2 in range(max(x_1 - d, 0), min(x_1 + d + 1, m), 1):
            for y_2 in range(max(y_1 - d, 0), min(y_1 + d + 1, m), 1):
                # if (x_1, y_1) == (0,0) and dist(x_1, y_1, x_2, y_2)<=d:
                #     print((0,0), (x_2, y_2), dist(x_1, y_1, x_2, y_2), dist(x_1, y_1, x_2, y_2)<=d, x_1 < x_2, ((x_1 == x_2) and (y_1 < y_2)))
                if dist(x_1, y_1, x_2, y_2) <= d:
                    if x_1 < x_2 or ((x_1 == x_2) and (y_1 <= y_2)):
                        edges.add(((x_1, y_1),(x_2, y_2)))
    

    g.add_edges_from(edges)
    print('edges added')
    compatability_matrix = nx.adjacency_matrix(g)



    # print('(x, x)', np.array([node[0] for node in nodes]))
    # print('(x, x)', np.array([node[1] for node in nodes]))
    # print('--'*50)
    # for k, node in enumerate(nodes):
    #     if node == (4, 4):
    #         print(node, compatability_matrix[k])

    workload_sets, rho_m, rho_n = bipartite_workload_decomposition(compatability_matrix, lamda.ravel(), mu.ravel(), path=None)

    # for key, workload_set in workload_sets.items():
    #     print(key, workload_set['rho'], len(workload_set['supply_nodes']), len(workload_set['demnand_nodes']))
    # edge_count = len(compatability_matrix.nonzero()[0])
    # i_x = np.zeros(edge_count)
    # i_y = np.zeros(edge_count)
    # j_x = np.zeros(edge_count)
    # j_y = np.zeros(edge_count)

    # for k, (i, j) in enumerate(zip(*compatability_matrix.nonzero())):

    #     i_x[k] = node_map[i, 1]
    #     i_y[k] = node_map[i, 2]
        
    #     j_x[k] = node_map[j, 1]
    #     j_y[k] = node_map[j, 2]

    return compatability_matrix, g, lamda.ravel(), mu.ravel(), workload_sets, rho_m, rho_n, node_map #, (i_x, i_y, j_x, j_y)

def generate_grid_compatability_matrix(m, d=None, structure='tours', prt=True):


    d = floor(log(m))-1 if d is None else d

    print('d: ',d)

    g = nx.empty_graph(0,None)

    rows = range(m)
    cols = range(m)

    # adding all the nodes
    nodes = [(x, y) for x in rows for y in cols]

    g.add_nodes_from(nodes)

    def dist_mod_k(ix, iy, jx, jy, k,  p=1):

        dx = min(np.abs(ix - jx), np.abs(ix + k - jx), np.abs(jx + k - ix))
        dy = min(np.abs(iy - jy), np.abs(iy + k - jy), np.abs(jy + k - iy))

        return (dx**p + dy**p)**(1./p)

    def dist(ix, iy, jx, jy, k,  p=1):

        dx = np.abs(ix - jx)
        dy = np.abs(iy - jy)

        return (dx**p + dy**p)**(1./p)

    dist_func = dist_mod_k if structure == 'tours' else dist

    edges = set(
        ((ix, iy), (jx, jy)) 
        for ((ix, iy), (jx, jy)) in product(nodes, nodes)
        if dist_func(ix, iy, jx, jy, m) <=d )
    
    # for ((ix, iy), (jx, jy)) in product(nodes, nodes):
    #     print((ix, iy), (jx, jy), min(np.abs(ix - jx), np.abs(ix + m - jx), np.abs(jx + m - ix)), min(np.abs(iy - jy), np.abs(iy + m - jy), np.abs(jy + m - iy)) ,dist_func(ix, iy, jx, jy, m))

    g.add_edges_from(edges)

    compatability_matrix = nx.adjacency_matrix(g).todense().A







    return compatability_matrix, g


def generate_chain_compatability_matrix(m, k):

    d = math.ceil(2*log(m))
    base = [1]*k + [0]*(m-d)

    compatability_matrix = np.array([[(0 <= (j-i) < d) or ( 0 <= (j+n-i) < d) for j in range(n)] for i in range(n)])
    
    return compatability_matrix


def generate_erdos_renyi_compatability_matrix(m, n, p_edge=None, seed=None, max_iter=1000):

    if seed is not None:
        np.random.seed(seed)

    col_diag_2_powers = np.diag([2**i for i in range(n)])
    row_diag_2_powers = np.diag([2**i for i in range(m)])
    
    p_edge = 2*log(n)/n if p_edge is None else 0


    for i in range(max_iter) :
        print(i)
        #compatability_matrix = np.random.choice([0, 1], size=(m,n), p=[1. - p_edge, p_edge])
        compatability_matrix = (1*(np.random.uniform(0, 1, size=(m, n)) < p_edge)).astype(int)

        # check for all zero columns or rows
        if (compatability_matrix.sum(axis=0) == 0).sum() > 0:
            continue
        if (compatability_matrix.sum(axis=1) == 0).sum() > 0:
            continue
        # check for identical columns or rows
        count_unique_rows = len(np.unique(np.dot(compatability_matrix, col_diag_2_powers).sum(axis=1)))
        if count_unique_rows != m:
            continue
        count_unique_cols = len(np.unique(np.dot(row_diag_2_powers, compatability_matrix).sum(axis=1)))
        if count_unique_cols != n:
            continue

        return compatability_matrix

    else:
        print('could not generate a valid compatability_matrix after {} attempts'.format(max_iter))
        return None


def generate_erdos_renyi_compatability_matrix_large(m, n, p_edge=None, seed=None, max_iter=1000):

    if seed is not None:
        np.random.seed(seed)

    col_diag_2_powers = np.diag([2**i for i in range(n)])
    row_diag_2_powers = np.diag([2**i for i in range(m)])
    
    p_edge = 2*log(n)/n if p_edge is None else 0
    r_n = range(n)
    
    for k in range(max_iter) :

        rows = []
        cols = []
        data = []

        degs = np.random.binomial(n=n, p=p_edge, size=m)
        
        for i in range(m):
            qual = fast_choice(r_n, degs[i])
            for j in qual:
                rows.append(i)
                cols.append(j)
                data.append(1.0)


        compatability_matrix = sps.coo_matrix((data, (rows, cols)), shape=(m, n))

        #compatability_matrix = np.random.choice([0, 1], size=(m,n), p=[1. - p_edge, p_edge])
        # compatability_matrix = (1*(np.random.uniform(0, 1, size=(m, n)) < p_edge)).astype(int)
        
        # check for all zero columns or rows
        if (compatability_matrix.sum(axis=0) == 0).sum() > 0:
            # print('zero col')
            continue
        if (compatability_matrix.sum(axis=1) == 0).sum() > 0:
            # print('zero row')
            continue
        # check for identical columns or rows
        count_unique_rows = sp_unique(compatability_matrix, axis=0).shape[0]
        if count_unique_rows != m:
            # print('dup col')
            continue
        count_unique_cols = sp_unique(compatability_matrix, axis=1).shape[0]
        if count_unique_cols != n:
            # print('dup row')
            continue

        return sps.csr_matrix(compatability_matrix)

    else:
        print('could not generate a valid compatability_matrix after {} attempts'.format(max_iter))
        return None


def generate_nx_flow_graph(compatability_matrix, alpha=None, beta=None):

    m, n = compatability_matrix.shape

    customer_nodes = ['c' + str(i) for i in range(m)]
    server_nodes = ['s' + str(j) for j in range(n)]

    custommer_server_edges = [('c' + str(i), 's' + str(j)) for i,j in zip(*compatability_matrix.nonzero())]
    origin_customer_edges = [('o', 'c' + str(i)) for i in range(m)]
    server_destination_edges = [('s' + str(j), 'd') for j in range(n)]

    od_flow_graph = nx.DiGraph()
    
    od_flow_graph.add_nodes_from(['o'] + customer_nodes + server_nodes + ['d'])
    od_flow_graph.add_edges_from(origin_customer_edges + custommer_server_edges + server_destination_edges)

    if alpha is not None:

        origin_customer_edge_capacities = dict((('o','c' + str(i)), alpha[i]) for i in range(m))
        server_destination_edge_capacities = dict((('s' + str(j), 'd'), beta[j]) for j in range(n))
        custommer_server_edge_capacities = dict(((ci, sj), 2.) for ci, sj in custommer_server_edges)
        
        nx.set_edge_attributes(
            od_flow_graph,
            {
                **origin_customer_edge_capacities,
                **custommer_server_edge_capacities,
                **server_destination_edge_capacities
            },
            'capacity')

    return od_flow_graph


def verify_crp_condition(compatability_matrix, alpha, beta):

    m, n = compatability_matrix.shape

    customer_nodes = ['c' + str(i) for i in range(m)]
    server_nodes = ['s' + str(j) for j in range(n)]

    custommer_server_edges = [('c' + str(i), 's' + str(j)) for i,j in zip(*compatability_matrix.nonzero())]
    origin_customer_edges = [('o', 'c' + str(i)) for i in range(m)]
    server_destination_edges = [('s' + str(j), 'd') for j in range(n)]

    od_flow_graph = nx.DiGraph()
    
    od_flow_graph.add_nodes_from(['o'] + customer_nodes + server_nodes + ['d'])
    od_flow_graph.add_edges_from(origin_customer_edges + custommer_server_edges + server_destination_edges)

    origin_customer_edge_capacities = dict((('o','c' + str(i)), alpha[i]) for i in range(m))
    server_destination_edge_capacities = dict((('s' + str(j), 'd'), beta[j]) for j in range(n))
    custommer_server_edge_capacities = dict(((ci, sj), 2.) for ci, sj in custommer_server_edges)
    
    nx.set_edge_attributes(
        od_flow_graph,
        {
            **origin_customer_edge_capacities,
            **custommer_server_edge_capacities,
            **server_destination_edge_capacities
        },
        'capacity')

    max_flow_val, max_flow = nx.maximum_flow(od_flow_graph, 'o', 'd')
    
    if max_flow_val < 1.:
        return False, 'flow<1'
        
    active_customer_server_edges = [
        (node, neighbor) 
            for node, flow_from_node in max_flow.items() 
                for neighbor, node_neighbor_flow in flow_from_node.items()
                    if node_neighbor_flow > 0 and node != 'o' and neighbor != 'd'
    ]

    customer_nodes = ['c' + str(i) for i in range(m)]
    server_nodes = ['s' + str(j) for j in range(n)]

    active_edge_graph = nx.Graph()
    active_edge_graph.add_nodes_from(customer_nodes + server_nodes)
    active_edge_graph.add_edges_from(active_customer_server_edges)

    if nx.is_connected(active_edge_graph):

        return True, 'crp_holds'

    else:
        return False, 'not_connected'



    
if __name__ == '__main__':  

    np.set_printoptions(threshold=sys.maxsize)
    generate_grid_compatability_matrix_with_map(100,2,0.2)
#     # Test
#     # Create a large sparse matrix with elements in [0, 10]
#     A = 10*sps.random(10000, 3, 0.5, format='csr')
#     A = np.ceil(A).astype(int)


#     # unique rows
#     A_uniq = sp_unique(A, axis=0).toarray()
#     A_uniq = lexsort_row(A_uniq)
#     A_uniq_numpy = np.unique(A.toarray(), axis=0)
#     assert (A_uniq == A_uniq_numpy).all()

#     # unique columns
#     A_uniq = sp_unique(A, axis=1).toarray()
#     A_uniq = lexsort_row(A_uniq.T).T
#     A_uniq_numpy = np.unique(A.toarray(), axis=1)
#     assert (A_uniq == A_uniq_numpy).all()
#     print((A_uniq == A_uniq_numpy).sum())

