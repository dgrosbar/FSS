import networkx as nx
import numpy as np
from math import log
from scipy import sparse as sps
from utilities import fast_choice, sp_unique


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

        dx = min((np.abs(ix - jx)), (np.abs(ix + k - jx)))
        dy = min((np.abs(iy - jy)), (np.abs(iy + k - jy)))

        return (dx**p + dy**p)**(1./p)

    def dist(ix, iy, jx, jy, k,  p=1):

        dx = np.abs(ix - jx)
        dy = np.abs(iy - jy)

        return (dx**p + dy**p)**(1./p)

    dist_func = dist_mod_k if structure == 'tours' else dist

    edges = set(
        ((ix, iy), (jx, jy)) 
        for ((ix, iy), (jx, jy)) in product(nodes,nodes)
        if dist_func(ix, iy, jx, jy, m) <=d )

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



# if __name__ == '__main__':    
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

