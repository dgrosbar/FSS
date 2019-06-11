import numpy as np
from numpy import ma
import networkx as nx
from itertools import permutations, combinations, chain, product
from functools import reduce
import operator
import sys
from time import time
import datetime as dt
import os
import functools
from scipy import sparse as sps
from math import floor, ceil, log, exp
import multiprocessing as mp

from simulators import *
from generators import *
from utilities import *
from mr_calc_and_approx import *
from graphs_and_tabels import *


BASE_EXAMPLES = {

    2: (
        np.array([
            [1,1],
            [0,1]]),

        np.array([.14, .15])/sum([.14, .15]),
        np.array([.04, .25])/sum([.04, .25])
    ),


    4: (
        np.array([
            [1,1,1,0],
            [0,1,1,1],
            [1,0,1,1],
            [1,1,0,1]]),
        np.array([.14, .15, .16, .17])/sum([.14, .15, .16, .17]),
        np.array([.04, .25, .06, .27])/sum([.04, .25, .06, .27])
    ),


    6: (
        np.array([
            [1,1,1,0,0,0],
            [0,1,1,1,0,0],
            [0,0,1,1,1,0],
            [0,0,0,1,1,1],
            [1,0,0,0,1,1],
            [1,1,0,0,0,1]]),
        np.array([.14, .15, .16, .17, .18, .20])/sum([.14, .15, .16, .17, .18, .20]),
        np.array([.04, .25, .06, .27, .08, .30])/sum([.04, .25, .06, .27, .08, .30])
    ),
    
    8:(
        np.array([
            [1,1,1,0,0,0,1,1],
            [0,1,1,1,0,0,0,1],
            [0,0,1,1,1,0,0,0],
            [0,0,0,1,1,1,0,0],
            [1,0,0,0,1,1,1,0],
            [1,1,0,0,0,1,1,1],
            [1,1,1,0,0,0,1,1],
            [0,1,1,1,0,0,0,1]]),
        np.array([.14, .15, .16, .17, .18, .20, .14, .15])/sum([.14, .15, .16, .17, .18, .20, .14, .15]),
        np.array([.04, .25, .06, .27, .08, .30, .04, .25])/sum([.04, .25, .06, .27, .08, .30, .04, .25])
    ),
    10:(
        np.array([
            [1,1,1,0,0,0,1,1,1,0],
            [0,1,1,1,0,0,0,1,1,1],
            [0,0,1,1,1,0,0,0,1,1],
            [0,0,0,1,1,1,0,0,0,1],
            [1,0,0,0,1,1,1,0,0,0],
            [1,1,0,0,0,1,1,1,0,0],
            [1,1,1,0,0,0,1,1,1,0],
            [0,1,1,1,0,0,0,1,1,1],
            [0,0,1,1,1,0,0,0,1,1],
            [0,0,0,1,1,1,0,0,0,1]]),
        np.array([.14, .15, .16, .17, .18, .20, .14, .15, .16, .17])/sum([.14, .15, .16, .17, .18, .20, .14, .15, .16, .17]),
        np.array([.04, .25, .06, .27, .08, .30, .04, .25, .06, .27])/sum([.04, .25, .06, .27, .08, .30, .04, .25, .06, .27])
    )
}


def grids_exp(filename='grids_exp'):

    jt_perm_dict = {9: list(jpermute(range(9)))}
    print_progress = True

    for structure in ['torus']:

        for sqrt_m in [30]:

            exact = (sqrt_m == 3)
            m =sqrt_m**2
            jt_perms = jt_perm_dict[9] if sqrt_m ==3 else None

            k = 0
            while k < 20:
                alpha = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized customer frecompatability_matrixuency       
                beta = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized server frecompatability_matrixuency
                alpha = alpha/alpha.sum()
                beta = beta/beta.sum()
                valid = False
                d = 0
                
                for d in range(1, 1 + sqrt_m//3, 1):

                    compatability_matrix, g = generate_grid_compatability_matrix(sqrt_m, d)
                    valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)
                    
                    if valid:
                        k += 1
                        arc_dist=d
                        print(k-1, str(sqrt_m) + 'x' + str(sqrt_m), 'd=', arc_dist)
                        print('-'*75)
                        # if sqrt_m < 9:
                        #     node_nums = dict(zip(list(g.nodes), range(m)))
                        #     node_edge_nums = dict((node, [(k,j) for k,j in g.edges if k == node or j==node]) for node in g.nodes)
                        #     for node, edges in node_edge_nums.items():
                        #         node_edge_nums[node] = [node_nums[k] if k!=node else node_nums[j] for k,j in edges]

                        #     print('(_, _)','i', np.array([k for k in range(m)]))
                        #     for node in g.nodes:
                        #         print(node, node_nums[node], compatability_matrix[node_nums[node],:], sorted(node_edge_nums[node]))

                        res_df_k = compare_approximations_generic(compatability_matrix, alpha, beta, jt_perms, exact, print_progress, exp_num=k, size=str(sqrt_m) + 'x' + str(sqrt_m), arc_dist=arc_dist, structure=structure)
                
                        if os.path.exists(filename + '.csv'):
                            res_df = pd.read_csv(filename + '.csv')
                            res_df = pd.concat([res_df, res_df_k], axis=0)
                            res_df.to_csv(filename + '.csv', index=False)
                        else:
                            res_df = res_df_k
                            res_df.to_csv(filename + '.csv', index=False)


def grids_exp_for_parallel(filename='grids_exp_parallel_extra_30x30_5_8'):

    jt_perm_dict = {9: list(jpermute(range(9)))}
    print_progress = True

    for structure in ['torus']:

        for sqrt_m in [30]:

            exact = False
            m =sqrt_m**2
            jt_perms = jt_perm_dict[9] if sqrt_m ==3 else None

            k = 5
            while k < 8:
                exps = []
                while len(exps) < 2:

                    alpha = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized customer frecompatability_matrixuency       
                    beta = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized server frecompatability_matrixuency
                    alpha = alpha/alpha.sum()
                    beta = beta/beta.sum()
                    valid = False
                    d = 3
                    compatability_matrix, g = generate_grid_compatability_matrix(sqrt_m, d)
                    valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)
                    
                    if valid:
                        k += 1
                        arc_dist=d
                        print(k-1, str(sqrt_m) + 'x' + str(sqrt_m), 'd=', arc_dist)
                        print('-'*75)
                        aux_data = {'exp_num': k, 'size': str(sqrt_m) + 'x' + str(sqrt_m), 'arc_dist': arc_dist, 'structure': structure}
                        exps.append([compatability_matrix, alpha, beta, jt_perms, exact, print_progress, aux_data])
                            
                pool = mp.Pool(processes=2)
                exps_res = pool.starmap(compare_approximations_generic_parallel, exps)

                exps_res_df = pd.concat(exps_res, axis=0)

                exps_res_df = go_back_and_solve_qp(exps_res_df) 
                
                if os.path.exists(filename + '.csv'):
                    res_df = pd.read_csv(filename + '.csv')
                    res_df = pd.concat([res_df, exps_res_df], axis=0)
                    res_df.to_csv(filename + '.csv', index=False)
                else:
                    res_df = exps_res_df
                    res_df.to_csv(filename + '.csv', index=False)


def erdos_renyi_exp_for_parallel(filename='erdos_renyi_exp3'):


    for structure in ['erdos_renyi']:
        
        for m in [500]:

            exact = False
            jt_perms = None
            print_progress = False

            k = 0
            p = 7
            while k < 30:
                exps = []
                while len(exps) < p:

                    alpha = np.random.exponential(scale=1, size=m) # obtain values for non normelized customer frecompatability_matrixuency       
                    beta = np.random.exponential(scale=1, size=m) # obtain values for non normelized server frecompatability_matrixuency
                    alpha = alpha/alpha.sum()
                    beta = beta/beta.sum()
                    valid = False
                    compatability_matrix = generate_erdos_renyi_compatability_matrix_large(m, m)
                    valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)
                    
                    if valid:
                        k += 1
                        p_edge = 2*m/log(m)
                        print(k-1, str(m) + 'x' + str(m))
                        print('-'*75)
                        aux_data = {'exp_num': k, 'size': str(m) + 'x' + str(m), 'structure': structure, 'p_edge': p_edge}
                        exps.append([compatability_matrix, alpha, beta, jt_perms, exact, print_progress, aux_data])
                

                pool = mp.Pool(processes=p)
                print(len(exps))
                # exps_res = compare_approximations_generic_parallel(*exps[0])
                exps_res = pool.starmap(compare_approximations_generic_parallel, exps)
                for res_df_k in exps_res:
                    if os.path.exists(filename + '.csv'):
                        res_df = pd.read_csv(filename + '.csv')
                        res_df = pd.concat([res_df, res_df_k], axis=0)
                        res_df.to_csv(filename + '.csv', index=False)
                    else:
                        res_df = res_df_k
                        res_df.to_csv(filename + '.csv', index=False)


def growing_chains_exp(filename='growing_chains'):


    #jt = dict((v, [(x[0] , x[1]) for x in jpermute(range(v))]) for v in [5,7,9])

    # for k, n in [(3,5), (3,7),(5,7), (3, 9), (5, 9), (7,9), (3, 30), (5, 30), (7, 30), (9, 30), (11, 30), (3, 100), (5, 100), (7, 100), (9, 100), (11, 100)]:
    for k,n in list(product([7, 9,11,13, 15, 17, 19],[200, 300])) + [(13, 100), (17, 100)]:


        base = [1]*k + [0]*(n-k)
        compatability_matrix = 1*np.array([[(0<=(j-i)<k) or (0<=(j+n-i)<k) for j in range(n)] for i in range(n)])
        alpha = np.array([1./n]*n)
        beta = np.array([1./n]*n)
        nnz = compatability_matrix.nonzero()

        print('-'*30 + str(n) +','+str(k) + '-'*30)

        if n <=10:
            jt_perms = jt[n]
            matching_rates = adan_weiss_fcfs_alis_matching_rates_vec_jt_numba(compatability_matrix, alpha, beta, jt_perms)
            matching_rates_stdev = np.zeros((n, n))
        else:
            matching_rates, matching_rates_stdev = simulate_matching_sequance(compatability_matrix, alpha, beta, prt=False, sims=1, sim_len=n*k*5000)

        cur_res_df = pd.DataFrame.from_dict({
            'i': nnz[0],
            'j': nnz[1],
            'alpha': alpha[nnz[0]],
            'beta': beta[nnz[1]],
            'matching_rate': matching_rates[nnz],
        })
        cur_res_df.loc[:, 'n'] = n
        cur_res_df.loc[:, 'k'] = k
        cur_res_df.loc[:, 'arc_type'] = np.where(cur_res_df['i']<=cur_res_df['j'], cur_res_df['j'] - cur_res_df['i'], cur_res_df['j'] + n - cur_res_df['i'])

        if os.path.exists(filename + '.csv'):
            res_df = pd.read_csv(filename + '.csv')
            res_df = pd.concat([res_df, cur_res_df], axis=0)
            res_df.to_csv(filename + '.csv', index=False)
        else:
            res_df = cur_res_df
            res_df.to_csv(filename + '.csv', index=False)


def compare_entropy_and_ohn_law_approximation(filename='FZ_Kaplan_exp_100'):


    def generate_valid_compatability_matrix(m, n, p_edge=1./2, max_iter=10000, min_edge_density=0.4, max_edge_density=0.7):


        # Will produce a random integer between 7 and 11

        col_diag_2_powers = np.diag([2**i for i in range(n)])
        row_diag_2_powers = np.diag([2**i for i in range(m)])

        for i in range(max_iter) :
            
            #compatability_matrix = np.random.choice([0, 1], size=(m,n), p=[1. - p_edge, p_edge])
            compatability_matrix = (1*(np.random.uniform(0, 1, size=(m,n)) < p_edge)).astype(int)
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
            if min_edge_density <= compatability_matrix.sum()/(m*n) < max_edge_density:
                return compatability_matrix

        else:
            print('could not generate a valid compatability_matrix after {} attempts'.format(max_iter))
            return None


    def generate_valid_experiment(compatability_matrix, dist='exponential', max_iter=100):

            m, n = compatability_matrix.shape


            customer_nodes = ['c' + str(i) for i in range(m)]
            server_nodes = ['s' + str(j) for j in range(n)]

            custommer_server_edges = [('c' + str(i), 's' + str(j)) for i,j in zip(*compatability_matrix.nonzero())]
            custommer_server_edge_capacities = dict(((ci, sj), 2.) for ci, sj in custommer_server_edges)

            origin_customer_edges = [('o', 'c' + str(i)) for i in range(m)]
            server_destination_edges = [('s' + str(j), 'd') for j in range(n)]

            od_flow_graph = nx.DiGraph()
            
            od_flow_graph.add_nodes_from(['o'] + customer_nodes + server_nodes + ['d'])
            od_flow_graph.add_edges_from(origin_customer_edges + custommer_server_edges + server_destination_edges)

            connect_fails = 0
            flow_fails = 0

            for i in range(max_iter):

                if i > 0 and i % 10 == 0: 
                    print(i, 'attempts')
                    print(flow_fails, ' flow_fails')
                    print(connect_fails, ' connect_fails')

                if dist == 'exponantial':
                    alpha = np.random.exponential(scale=1, size=m) # obtain values for non normelized customer frecompatability_matrixuency       
                    beta = np.random.exponential(scale=1, size=n) # obtain values for non normelized server frecompatability_matrixuency
                else: # dist == uniform
                    alpha = np.random.uniform(size=m) # obtain values for non normelized customer frecompatability_matrixuency       
                    beta = np.random.uniform(size=n) # obtain values for non normelized server frecompatability_matrixuency
                
                alpha = alpha/alpha.sum()
                beta = beta/beta.sum()

                origin_customer_edge_capacities = dict((('o','c' + str(i)), alpha[i]) for i in range(m))
                server_destination_edge_capacities = dict((('s' + str(j), 'd'), beta[j]) for j in range(n))

                nx.set_edge_attributes(od_flow_graph,{
                    **origin_customer_edge_capacities,
                    **custommer_server_edge_capacities,
                    **server_destination_edge_capacities}, 'capacity')

                max_flow_val, max_flow = nx.maximum_flow(od_flow_graph, 'o', 'd')
                
                if max_flow_val < 1.:
                    flow_fails +=1
                    continue

                active_customer_server_edges = [
                    (node, neighbor) 
                        for node, flow_from_node in max_flow.items() 
                            for neighbor, node_neighbor_flow in flow_from_node.items()
                                if node_neighbor_flow > 0 and node != 'o' and neighbor != 'd'
                ]

                active_edge_graph = nx.Graph()
                active_edge_graph.add_nodes_from(customer_nodes + server_nodes)
                active_edge_graph.add_edges_from(active_customer_server_edges)

                if nx.is_connected(active_edge_graph):

                    return alpha, beta

                else:
                    connect_fails += 1

            else:
                print('could not generate a valid experiment after {} attempts'.format(max_iter))
                return None, None

    # "3 graph densities (low-(0, 0.4), medium-(0.4, 0.7), high-(0.7, 1)) × 
    #  15 random compatibility graphs for each density ×
    #  40 (20 uniform + 20 exponential) random random customer arrival/server availability rates for each compatibility graph

    densities = {'low': (0, 0.4), 'medium':(0.4, 0.7), 'high':(0.7, 1)}
    graphs_per_density = 15
    exp_per_graph = 20

    for density, (min_edge_density, max_edge_density) in densities.items():

        print('generating exp with density {}'.format(density))
        k = 0
        k_fail = 0

        while k < graphs_per_density:

            print('generating graph no. {} for density {}'.format(k, density))
            m = 10
            n = np.random.randint(7, 11) # genrates a random integer between 7-10
            compatability_matrix = generate_valid_compatability_matrix(m, n , 1./2, 100, min_edge_density, max_edge_density)
            
            if compatability_matrix is not None:

                print(np.array2string(compatability_matrix, max_line_width=np.inf))
                l = 0
                graph_k_res_df = pd.DataFrame()

                while l < exp_per_graph:

                    exact = l==0 or n < 10

                    print('generating exponential distribution experiment no. {} for graph no. {} for density {}'.format(l, k, density))
                    alpha, beta = generate_valid_experiment(compatability_matrix, 'exponential')
                    if alpha is None:
                        break
                    print('alpha:', np.array2string(alpha, max_line_width=np.inf))
                    print('beta:', np.array2string(beta, max_line_width=np.inf))

                    graph_k_exp_l_expo_res_df = compare_approximations(compatability_matrix, alpha, beta, density, k, 'exponential', l, exact)
                    
                    if graph_k_exp_l_expo_res_df is None:
                        break

                    print('generating unifrom distribution experiment no. {} for graph no. {} for density {}'.format(l, k, density))
                    alpha, beta = generate_valid_experiment(compatability_matrix, 'unifrom')
                    if alpha is None:
                        break
                    print('alpha:', np.array2string(alpha, max_line_width=np.inf))
                    print('beta:', np.array2string(beta, max_line_width=np.inf))                    

                    graph_k_exp_l_unif_res_df = compare_approximations(compatability_matrix, alpha, beta, density, k, 'uniform', l, exact)
                    if graph_k_exp_l_unif_res_df is None:
                        break

                    graph_k_res_df = pd.concat([graph_k_res_df, graph_k_exp_l_expo_res_df, graph_k_exp_l_unif_res_df], axis=0)

                    l += 1

                else:

                    k +=1
                    if os.path.exists(filename + '.csv'):
                        res_df = pd.read_csv(filename + '.csv')
                        res_df = pd.concat([res_df, graph_k_res_df], axis=0)
                        res_df.to_csv(filename + '.csv', index=False)
                    else:
                        res_df = graph_k_res_df
                        res_df.to_csv(filename + '.csv', index=False)


def compare_approximations(compatability_matrix, alpha, beta, density, k, dist, l, jt_perms=None, exact=False, print_progress=False):


    m, n = compatability_matrix.shape
    s = time()
    print('getting entropy approximation')
    converge, ent_matching_rates, gap_pct = entropy_approximation(compatability_matrix, alpha, beta)
    print(time() - s)
    s = time()
    print('ohm law approximation')
    try:
        ohm_matching_rates = ohm_law_approximation(compatability_matrix, alpha, beta)
    except Exception as exc:
        print(exc)
        return None
    print(time() - s)
    
    if exact:
        print('getting exact matching_rates')
        s = time()
        exact_matching_rates = adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta, jt_perms, print_progress)
        sim_matching_rates = np.zeros((m,n))
        sim_matching_rates_stdev = np.zeros((m,n))
        sim_matching_rates_upper_ci = np.zeros((m,n))
        sim_matching_rates_lower_ci = np.zeros((m,n))
        print(time() - s)
    else:
        exact_matching_rates = np.zeros((m,n))
        print('getting sim matching_rates')
        s = time()
        sim_matching_rates, sim_matching_rates_stdev = simulate_matching_sequance(compatability_matrix, alpha, beta)
        print(time() - s)
        sim_matching_rates_upper_ci = sim_matching_rates + 1.96 * sim_matching_rates_stdev
        sim_matching_rates_lower_ci = sim_matching_rates - 1.96 * sim_matching_rates_stdev

    cols = [
        'timestamp', 'density_level', 'graph_no', 'm', 'n', 'max_edges', 'edge_count', 'edge_density', 'exp_no', 'alpha_dist', 'beta_dist', 'utilization', 'exact',
        'i', 'j', 'alpha', 'beta', 'exact_matching_rate', 'sim_matching_rate', 'entropy_approx', 'ohm_law_approx',
        'sim_matching_rate_CI_95_U', 'sim_matching_rate_CI_95_L', 'no_of_sims', 'sim_matching_rate_stdev', '95_CI_len', 
        'ohm_error_sim', 'ohm_error_abs_sim', 'ohm_error_pct_sim', 'ohm_error_abs_pct_sim',
        'ent_error_sim', 'ent_error_abs_sim', 'ent_error_pct_sim', 'ent_error_abs_pct_sim', 
        'ohm_error', 'ohm_error_abs','ohm_error_pct','ohm_error_abs_pct',
        'ent_error', 'ent_error_abs', 'ent_error_pct', 'ent_error_abs_pct',
        'sim_error', 'sim_error_abs',   'sim_error_pct',   'sim_error_abs_pct', 'in_CI'
    ]

    nnz = compatability_matrix.nonzero()

    res_df = pd.DataFrame.from_dict({
        'i': nnz[0],
        'j': nnz[1],
        'alpha': alpha[nnz[0]],
        'beta': beta[nnz[1]],
        'exact_matching_rate': exact_matching_rates[nnz],
        'sim_matching_rate': sim_matching_rates[nnz],
        'sim_matching_rate_stdev': sim_matching_rates_stdev[nnz],
        '95_CI_len': 1.96 * sim_matching_rates_stdev[nnz],
        'sim_matching_rate_CI_95_U': sim_matching_rates_upper_ci[nnz],
        'sim_matching_rate_CI_95_L': sim_matching_rates_lower_ci[nnz],
        'entropy_approx': ent_matching_rates[nnz],
        'ohm_law_approx': ohm_matching_rates[nnz],
    })

    res_df.loc[:, 'timestamp'] = dt.datetime.now() #
    res_df.loc[:, 'exact'] = exact
    res_df.loc[:, 'density_level'] = density
    res_df.loc[:, 'graph_no'] = k
    res_df.loc[:, 'max_edges'] = m * n
    res_df.loc[:, 'edge_count'] = len(nnz[0])
    res_df.loc[:, 'edge_density'] = len(nnz[0])/ (m*n)
    res_df.loc[:, 'exp_no'] = l
    res_df.loc[:, 'm'] = m
    res_df.loc[:, 'n'] = n
    res_df.loc[:, 'alph_dist'] = dist
    res_df.loc[:, 'beta_dist'] = dist
    res_df.loc[:, 'utilization'] = alpha.sum()/beta.sum()

    res_df.loc[:, 'no_of_sims'] = 30
    res_df.loc[:, 'ohm_error_sim'] = res_df['ohm_law_approx'] - res_df['sim_matching_rate']
    res_df.loc[:, 'ent_error_sim'] = res_df['entropy_approx'] - res_df['sim_matching_rate']
    res_df.loc[:, 'ohm_error_abs_sim'] = np.abs(res_df['ohm_error_sim'])
    res_df.loc[:, 'ent_error_abs_sim'] = np.abs(res_df['ent_error_sim'])
    res_df.loc[:, 'ohm_error_pct_sim'] = res_df['ohm_error_sim']/res_df['sim_matching_rate']
    res_df.loc[:, 'ent_error_pct_sim'] = res_df['ent_error_sim']/res_df['sim_matching_rate']
    res_df.loc[:, 'ohm_error_abs_pct_sim'] = np.abs(res_df['ohm_error_pct_sim'])
    res_df.loc[:, 'ent_error_abs_pct_sim'] = np.abs(res_df['ent_error_pct_sim'])

    res_df.loc[:, 'ohm_error'] = res_df['ohm_law_approx'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ent_error'] = res_df['entropy_approx'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'sim_error'] = res_df['sim_matching_rate'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'in_CI'] = (
    (res_df['exact_matching_rate'] <= res_df['sim_matching_rate_CI_95_U']) &
    (res_df['exact_matching_rate'] >= res_df['sim_matching_rate_CI_95_L'])
    ) if exact else 0
    res_df.loc[:, 'ohm_error_abs'] = np.abs(res_df['ohm_error']) if exact else 0
    res_df.loc[:, 'ent_error_abs'] = np.abs(res_df['ent_error']) if exact else 0
    res_df.loc[:, 'sim_error_abs'] = np.abs(res_df['sim_error']) if exact else 0
    res_df.loc[:, 'ohm_error_pct'] = res_df['ohm_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ent_error_pct'] = res_df['ent_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'sim_error_pct'] = res_df['sim_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ohm_error_abs_pct'] = np.abs(res_df['ohm_error_pct']) if exact else 0
    res_df.loc[:, 'ent_error_abs_pct'] = np.abs(res_df['ent_error_pct']) if exact else 0
    res_df.loc[:, 'sim_error_abs_pct'] = np.abs(res_df['sim_error_pct']) if exact else 0

    if print_progress:
        print('alpha: ', alpha)
        print('beta: ', beta)
        print('density: ', density)
        print('compatability_matrix {}x{}:'.format(m, n))

        print(np.array2string(compatability_matrix, max_line_width=np.inf))

        print('--------------------------------ent------------------------------------')

        print(np.array2string(ent_matching_rates, max_line_width=np.inf))

        print('---------------------------------ohm------------------------------------')

        print(np.array2string(ohm_matching_rates, max_line_width=np.inf))

        print('---------------------------------exact-------------------------------------')

        print(np.array2string(exact_matching_rates, max_line_width=np.inf))

        print('--------------------------------------------------------------------------')

        print('ohm_max: ', np.amax(np.abs(exact_matching_rates - ohm_matching_rates)))
        print('ent_max: ', np.amax(np.abs(exact_matching_rates - ent_matching_rates)))

        print('ohm_sum: ', np.abs(exact_matching_rates - ohm_matching_rates).sum())
        print('ent_sum: ', np.abs(exact_matching_rates - ent_matching_rates).sum())

        print('x'*40)
        print('x'*40)

    res_df = res_df[cols]

    return res_df


def compare_approximations_generic(compatability_matrix, alpha, beta, jt_perms=None, exact=False, print_progress=False, seed=None, **kwargs):


    m, n = compatability_matrix.shape
    nnz = compatability_matrix.nonzero()
    edge_count = len(nnz[0])
    s = time()
    print('getting entropy approximation')
    converge, ent_matching_rates, gap_pct = entropy_approximation(compatability_matrix, alpha, beta)
    print(time() - s)
    s = time()
    print('ohm law approximation')
    try:
        ohm_matching_rates = ohm_law_approximation(compatability_matrix, alpha, beta)
    except Exception as exc:
        print(exc)
        return None
    print(time() - s)
    
    if exact:
        print('getting exact matching_rates')
        s = time()
        exact_matching_rates = adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta, jt_perms, print_progress)
        sim_matching_rates = np.zeros((m,n))
        sim_matching_rates_stdev = np.zeros((m,n))
        sim_matching_rates_upper_ci = np.zeros((m,n))
        sim_matching_rates_lower_ci = np.zeros((m,n))
        print(time() - s)
    else:
        exact_matching_rates = np.zeros((m,n))
        print('getting sim matching_rates')
        s = time()
        sim_matching_rates, sim_matching_rates_stdev = simulate_matching_sequance(compatability_matrix, alpha, beta, sim_len=edge_count*100)
        print(time() - s)
        sim_matching_rates_upper_ci = sim_matching_rates + 1.96 * sim_matching_rates_stdev
        sim_matching_rates_lower_ci = sim_matching_rates - 1.96 * sim_matching_rates_stdev

    cols = [
        'timestamp', 'm', 'n', 'max_edges', 'edge_count', 'edge_density', 'utilization', 'exact',
        'i', 'j', 'alpha', 'beta', 'exact_matching_rate', 'sim_matching_rate', 'entropy_approx', 'ohm_law_approx',
        'sim_matching_rate_CI_95_U', 'sim_matching_rate_CI_95_L', 'no_of_sims', 'sim_matching_rate_stdev', '95_CI_len', 
        'ohm_error_sim', 'ohm_error_abs_sim', 'ohm_error_pct_sim', 'ohm_error_abs_pct_sim',
        'ent_error_sim', 'ent_error_abs_sim', 'ent_error_pct_sim', 'ent_error_abs_pct_sim', 
        'ohm_error', 'ohm_error_abs','ohm_error_pct','ohm_error_abs_pct',
        'ent_error', 'ent_error_abs', 'ent_error_pct', 'ent_error_abs_pct',
        'sim_error', 'sim_error_abs',   'sim_error_pct',   'sim_error_abs_pct', 'in_CI'
    ] + [key for key in kwargs]

    

    res_df = pd.DataFrame.from_dict({
        'i': nnz[0],
        'j': nnz[1],
        'alpha': alpha[nnz[0]],
        'beta': beta[nnz[1]],
        'exact_matching_rate': exact_matching_rates[nnz],
        'sim_matching_rate': sim_matching_rates[nnz],
        'sim_matching_rate_stdev': sim_matching_rates_stdev[nnz],
        '95_CI_len': 1.96 * sim_matching_rates_stdev[nnz],
        'sim_matching_rate_CI_95_U': sim_matching_rates_upper_ci[nnz],
        'sim_matching_rate_CI_95_L': sim_matching_rates_lower_ci[nnz],
        'entropy_approx': ent_matching_rates[nnz],
        'ohm_law_approx': ohm_matching_rates[nnz]
    })

    res_df.loc[:, 'timestamp'] = dt.datetime.now() #
    res_df.loc[:, 'exact'] = exact
    res_df.loc[:, 'max_edges'] = m * n
    res_df.loc[:, 'edge_count'] = edge_count
    res_df.loc[:, 'edge_density'] = len(nnz[0])/ (m*n)
    res_df.loc[:, 'm'] = m
    res_df.loc[:, 'n'] = n
    res_df.loc[:, 'utilization'] = alpha.sum()/beta.sum()
    res_df.loc[:, 'no_of_sims'] = 30
    res_df.loc[:, 'ohm_error_sim'] = res_df['ohm_law_approx'] - res_df['sim_matching_rate']
    res_df.loc[:, 'ent_error_sim'] = res_df['entropy_approx'] - res_df['sim_matching_rate']
    res_df.loc[:, 'ohm_error_abs_sim'] = np.abs(res_df['ohm_error_sim'])
    res_df.loc[:, 'ent_error_abs_sim'] = np.abs(res_df['ent_error_sim'])
    res_df.loc[:, 'ohm_error_pct_sim'] = res_df['ohm_error_sim']/res_df['sim_matching_rate']
    res_df.loc[:, 'ent_error_pct_sim'] = res_df['ent_error_sim']/res_df['sim_matching_rate']
    res_df.loc[:, 'ohm_error_abs_pct_sim'] = np.abs(res_df['ohm_error_pct_sim'])
    res_df.loc[:, 'ent_error_abs_pct_sim'] = np.abs(res_df['ent_error_pct_sim'])
    res_df.loc[:, 'ohm_error'] = res_df['ohm_law_approx'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ent_error'] = res_df['entropy_approx'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'sim_error'] = res_df['sim_matching_rate'] - res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ohm_error_abs'] = np.abs(res_df['ohm_error']) if exact else 0
    res_df.loc[:, 'ent_error_abs'] = np.abs(res_df['ent_error']) if exact else 0
    res_df.loc[:, 'sim_error_abs'] = np.abs(res_df['sim_error']) if exact else 0
    res_df.loc[:, 'ohm_error_pct'] = res_df['ohm_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ent_error_pct'] = res_df['ent_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'sim_error_pct'] = res_df['sim_error']/res_df['exact_matching_rate'] if exact else 0
    res_df.loc[:, 'ohm_error_abs_pct'] = np.abs(res_df['ohm_error_pct']) if exact else 0
    res_df.loc[:, 'ent_error_abs_pct'] = np.abs(res_df['ent_error_pct']) if exact else 0
    res_df.loc[:, 'sim_error_abs_pct'] = np.abs(res_df['sim_error_pct']) if exact else 0
    res_df.loc[:, 'in_CI'] = (
        (res_df['exact_matching_rate'] <= res_df['sim_matching_rate_CI_95_U']) &
        (res_df['exact_matching_rate'] >= res_df['sim_matching_rate_CI_95_L'])
    ) if exact else 0
    for key, val in kwargs.items():
        res_df.loc[:, key] = val

    if print_progress:
        print('alpha: ', alpha)
        print('beta: ', beta)
        print('compatability_matrix {}x{}:'.format(m, n))

        print(np.array2string(compatability_matrix, max_line_width=np.inf))

        print('--------------------------------- ent ------------------------------------')

        print(np.array2string(ent_matching_rates, max_line_width=np.inf))

        print('--------------------------------- ohm ------------------------------------')

        print(np.array2string(ohm_matching_rates, max_line_width=np.inf))

        print('-------------------------------- exact -----------------------------------')

        print(np.array2string(exact_matching_rates, max_line_width=np.inf))

        print('--------------------------------- sim ------------------------------------')

        print(np.array2string(sim_matching_rates, max_line_width=np.inf))

        print('--------------------------------------------------------------------------')

        if exact:

            print('ohm_max: ', np.amax(np.abs(exact_matching_rates - ohm_matching_rates)))
            print('ent_max: ', np.amax(np.abs(exact_matching_rates - ent_matching_rates)))

            print('ohm_sum: ', np.abs(exact_matching_rates - ohm_matching_rates).sum())
            print('ent_sum: ', np.abs(exact_matching_rates - ent_matching_rates).sum())

        else:

            print('ohm_max: ', np.amax(np.abs(sim_matching_rates - ohm_matching_rates)))
            print('ent_max: ', np.amax(np.abs(sim_matching_rates - ent_matching_rates)))

            print('ohm_sum: ', np.abs(sim_matching_rates - ohm_matching_rates).sum())
            print('ent_sum: ', np.abs(sim_matching_rates - ent_matching_rates).sum())            

        print('x'*40)
        print('x'*40)

    res_df = res_df[cols]

    return res_df


def compare_approximations_generic_parallel(compatability_matrix, alpha, beta, jt_perms=None, exact=False, print_progress=False, aux_data=None):


    m, n = compatability_matrix.shape
    nnz = compatability_matrix.nonzero()
    edge_count = len(nnz[0])
    s = time()
    print('getting entropy approximation')
    converge, ent_matching_rates, gap_pct = entropy_approximation(compatability_matrix, alpha, beta)
    try:
        ohm_law_approx, _, _ = ohm_law_approximation(compatability_matrix, alpha, beta)
    except:
        ohm_law_approx= -1 * np.ones((m,n))


    print(time() - s)
    s = time()

    
    if exact:

        exact_matching_rates = adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta, jt_perms, print_progress)
        sim_matching_rates = np.zeros((m,n))
        sim_matching_rates_stdev = np.zeros((m,n))

    else:
        exact_matching_rates = np.zeros((m,n))

        s = time()
        sim_matching_rates, sim_matching_rates_stdev = simulate_matching_sequance(
            compatability_matrix,
            alpha,
            beta,
            sim_len=edge_count * 1000,
            sim_name=aux_data.get('sim_name', ''),
            prt=True
        )

    cols = [
        'timestamp', 'm', 'n', 'max_edges', 'edge_count', 'edge_density', 'utilization', 'exact', 'no_of_sims',
        'i', 'j', 'alpha', 'beta', 'exact_matching_rate', 'sim_matching_rate', 'sim_matching_rate_stdev',
        'entropy_approx', 'ohm_approx'] + [key for key in aux_data]


    res_df = pd.DataFrame.from_dict({
        'i': nnz[0],
        'j': nnz[1],
        'alpha': alpha[nnz[0]],
        'beta': beta[nnz[1]],
        'exact_matching_rate': exact_matching_rates[nnz],
        'sim_matching_rate': sim_matching_rates[nnz],
        'sim_matching_rate_stdev': sim_matching_rates_stdev[nnz],
        'entropy_approx': ent_matching_rates[nnz],
        'ohm_approx': ohm_law_approx[nnz]
    })

    res_df.loc[:, 'timestamp'] = dt.datetime.now() #
    res_df.loc[:, 'exact'] = exact
    res_df.loc[:, 'max_edges'] = m * n
    res_df.loc[:, 'edge_count'] = edge_count
    res_df.loc[:, 'edge_density'] = len(nnz[0])/ (m*n)
    res_df.loc[:, 'm'] = m
    res_df.loc[:, 'n'] = n
    res_df.loc[:, 'utilization'] = alpha.sum()/beta.sum()
    res_df.loc[:, 'no_of_sims'] = 30
    if aux_data is not None:
        for key, val in aux_data.items():
            res_df.loc[:, key] = val
    # if print_progress:
        # print('alpha: ', alpha)
        # print('beta: ', beta)
        # print('compatability_matrix {}x{}:'.format(m, n))

        # print(np.array2string(compatability_matrix, max_line_width=np.inf))

        # print('--------------------------------- ent ------------------------------------')

        # print(np.array2string(ent_matching_rates, max_line_width=np.inf))

        # print('--------------------------------- ohm ------------------------------------')

        # print(np.array2string(ohm_matching_rates, max_line_width=np.inf))

        # print('-------------------------------- exact -----------------------------------')

        # print(np.array2string(exact_matching_rates, max_line_width=np.inf))

        # print('--------------------------------- sim ------------------------------------')

        # print(np.array2string(sim_matching_rates, max_line_width=np.inf))

        # print('--------------------------------------------------------------------------')

        # if exact:

        #     print('ohm_max: ', np.amax(np.abs(exact_matching_rates - ohm_matching_rates)))
        #     print('ent_max: ', np.amax(np.abs(exact_matching_rates - ent_matching_rates)))

        #     print('ohm_sum: ', np.abs(exact_matching_rates - ohm_matching_rates).sum())
        #     print('ent_sum: ', np.abs(exact_matching_rates - ent_matching_rates).sum())

        # else:

        #     print('ohm_max: ', np.amax(np.abs(sim_matching_rates - ohm_matching_rates)))
        #     print('ent_max: ', np.amax(np.abs(sim_matching_rates - ent_matching_rates)))

        #     print('ohm_sum: ', np.abs(sim_matching_rates - ohm_matching_rates).sum())
        #     print('ent_sum: ', np.abs(sim_matching_rates - ent_matching_rates).sum())            

        # print('x'*40)
        # print('x'*40)

    res_df = res_df[cols]

    return res_df


def go_back_and_calc(input_filename, output_filename):#, output_filename):

    jt_perms = list(jpermute(range(10)))
    res_df = pd.read_csv(input_filename + '.csv')
    res_df_not_exact = res_df[~res_df['exact']]
    res_df_pre_exact = res_df[res_df['exact']]
    #print(res_df[['timestamp','density_level', 'n', 'graph_no', 'exact']].drop_duplicates().groupby(by=['density_level', 'graph_no','n', 'exact'], as_index=False).count())
    for key, exp in res_df_not_exact.groupby(by=['timestamp'], as_index=False):
        
        k = exp['graph_no'].iloc[0]
        l = exp['exp_no'].iloc[0]
        density = exp['density_level'].iloc[0]
        dist = exp['alph_dist'].iloc[0]

        print('k: ', k)
        print('l: ', l)
        print('density_level: ', density)
        print('dist: ', dist)

        alpha = exp[['i','alpha']].drop_duplicates().sort_values(by='i')
        #print(alpha)
        alpha = np.array(alpha['alpha'])
        m = len(alpha)
        #print(np.array2string(alpha,  max_line_width=np.inf))
        beta = exp[['j','beta']].drop_duplicates().sort_values(by='j')
        #print(beta)
        beta = np.array(beta['beta'])
        n = len(beta)
        #print(np.array2string(beta,  max_line_width=np.inf))
        edges = exp[['i','j']].drop_duplicates()
        i_idx = np.array(edges['i'])
        j_idx = np.array(edges['j'])
        #print(edges)
        compatability_matrix = np.zeros((m, n))
        compatability_matrix[i_idx, j_idx] = 1
        #print(np.array2string(compatability_matrix,  max_line_width=np.inf))
        res_df_new_exact = compare_approximations(compatability_matrix, alpha, beta, density, k, dist, l, jt_perms, exact=True, print_progress=True)
        

        if os.path.exists(output_filename + '.csv'):
            res_df_exact = pd.read_csv(output_filename + '.csv')
            res_df_exact = pd.concat([res_df_exact, res_df_new_exact], axis=0)
            res_df_exact.to_csv(output_filename + '.csv', index=False)
        else:
            res_df_exact = pd.concat([res_df_pre_exact, res_df_new_exact], axis=0)
            res_df_exact.to_csv(output_filename + '.csv', index=False)


    # i = 1
    # for key, grp in res_df.groupby(by='timestamp'):
    #     print(i, key, grp[['density_level']].drop_duplicates())
    #     i+=1

    #     print(grp[['density_level', 'n', 'm', 'i', 'j', 'alpha', 'beta', 'exact_matching_rate']])


def go_back_and_verify_ohm(infilename='FZ_Kaplan_exp', filename='w_v'):

    df = pd.read_csv(infilename + '.csv')
    bad_solutions = []
    bad_matching_rates = []
    inconsistant = []
    negative_matching_rates = []
    singuler = []

    for timestamp, exp in df.groupby(['timestamp']):

        exp_data = exp[['m', 'n', 'graph_no', 'exp_no', 'beta_dist']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = list(zip(exp['i'], exp['j']))
        # compatability_matrix[nnz] = 1.
        printarr(compatability_matrix, 'compatability_matrix')
        alpha_beta = (np.diag(alpha).dot(compatability_matrix)).dot(np.diag(beta))

        bottom_left = alpha_beta.T
        upper_right = -1 * alpha_beta
        upper_left = np.diag(alpha_beta.sum(axis=1))
        bottom_right = -1 * np.diag(alpha_beta.sum(axis=0))

        A = np.vstack([
            np.hstack([upper_left, upper_right]), 
            np.hstack([bottom_left, bottom_right]),
            ])

        b = np.vstack([alpha.reshape((m, 1)), beta.reshape((n, 1))])
        try:
            v_w = np.linalg.solve(A, b)
        except:
            print('singuler: ', timestamp)
            singuler.append(timestamp)
        
        if not np.allclose(np.dot(A, v_w), b):
            print('bad_solution: ', timestamp)
            bad_solutions.append(timestamp)
            continue

        v = np.squeeze(v_w[:m,:])
        w = np.squeeze(v_w[m:,:])
        w_n = w[n-1]
        v = v - w_n
        w = w - w_n
        v_i = [v[int(i)] for i in exp['i']]
        w_j = [w[int(j)] for j in exp['j']]

        # print('v', v)
        # print('w', w)
        # print(np.dot(A, v_w))
        
        delta_v_w = (np.diag(v).dot(compatability_matrix) - compatability_matrix.dot(np.diag(w)))
        matching_rates = delta_v_w * alpha_beta

        prev_matching_rates = np.zeros((m,n))

        for k, row in exp.iterrows():
            prev_matching_rates[int(row['i']), int(row['j'])] = float(row['ohm_law_approx'])

        if not np.allclose(prev_matching_rates, matching_rates):
            print('inconsistant: ', timestamp)
            inconsistant.append(timestamp)            

        if not np.allclose(matching_rates.sum(axis=1), alpha) or not np.allclose(matching_rates.sum(axis=0), beta):
            print('bad_matching_rates: ', timestamp)
            bad_matching_rates.append(timestamp)

        if not np.allclose(matching_rates.sum(axis=1), alpha) or not np.allclose(matching_rates.sum(axis=0), beta):
            print('bad_matching_rates: ', timestamp)
            bad_matching_rates.append(timestamp)


        if (1*(matching_rates < 0)).sum() > 0:
            negative_matching_rates.append(timestamp)

        w_v_df = pd.DataFrame.from_dict({'i': [i for i in exp['i']], 'j':[j for j in exp['j']], 'v_i': v_i, 'w_j': w_j})
        w_v_df.loc[:, 'timestamp'] = timestamp
        #print(w_v_df)
        if os.path.exists(filename + '.csv'):
            res_df = pd.read_csv(filename + '.csv')
            res_df = pd.concat([res_df, w_v_df], axis=0)
            res_df.to_csv(filename + '.csv', index=False)
        else:
            res_df = w_v_df
            res_df.to_csv(filename + '.csv', index=False)


    if True: 
        bad_solutions = pd.DataFrame.from_dict({'timestamp': bad_solutions})
        bad_matching_rates = pd.DataFrame.from_dict({'timestamp': bad_matching_rates})
        inconsistant = pd.DataFrame.from_dict({'timestamp': inconsistant})
        negative_matching_rates = pd.DataFrame.from_dict({'timestamp': negative_matching_rates})
        singuler = pd.DataFrame.from_dict({'timestamp': singuler})
        bad_solutions.loc[:, 'problem'] = 'bad_solutions'
        bad_matching_rates.loc[:, 'problem'] = 'bad_matching_rates'
        inconsistant.loc[:, 'problem'] = 'inconsistant'
        negative_matching_rates.loc[:, 'problem'] = 'negative_matching_rates'
        singuler.loc[:, 'problem'] = 'singuler'

        bads = pd.concat([bad_solutions ,bad_matching_rates ,inconsistant ,negative_matching_rates, singuler], axis=0)

        bads.to_csv('bads.csv', index=False)


def ohm_singular(filename='FZ_Kaplan_exp'):

    res_df = pd.read_csv(filename + '.csv')
    
    for timestamp in ['2019-05-12 22:07:58.414378', '2019-05-12 22:33:32.437647', '2019-05-12 23:50:08.914732', '2019-05-13 04:38:47.078754']:

        exp = res_df[res_df['timestamp'] == timestamp]   
        exp_data = exp[['m', 'n', 'graph_no', 'exp_no', 'beta_dist']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        col_diag_2_powers = np.diag([2**i for i in range(n)])
        row_diag_2_powers = np.diag([2**i for i in range(m)])

        count_unique_rows = len(np.unique(np.dot(compatability_matrix, col_diag_2_powers).sum(axis=1)))
        if count_unique_rows != m:
            print('identical')
        count_unique_cols = len(np.unique(np.dot(row_diag_2_powers, compatability_matrix).sum(axis=1)))
        if count_unique_cols != n:
            print('identical')

        printarr(compatability_matrix, timestamp)


def go_back_and_solve_qp(df):

    quad_df_list = []

    for timestamp, exp in df.groupby(['timestamp']):

        exp_data = exp[['m', 'n','exp_num']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()
        try:
            quad_matching_rates = quadratic_approximation(compatability_matrix, alpha, beta)
        except:
            print('bad_quad:', timestamp)
            quad_matching_rates = np.zeros((m,n))
        if quad_matching_rates is None:
            print('bad_quad:', timestamp)
            quad_matching_rates = np.zeros((m,n))
        try:
            ohm_matching_rates,_,_ = ohm_law_approximation(compatability_matrix, alpha, beta)
        except:
            print('bad_ohm:', timestamp)
            ohm_matching_rates = np.zeros((m,n))
        if ohm_matching_rates is None:
            print('bad_ohm:', timestamp)
            ohm_matching_rates = np.zeros((m,n))


        quad_df_k = pd.DataFrame.from_dict({
            'i': nnz[0],
            'j': nnz[1],
            'quad_approx': quad_matching_rates[nnz],
            'ohm_approx': ohm_matching_rates[nnz]
        })

        quad_df_k.loc[:, 'timestamp'] = timestamp
        quad_df_list.append(quad_df_k)

    quad_df = pd.concat(quad_df_list, axis=0)

    df = pd.merge(left=df, right=quad_df, on=['timestamp', 'i', 'j'], how='left')

    return df


def go_back_and_approximate_sbpss(filename='erdos_renyi_exp4'):

    df = pd.read_csv(filename + '.csv')
    newfilename = 'erdos_renyi_exp4_w_approx'

    p = 8
    k = 0
    exps = []
    pool = mp.Pool(processes=8)
    for n in range(7, 11, 1):
        for timestamp, exp in df[df['n'] == n].groupby(by=['timestamp'], as_index=False):
            exps.append([exp, timestamp])
        print('no_of_exps:', len(exps), 'n:', n)
        print('starting work with {} cpus'.format(p))
        sbpss_dfs = pool.starmap(approximate_sbpss, exps)
        exps = []
        sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
        write_df_to_file('FZ_Kaplan_exp_sbpss2', sbpss_df)


def approximate_sbpss(exp, timestamp):


        exp_data = exp[['m', 'n', 'graph_no', 'exp_no', 'beta_dist']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]

        print('graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist)

        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()

        pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
        no_of_edges = len(nnz[0])

        sbpss_df = []
        exact = n <= 10

        for rho in [0.01] + [0.05] + [0.1*i for i in range(1, 10, 1)] + [.95, .99]:

            
            lamda = alpha * rho
            mu = beta
            pad_lamda = np.append(alpha*rho, 1. - rho)

            print(
                'rho:' , rho, '\n',
                'lamda: ', np.array2string(lamda, max_line_width=np.inf), '\n',
                'mu', np.array2string(mu, max_line_width=np.inf)
                )

            _, heavy_traffic_approx_entropy, _ = entropy_approximation(pad_compatability_matrix, pad_lamda, mu)
            heavy_traffic_approx_entropy = heavy_traffic_approx_entropy[:m, :]
            low_traffic_approx_entropy = local_entropy(compatability_matrix, alpha * rho, beta)
            if exact:
                heavy_traffic_approx_exact = adan_weiss_fcfs_alis_matching_rates(pad_compatability_matrix, pad_lamda, mu, print_progress= False)
                heavy_traffic_approx_exact = heavy_traffic_approx_exact[:m, :]
            else:
                heavy_traffic_approx_exact = np.zeros((m,n))

            print('ending___ graph_no: ', graph_no, 'exp_no: ', exp_no, 'beta_dist: ', beta_dist, 'rho: ', rho, 'duration: ', time() - s)
            sim_matching_rates, sim_matching_rates_stdev = simulate_queueing_system(compatability_matrix, alpha * rho, beta, sims=30)

            if exact:
                print('pct_error_rho_exact:'    , np.abs(sim_matching_rates - ((1 - rho) * low_traffic_approx_entropy + (rho) * heavy_traffic_approx_exact)).sum()/rho)
                print('pct_error_rho^2_exact:'  , np.abs(sim_matching_rates - ((1 - rho**2) * low_traffic_approx_entropy + (rho**2) * heavy_traffic_approx_exact)).sum()/rho)
            print('pct_error_rho_entropy:'  , np.abs(sim_matching_rates - ((1 - rho) * low_traffic_approx_entropy + (rho) * heavy_traffic_approx_entropy)).sum()/rho)
            print('pct_error_rho_2_entropy:', np.abs(sim_matching_rates - ((1 - rho**2) * low_traffic_approx_entropy + (rho**2) * heavy_traffic_approx_entropy)).sum()/rho)

            sbpss_rho_df = pd.DataFrame.from_dict({
                'i': nnz[0],
                'j': nnz[1],
                'sim_matching_rates': sim_matching_rates[nnz],
                'sim_matching_rates_stdev': sim_matching_rates_stdev[nnz],
                'heavy_traffic_approx_entropy': heavy_traffic_approx_entropy[nnz], 
                'heavy_traffic_approx_exact':heavy_traffic_approx_exact[nnz], 
                'low_traffic_approx_entropy': low_traffic_approx_entropy[nnz], 
                'rho_approx_w_exact': rho * heavy_traffic_approx_exact[nnz] + (1. - rho) * low_traffic_approx_entropy[nnz], 
                'rho^2_approx_w_exact': (rho**2) * heavy_traffic_approx_exact[nnz] + (1. - rho**2) * low_traffic_approx_entropy[nnz],# 'rho_approx': rho * heavy_traffic_approx_entropy[nnz] + (1. - rho) * low_traffic_approx_entropy[nnz],
                'rho_2_approx': (rho**2) * heavy_traffic_approx_entropy[nnz] + (1. - rho**2) * low_traffic_approx_entropy[nnz]
            })

            sbpss_rho_df.loc[:, 'timestamp'] = timestamp
            sbpss_rho_df.loc[:, 'rho'] = rho
            sbpss_rho_df.loc[:, 'm'] = m
            sbpss_rho_df.loc[:, 'n'] = n
            sbpss_rho_df.loc[:, 'graph_no'] = graph_no
            sbpss_rho_df.loc[:, 'exp_no'] = exp_no
            sbpss_rho_df.loc[:, 'beta_dist'] = beta_dist

            sbpss_df.append(sbpss_rho_df)

        return sbpss_df


def go_back_and_approximate_sbpss_customer_dependet(filename='FZ_final_w_qp'):

    df = pd.read_csv(filename + '.csv')
    p = 7
    pool = mp.Pool(processes=p)

    for n in range(7,11,1):
        exps = []
        for timestamp, exp in df[df['n'] == n].groupby(by=['timestamp'], as_index=False):
            exps.append([exp, timestamp])
            if len(exps) == p:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_customer_dependent, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_cd', sbpss_df)
                exps = []
        else:
            if len(exps) > 0:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_customer_dependent, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_cd', sbpss_df)
                exps = []   



def approximate_sbpss_customer_dependent(exp, timestamp):

        exp_data = exp[['m', 'n', 'graph_no', 'exp_num', 'beta_dist']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_num'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]

        print('graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist)

        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()

        # pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
        no_of_edges = len(nnz[0])

        sbpss_df = []

        for split in ['one', 'half', 'rand']:

            if split == 'one':
                theta = np.ones(m)
            elif split == 'half':
                theta = 0.5 * np.ones(m)
            else:
                theta = np.random.uniform(0.1, 0.9, m)

            for rho in [0.01] + [0.05] + [0.1*i for i in range(1, 10, 1)] + [.95, .99]:
            # for rho in [0.1, 0.9]:
                
                st = time()            
                lamda = np.ones(m) * rho * theta
                s = alpha * (1./theta)
                eta = lamda * s
                mu = beta

                print(
                    'rho:' , rho, '\n',
                    'lamda: ', np.array2string(lamda, max_line_width=np.inf), '\n',
                    'eta: ', np.array2string(eta, max_line_width=np.inf), '\n',
                    'mu', np.array2string(mu, max_line_width=np.inf)
                    )

                _, heavy_traffic_approx_entropy_eta, _ = entropy_approximation(compatability_matrix, eta, mu, pad=True)
                heavy_traffic_approx_entropy = np.dot(np.diag(1./s), heavy_traffic_approx_entropy_eta)
                low_traffic_approx_entropy = local_entropy(compatability_matrix, lamda, beta, s)
                heavy_traffic_approx_exact = np.zeros((m,n))
                sim_results = simulate_queueing_system(compatability_matrix, lamda, beta, s=s, sims=30)
                print('ending - graph_no: ', graph_no, 'exp_no: ', exp_no, 'beta_dist: ', beta_dist, 'rho: ', rho, 'split:', split ,'duration: ', time() - st)
                print('pct_error_rho_entropy:'  , np.abs(sim_results['matching_rates_mean'] - ((1 - rho) * low_traffic_approx_entropy + (rho) * heavy_traffic_approx_entropy)).sum()/lamda.sum())
                sbpss_rho_df = pd.DataFrame.from_dict({
                    'i': nnz[0],
                    'j': nnz[1],
                    'alpha': alpha[nnz[0]],
                    'beta': beta[nnz[1]],
                    'theta': theta[nnz[0]],
                    'lamda': lamda[nnz[0]],
                    'mu': beta[nnz[1]],
                    's': s[nnz[0]], 
                    'sim_matching_rates': sim_results['matching_rates_mean'][nnz],
                    'sim_matching_rates_stdev': sim_results['matching_rates_stdev'][nnz],
                    'heavy_traffic_approx_entropy': heavy_traffic_approx_entropy[nnz],  
                    'low_traffic_approx_entropy': low_traffic_approx_entropy[nnz], # 'rho_approx': rho * heavy_traffic_approx_entropy[nnz] + (1. - rho) * low_traffic_approx_entropy[nnz],
                    'rho_approx': (rho) * heavy_traffic_approx_entropy[nnz] + (1. - rho) * low_traffic_approx_entropy[nnz],
                    'waiting_time': sim_results['waiting_times_mean'][nnz[0]],
                    'waiting_time_stdev': sim_results['waiting_times_stdev'][nnz[0]],
                    'sig_waiting_time_mean' : sim_results['waiting_times_stdev_mean'][nnz[0]],
                    'sig_waiting_time_stdev': sim_results['waiting_times_stdev_stdev'][nnz[0]]
                })

                sbpss_rho_df.loc[:, 'timestamp'] = timestamp
                sbpss_rho_df.loc[:, 'rho'] = rho
                sbpss_rho_df.loc[:, 'm'] = m
                sbpss_rho_df.loc[:, 'n'] = n
                sbpss_rho_df.loc[:, 'graph_no'] = graph_no
                sbpss_rho_df.loc[:, 'exp_no'] = exp_no
                sbpss_rho_df.loc[:, 'beta_dist'] = beta_dist
                sbpss_rho_df.loc[:, 'split'] = split

                sbpss_df.append(sbpss_rho_df)

        return sbpss_df


def find_bad_sum():

    df_full = pd.read_csv('FZ_Kaplan_exp_sbpss_fixed_adj.csv')


    max_cases = []

    for key, grp in df_full.groupby(['timestamp','rho']):

        exp_data = grp[['m', 'n', 'graph_no', 'exp_no', 'beta_dist', 'rho']].drop_duplicates()

        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        rho = exp_data['rho'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]

        sum_mr = np.asscalar(grp[['sim_matching_rates']].sum())
        sum_ha = np.asscalar(grp[['heavy_approx']].sum())
        sum_la = np.asscalar(grp[['light_approx']].sum())

        if (np.abs(sum_mr - sum_ha)/sum_mr) > 0.05:
            print(exp_data)
            print(sum_mr)
            print(sum_ha)

        # for k, row in alpha_data.iterrows():
        #     alpha[int(row['i'])] = float(row['alpha'])

        # for k, row in beta_data.iterrows():
        #     beta[int(row['j'])] = float(row['beta'])

        # for k, row in case_df.iterrows():

        #     compatability_matrix[int(row['i']), int(row['j'])] = 1
        #     matching_rates[int(row['i']), int(row['j'])] = float(row['sim_matching_rates'])
        #     rho_approx[int(row['i']), int(row['j'])] = float(row['rho_approx'])
        #     heavy_approx[int(row['i']), int(row['j'])] = float(row['heavy_approx'])
        #     light_approx[int(row['i']), int(row['j'])] = float(row['light_approx'])
        #     print('{},{},{:.5f},{:.5f},{:.5f}'.format(row['i'], row['j'], row['sim_matching_rates'], row['heavy_approx'], np.abs(row['heavy_approx'] - row['sim_matching_rates'])))

        # printarr(matching_rates.sum(axis=0), 'mr_sum_0')
        # printarr(heavy_approx.sum(axis=0), 'ha_sum_0')
        # printarr(matching_rates.sum(axis=1), 'mr_sum_1')
        # printarr(heavy_approx.sum(axis=1), 'ha_sum_1')
        # printarr(matching_rates.sum(), 'mr_sum')
        # printarr(heavy_approx.sum(), 'ha_sum')
        # printarr(matching_rates, 'sim_matching_rates')
        # printarr(heavy_approx, 'heavy_approx')
        # printarr(light_approx, 'light_approx')


def go_back_and_adjust_sums():


    df_full = pd.read_csv('FZ_Kaplan_exp_sbpss_fixed.csv')

    sim_sum_rates = df_full[['timestamp', 'rho', 'sim_matching_rates']].groupby(by=['timestamp', 'rho'], as_index=False).sum().rename(columns={'sim_matching_rates':'sum_sim_rates'})
    df_full = pd.merge(left=df_full, right=sim_sum_rates, on=['rho', 'timestamp'], how='left')
    df_full.loc[:, 'adj_sim_matching_rates'] = df_full['sim_matching_rates'] * df_full['rho']/df_full['sum_sim_rates']
    df_full.loc[:, 'sim_rate_gap'] = np.abs(df_full['rho'] - df_full['sum_sim_rates'])/df_full['rho']

    df_full.to_csv('FZ_Kaplan_exp_sbpss_fixed_adj.csv')


def fix_bad():


    df = pd.read_csv('FZ_Kaplan_exp_sbpss_bad2.csv')

    cols = list(df.columns.values)

    df = df.drop(columns=['light_approx'])

    correct_light = []    

    for key, exp in df.groupby(['timestamp', 'rho']):

        timestamp, rho = key
        exp_data = exp[['m', 'n', 'graph_no', 'exp_no', 'beta_dist', 'rho']].drop_duplicates()
        alpha_data = exp[['i', 'alpha']].drop_duplicates()
        beta_data = exp[['j', 'beta']].drop_duplicates()
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        rho = exp_data['rho'].iloc[0]
        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in exp.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()
        light_approx = local_entropy(compatability_matrix, alpha * rho, beta, prt=False)
        printarr(light_approx)
        correct_light.append(
            pd.DataFrame.from_dict({
                'timestamp': [timestamp]*len(nnz[0]),
                'rho': [rho]*len(nnz[0]),
                'i': nnz[0],
                'j': nnz[1],
                'light_approx': light_approx[nnz]
            })
            )
    correct_light = pd.concat(correct_light, axis=0)

    df = pd.merge(left=df, right=correct_light, how='left', on=['timestamp', 'rho' ,'i', 'j'])
    df = df[cols]
    df_good = pd.read_csv('FZ_Kaplan_exp_sbpss_good2.csv')
    df = pd.concat([df, df_good], axis=0).sort_values(by=['timestamp','rho' ,'i', 'j'])
    df['rho_approx'] = df['rho'] * df['heavy_approx'] + (1. - df['rho']) * df['light_approx']
    df['rho_2_approx'] = (df['rho']**2) * df['heavy_approx'] + (1. - df['rho']**2) * df['light_approx']
    df.to_csv('FZ_Kaplan_exp_sbpss_fixed.csv', index=False)


def join_df_from_files(files_to_be_joined, join_cols, writefile):
    
    main_df = pd.read_csv(files_to_be_joined[0] + '.csv')
    for file in  files_to_be_joined[1:]:

        df = pd.read_csv(file + '.csv')
        main_df = pd.merge(left=main_df, right=df, on=join_cols, how='left')

    main_df.to_csv(writefile + '.csv', index=False)    


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    go_back_and_approximate_sbpss_customer_dependet()
    # df = pd.read_csv('erdos_renyi_exp_final.csv')
    # df = go_back_and_solve_qp(df)
    # df.to_csv('erdos_renyi_exp_final_w_qp.csv', index=False)
    # go_back_and_solve_qp('grids_exp_parallel_extra_9x9')
    # join_df_from_files(['grids_exp_parallel_extra_9x9', 'grids_exp_parallel_extra_w_qp'], join_cols=['timestamp', 'i', 'j'], writefile='grids_exp_parallel_w_qp_full')


    

    
    # comparison_graph4('FZ_Kaplan_exp', ['ent', 'diss'])
    # m,n = compatability_matrix.shape
    # compatability_matrix, alpha, beta = BASE_EXAMPLES[6]

    # matching_rates_g= quadratic_approximation_gurobi(compatability_matrix, alpha, beta)
    # matching_rates_c= quadratic_approximation(compatability_matrix, alpha, beta)

    # printarr(matching_rates_g, 'gurobi')
    # printarr(matching_rates_c, 'cplex')
    # rho = 0.8
    # simulate_queueing_system(compatability_matrix, np.one(m) * rho, beta, s=alpha ,sims=30)

    # sps_compatability_matrix = sps.csr_matrix(compatability_matrix)
    # mr, _ = simulate_matching_sequance(sps_compatability_matrix, alpha, beta, sims=30)
    # mre  = adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta)

    # printarr(mr, 'sim')
    # printarr(mre, 'exact')
    # simulate_queueing_system(compatability_matrix, alpha * 0.9, beta, prt=False, sims=3, sim_len=1000000, seed=1, sim_name='sim')
    # m, n = compatability_matrix.shape
    # # another_type_of_entropy(compatability_matrix, alpha * 0.2, beta)

    # # m, n = compatability_matrix.shape

    # for rho in np.arange(.05, 1., .05):
    #     print('-'*75)
    #     print('-'*35 + str(rho) + '-'*35)
    #     print('-'*75)

    # reses = []
    # print('--rho--', '--mrn-', '-mrl--', '--mre-', '-mren-', '-mrel-', '-mrnl-')
    # for rho in np.append(np.arange(0.1, 1., .1), .95) :

    #     mrn = node_entropy(compatability_matrix, alpha*rho, beta)
    #     mrl = local_entropy(compatability_matrix, alpha*rho, beta)
    #     _, mre, _ = entropy_approximation(compatability_matrix, alpha*rho, beta, pad=True)

    #     mr, mr_stdev = simulate_queueing_system(compatability_matrix, alpha*rho, beta, sim_len=100000, sims=30)

    #     mren = mrn*(1-rho) + mre*(rho)
    #     mrel = mrl*(1-rho) + mre*(rho)
    #     mrnl = mrl*(1-rho) + mrn*(rho)

    #     mrn = np.asscalar(np.abs(mr - mrn).sum()/rho)
    #     mrl = np.asscalar(np.abs(mr - mrl).sum()/rho)
    #     mre = np.asscalar(np.abs(mr - mre).sum()/rho)
    #     mren = np.asscalar(np.abs(mr - mren).sum()/rho)
    #     mrel = np.asscalar(np.abs(mr - mrel).sum()/rho)
    #     mrnl = np.asscalar(np.abs(mr - mrnl).sum()/rho)



    #     print('{:.2f}: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(rho, mrn, mrl, mre, mren, mrel, mrnl))


    #     pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
    #     pad_lam = np.append(alpha*rho, 1. - rho)

    #     mr_a1 = adan_weiss_fcfs_alis_matching_rates(pad_compatability_matrix, pad_lam, beta)

    #     mr_a1 = mr_a1[:m, :]        

    #     # mr_a = adan_weiss_fcfs_alis_matching_rates(pad_compatability_matrix, pad_lam, beta)
    #     mr_a2 = another_type_of_entropy(compatability_matrix, alpha * rho, beta)

    #     _, mr_a4, _ = entropy_approximation(pad_compatability_matrix, pad_lam, beta)
    #     mr_a4 = mr_a4[:m, :]


    #     mr_a3 = (mr_a2 * (1 - rho**2)) + (mr_a1 * rho**2)
    #     mr_a5 = (mr_a2 * (1 - rho**2)) + (mr_a4 * rho**2)

    #     printarr(mr, 'sim')
    #     printarr(mr_a1, 'approx1')
    #     printarr(mr_a2, 'approx2')
    #     printarr(mr_a3, 'approx3')
    #     printarr(mr_a4, 'approx4')
    #     printarr(mr_a5, 'approx5')
        # printarr(np.dot(np.diag(1./mr.sum(axis=1)), mr), 'sim_pct')
        # printarr(np.dot(np.diag(1./(alpha * rho)), mr_a), 'approx_pct')
        # printarr(mr.sum(axis=0), 'col_sums_sim')
        # printarr(mr.sum(axis=1), 'row_sums_sim')
        # printarr(mr_a.sum(axis=0), 'col_sums_approx')
        # printarr(mr_a.sum(axis=1), 'row_sums_approx')
        # printarr(mr_a[:m, :].sum(axis=0), 'col_sums_approx')
        # printarr(mr_a[:m, :].sum(axis=1), 'row_sums_approx')
        #printarr(mr.sum(axis=0), 'sim_work: ' + str(mr.sum()))
        #printarr(beta - mr_a, 'approx_work:' + str())
        # # printarr(beta - mr_a[m, :], 'approx_work:' + str())
        # printarr(1. - ((beta - mr.sum(axis=0))/ beta), 'sim_util')
        # printarr((beta - mr_a[m, :]) / beta, 'approx_util')

        #print('sim_util_sum :', mr.sum())
        #print('approx_util_sum :', (beta - mr_a).sum())
        # print('approx_util_sum :', (beta - mr_a[m, :]).sum())

        # print('sum_abs_error_1/flow:', np.abs(mr-mr_a1).sum()/rho)
        # print('sum_abs_error_2/flow:', np.abs(mr-mr_a2).sum()/rho)
        # print('sum_abs_error_3/flow:', np.abs(mr-mr_a3).sum()/rho)
        # print('sum_abs_error_4/flow:', np.abs(mr-mr_a4).sum()/rho)
        # print('sum_abs_error_5/flow:', np.abs(mr-mr_a5).sum()/rho)
        # print(np.abs(mr-mr_a[:m, :]).sum())
        #  print('---------------------------')

    # mr = dissapative_approximation(compatability_matrix, alpha, beta)

    # printarr(mr, 'dissapative')

    # mr = adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta)

    # printarr(mr, 'exact')
