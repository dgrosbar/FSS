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
from copy import copy

from simulators import *
from generators import *
from utilities import *
from mr_calc_and_approx import *
from graphs_and_tabels import *
import gc

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


def create_maps(m,d, zeta):

     for k in range(30):
        compatability_matrix, g, alpha, beta, workload_sets, rho_m, rho_n, node_map = generate_grid_compatability_matrix_with_map(m, d , zeta)
        c_i = -1* np.ones(m**2)
        for workload_set in workload_sets.values():
            for key, val in workload_set.items():
                print(key, val)
            c = len(workload_set['supply_nodes'])
            for node in workload_set['demand_nodes']:
                c_i[node] = c

        node_map = pd.DataFrame(data=node_map, columns=['node','x','y'])

        res_dict = {'col': dict(), 'row': dict(), 'mat': dict(), 'aux': dict()}
        res_dict['col']['rho_j_MMF'] = rho_n
        res_dict['row']['rho_i_MMF'] = rho_m
        res_dict['row']['c_MMF'] = c_i
        res_dict['aux']['exp_no'] = k
        res_dict['aux']['arc_dist'] = s
        res_dict['aux']['zeta'] = zeta

        map_df = log_res_to_df(compatability_matrix, alpha=alpha, beta=alpha, result_dict=res_dict)
        map_df = pd.merge(left=map_df, right=node_map.rename(columns={'x': 'x_i', 'y': 'y_i', 'node': 'i'}), on='i')
        map_df = pd.merge(left=map_df, right=node_map.rename(columns={'x': 'x_j', 'y': 'y_j', 'node': 'j'}), on='j')
        write_df_to_file('map_exps', map_df)


def map_exp_for_parallel(p=30, filename='map_exps'):

    print_progress = True
    input_df = pd.read_csv(filename + '.csv') 
    exps = []
    for timestamp, exp in input_df.groupby(by='timestamp'):
        exps.append([timestamp, exp])
    for structure in ['torus']:
        if p > 1:
            pool = mp.Pool(processes=p)
            exps_res = pool.starmap(sbpss_exp, exps)
            pool.close()
            pool.join()
        else:
            sbpss_exp(*exps[0])


def sbpss_exp(timestamp, exp, filename):

    exp_data = exp[['m', 'n', 'exp_no']].drop_duplicates()
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

    valid = False
    np.random.seed(k)
    v = np.random.randint(1, 10**6)
    np.random.seed(v)
    # aux_exp_data = {'size': str(m) + 'x' + str(m), 'arc_dist': d, 'structure': '', 'exp_no': k, 'seed': v}    
    s = np.ones(m)
    c = np.zeros((m, n))
    nnz = compatability_matrix.nonzero()
    pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
    no_of_edges = len(nnz[0])    
    exact = n <= 10

    for rho in [0.6, 0.7, 0.8, 0.9] + [.95, .99, 1] + [0.01, 0.05, 0.1, 0.2, 0.4]:

        st = time()

        lamda = alpha * rho
        mu = beta
        pad_lamda = np.append(alpha*rho, 1. - rho)

        fcfs_approx = fast_entropy_approximation(compatability_matrix, lamda, mu, pad=(rho < 1))
        
        if rho < 1:
            fcfs_approx = fcfs_approx[:m]
        try:
            alis_approx = fast_alis_approximation(1. * compatability_matrix, alpha, beta, rho) if m < 900 else np.zeros((m, n))
        except:
            alis_approx = np.zeros((m, n))
        
        q_fcfs = fcfs_approx * (1./mu - fcfs_approx.sum(axis=0))
        q_fcfs = q_fcfs/q_fcfs.sum(axis=0)

        if rho >= 0.6 and rho < 1:
            
            r_fcfs_weighted, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, 0, weighted=True)
            if r_fcfs_weighted is not None:
                r_fcfs_weighted = r_fcfs_weighted[:m, :]
                q_fcfs_weighted = r_fcfs_weighted * (1./mu - r_fcfs_weighted.sum(axis=0))
                q_fcfs_weighted = q_fcfs_weighted/q_fcfs_weighted.sum(axis=0)
                w_fcfs_weighted  = np.divide(q_fcfs_weighted, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))
                w_exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted, prt_all=True, prt=True)

                w_exp_res['mat']['fcfs_approx'] = r_fcfs_weighted
                w_exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
                w_exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * w_exp_res['mat']['alis_approx'] + (rho) * w_exp_res['mat']['fcfs_approx']

                w_exp_res['aux']['rho'] = rho
                w_exp_res['aux']['gamma'] = 0
                w_exp_res['aux']['policy'] = 'weighted_fcfs_alis'

                sbpss_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, w_exp_res, timestamp, aux_exp_data)
                write_df_to_file(filename, sbpss_df)
        
        if rho == 1:
            exp_res = simulate_matching_sequance(compatability_matrix, alpha, beta, prt_all=True, prt=True)
        else:
            exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, prt_all=True, prt=True)
        
        exp_res['mat']['fcfs_approx'] = fcfs_approx
        exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
        exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * exp_res['mat']['alis_approx'] + (rho) * exp_res['mat']['fcfs_approx']
        
        exp_res['aux']['rho'] = rho
        exp_res['aux']['gamma'] = 0
        exp_res['aux']['policy'] = 'fcfs_alis'

        sbpss_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, exp_res, timestamp, aux_exp_data)
        write_df_to_file(filename, sbpss_df)

        print('ending - structure: ', aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'], ' rho: ', rho, ' duration: ', time() - st)
        print('pct_error_fcfs_alis_approx:'  , np.abs(exp_res['mat']['sim_matching_rates'] - exp_res['mat']['fcfs_alis_approx']).sum()/lamda.sum())
        
        gc.collect()


    def log_ot_data(res, c, w , q, gamma, policy, rho, c_type):

        res['mat']['c'] = c
        res['mat']['w'] = w
        res['mat']['q'] = q 
        res['aux']['gamma'] = gamma
        res['aux']['policy'] = policy
        res['aux']['rho'] = rho
        res['aux']['c_type'] = c_type

        return res

    for c_type in ['dist']:

        for rho in [.4, .6, .8, .9, .95, .99]:

            lamda = rho * alpha
            mu = beta 
            s = np.ones(m)

            if c_type == 'rand':
                c = np.random.exponential(1, (m, n)) * compatability_matrix
            else:
                c = np.zeros((m, n))
                for i in range(m):
                    for j in range(n):
                        c[i,j] = 1 + abs(i - j)
                c = c * compatability_matrix
                c = c/c.sum()

            lamda_pad = np.append(lamda, mu.sum() - lamda.sum())
            c_pad = np.vstack([c, np.zeros((1, n))])
            compatability_matrix_pad = np.vstack([compatability_matrix, np.ones((1, n))])

            r_pad = sinkhorn_stabilized(c_pad, lamda_pad, mu, compatability_matrix_pad, 0.01)
            min_c = (c_pad * r_pad).sum()
            r_pad = sinkhorn_stabilized(-1*c_pad, lamda_pad, mu, compatability_matrix_pad, 0.01)
            max_c = (c_pad * r_pad).sum()
            c_diff = max_c - min_c

            r_fcfs = entropy_approximation(compatability_matrix, lamda, mu, pad=True)
            q_fcfs = r_fcfs * (1./(mu - r_fcfs.sum(axis=0)))
            q_fcfs = q_fcfs/q_fcfs.sum(axis=0)

            min_ent = (-1 * lamda * np.log(lamda)).sum()
            max_ent = (-1 * r_fcfs * np.log(r_fcfs, out=np.zeros_like(r_fcfs), where=(r_fcfs != 0))).sum()

            ent_diff = max_ent - min_ent
            c = (ent_diff/c_diff) * c

            
            for gamma in [0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]:

                print('gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])

                r_fcfs_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=False)
                if r_fcfs_ot is None:
                    print('failed unweighted ', 'gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])
                r_fcfs_weighted_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=True)
                if r_fcfs_weighted_ot is None:
                    print('failed weighted ', 'gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])

                if r_fcfs_ot is not None and r_fcfs_weighted_ot is not None:

                    r_fcfs_ot = r_fcfs_ot[:m, :]
                    q_fcfs_ot = r_fcfs_ot * (1./mu - r_fcfs_ot.sum(axis=0))
                    q_fcfs_ot = q_fcfs_ot/q_fcfs_ot.sum(axis=0)
                    w_fcfs_ot = np.divide(q_fcfs_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_ot = log_ot_data(sim_res_fcfs_alis_ot, c, w_fcfs_ot , q_fcfs_ot, gamma, 'fcfs_alis_ot', rho, c_type)
                    df_fcfs_alis_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename, df_fcfs_alis_ot)

                    r_fcfs_weighted_ot = r_fcfs_weighted_ot[:m, :]
                    q_fcfs_weighted_ot = r_fcfs_weighted_ot * (1./mu - r_fcfs_weighted_ot.sum(axis=0))
                    q_fcfs_weighted_ot = q_fcfs_weighted_ot/q_fcfs_weighted_ot.sum(axis=0)
                    w_fcfs_weighted_ot  = np.divide(q_fcfs_weighted_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_weighted_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_weighted_ot = log_ot_data(sim_res_fcfs_alis_weighted_ot, c, w_fcfs_weighted_ot,  q_fcfs_weighted_ot, gamma, 'weighted_fcfs_alis_ot', rho, c_type,)
                    df_fcfs_alis_weighted_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_weighted_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename, df_fcfs_alis_weighted_ot)

            w_greedy = np.divide(np.ones(c.shape), c, out=np.zeros_like(c), where=(c != 0))
            sim_res_greedy = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_greedy, w_only=True,  prt_all=True, prt=True)
            sim_res_greedy = log_ot_data(sim_res_greedy, c, c , 0 * compatability_matrix, 1, 'greedy', rho, c_type)

            df_greedy = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_greedy, timestamp, aux_exp_data)
            write_df_to_file(ot_filename, df_greedy)

    gc.collect()




    return None
   


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    create_maps(10)