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
# from graphs_and_tabels import *
import gc


def erdos_renyi_exp_for_parallel(p=30):

    jt_perm_dict = {9: list(jpermute(range(9)))}
    print_progress = True
    for structure in ['erdos_renyi']:
        for m in [100, 1000]:
            if p > 1:
                exps = [list(tup) for tup in zip([m]*p, range(1, p+1, 1), [structure]*p)]
                pool = mp.Pool(processes=p)
                exps_res = pool.starmap(sbpss_exp, exps)
                pool.close()
                pool.join()
            else:
                sbpss_exp(m, 1, structure)
        
def sbpss_exp(m, k, structure, filename='erdos_renyi_sbpss_uni_mu', ot_filename='erdos_renyi_sbpss_ot_uni_mu'):

    n = m
    p_edge = 2*log(m)/m
    aux_data = {'exp_num': k, 'size': str(m) + 'x' + str(m), 'structure': structure, 'p_edge': p_edge}

    valid = False
    np.random.seed(k)
    v = np.random.randint(1, 10**5)
    np.random.seed(v)
    aux_exp_data = {'size': str(m), 'structure': structure, 'exp_no': k, 'seed': v}    

    while not valid:
        compatability_matrix = generate_erdos_renyi_compatability_matrix_large(m, m)
        compatability_matrix = compatability_matrix.todense().A
        alpha = np.random.exponential(scale=1, size=m) # obtain values for non normelized customer frequency       
        # beta = np.random.exponential(scale=1, size=m) # obtain values for non normelized server frecquency
        beta = np.ones(n)
        alpha = alpha/alpha.sum()
        beta = beta/beta.sum()
        valid = False
        valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)

    if valid:
        print(k, str(m) + 'x' + str(m), 'p_edge: ', p_edge)
        print('-'*75)
        aux_exp_data['exp_no'] = k

    timestamp = dt.datetime.now()

    s = np.ones(m)
    c = np.zeros((m, n))
    nnz = compatability_matrix.nonzero()
    no_of_edges = len(nnz[0])    
    exact = n <= 10

    for rho in [0.6, 0.7, 0.8, 0.9] + [.95, .99, 1] + [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

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

    def log_ot_data(res, c, w , q, gamma, policy, rho, c_typ, c_min, c_max):

        res['mat']['c'] = c
        res['mat']['w'] = w
        res['mat']['q'] = q 
        res['aux']['gamma'] = gamma
        res['aux']['policy'] = policy
        res['aux']['rho'] = rho
        res['aux']['c_type'] = c_type
        res['aux']['c_min'] = c_min
        res['aux']['c_max'] = c_max

        return res

    for c_type in ['dist', 'rand']:

        for rho in [.6, .8, .9, .95]:

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
            c_min = (c_pad * r_pad).sum()
            r_pad = sinkhorn_stabilized(-1*c_pad, lamda_pad, mu, compatability_matrix_pad, 0.01)
            c_max = (c_pad * r_pad).sum()
            c_diff = c_max - c_min

            r_fcfs = entropy_approximation(compatability_matrix, lamda, mu, pad=True)
            q_fcfs = r_fcfs * (1./(mu - r_fcfs.sum(axis=0)))
            q_fcfs = q_fcfs/q_fcfs.sum(axis=0)

            min_ent = (-1 * lamda * np.log(lamda)).sum()
            max_ent = (-1 * r_fcfs * np.log(r_fcfs, out=np.zeros_like(r_fcfs), where=(r_fcfs != 0))).sum()

            ent_diff = max_ent - min_ent
            c = (ent_diff/c_diff) * c
            c_max = (ent_diff/c_diff)*c_max
            c_min = (ent_diff/c_diff)*c_min

            w_greedy = np.divide(np.ones(c.shape), c, out=np.zeros_like(c), where=(c != 0))
            sim_res_greedy = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_greedy, w_only=True,  prt_all=True, prt=True)
            sim_res_greedy = log_ot_data(sim_res_greedy, c, c , 0 * compatability_matrix, 1, 'greedy', rho, c_type, c_min, c_max)

            df_greedy = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_greedy, timestamp, aux_exp_data)
            write_df_to_file(ot_filename, df_greedy)
            
            for gamma in [0.1 * i for i in range(1, 10, 1)]:

                print('gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])

                r_fcfs_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=False)
                if r_fcfs_ot is None:
                    print('failed')
                r_fcfs_weighted_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=True)
                if r_fcfs_weighted_ot is None:
                    print('failed2')

                if r_fcfs_ot is not None and r_fcfs_weighted_ot is not None:

                    r_fcfs_ot = r_fcfs_ot[:m, :]
                    q_fcfs_ot = r_fcfs_ot * (1./mu - r_fcfs_ot.sum(axis=0))
                    q_fcfs_ot = q_fcfs_ot/q_fcfs_ot.sum(axis=0)
                    w_fcfs_ot = np.divide(q_fcfs_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_ot = log_ot_data(sim_res_fcfs_alis_ot, c, w_fcfs_ot , q_fcfs_ot, gamma, 'fcfs_alis_ot', rho, c_type, c_min, c_max)
                    df_fcfs_alis_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename, df_fcfs_alis_ot)

                    r_fcfs_weighted_ot = r_fcfs_weighted_ot[:m, :]
                    q_fcfs_weighted_ot = r_fcfs_weighted_ot * (1./mu - r_fcfs_weighted_ot.sum(axis=0))
                    q_fcfs_weighted_ot = q_fcfs_weighted_ot/q_fcfs_weighted_ot.sum(axis=0)
                    w_fcfs_weighted_ot  = np.divide(q_fcfs_weighted_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_weighted_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_weighted_ot = log_ot_data(sim_res_fcfs_alis_weighted_ot, c, w_fcfs_weighted_ot,  q_fcfs_weighted_ot, gamma, 'weighted_fcfs_alis_ot', rho, c_type, c_min, c_max)
                    df_fcfs_alis_weighted_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_weighted_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename, df_fcfs_alis_weighted_ot)
    
    gc.collect()

    return None
   


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    erdos_renyi_exp_for_parallel(30)