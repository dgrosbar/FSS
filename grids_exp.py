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


# def grids_exp_for_parallel(p=30):

#     jt_perm_dict = {9: list(jpermute(range(9)))}
#     print_progress = True

    
    
#     for structure in ['torus']:

#         for sqrt_m, d  in zip([30, 9, 3], [3, 2, 1]):

#             aux_exp_data = {'size': str(sqrt_m) + 'x' + str(sqrt_m), 'arc_dist': d, 'structure': structure}
#             exact = False
#             m = sqrt_m**2
#             k = 0
            
#             while k < 30:
            
#                 valid = False
#                 exps = []
#                 exps_ot = []
            
#                 while len(exps) < p:

#                     alpha = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized customer frequency       
#                     beta = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized server frecquency
#                     alpha = alpha/alpha.sum()
#                     beta = beta/beta.sum()
#                     valid = False
#                     compatability_matrix, g = generate_grid_compatability_matrix(sqrt_m, d)
#                     valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)
                    
#                     if valid:
#                         arc_dist=d
#                         print(k-1, str(sqrt_m) + 'x' + str(sqrt_m), 'd=', arc_dist)
#                         print('-'*75)
#                         aux_exp_data['exp_no'] = k
#                         exps.append([compatability_matrix, alpha, beta, dt.datetime.now(), copy(aux_exp_data), 'new_grid_sbpss2'])
#                         exps_ot.append([compatability_matrix, alpha, beta, dt.datetime.now(), copy(aux_exp_data), 'new_grid_sbpss_ot'])
#                         k += 1
                
#                 pool = mp.Pool(processes=p)
#                 exps_res = pool.starmap(sbpss_exp, exps)
#                 exps_ot_res = pool.starmap(spbss_ot_exp, exps_ot)
#                 pool.close()
#                 pool.join()
#                 # exps_res = sbpss_exp(*exps[0])
#                 # exps_ot_res = spbss_ot_exp(*exps_ot[0])


def grids_exp_for_parallel(p=30):

    jt_perm_dict = {9: list(jpermute(range(9)))}
    print_progress = True
    for structure in ['torus']:
        for sqrt_m, d  in zip([3, 9, 30], [1, 2, 3]):
            exps = [list(tup) for tup in zip([sqrt_m]*p, [d]*p, range(1, p+1, 1), [structure]*30)]
            pool = mp.Pool(processes=p)
            exps_res = pool.starmap(sbpss_exp, exps)

            pool.close()
            pool.join()


def sbpss_exp(sqrt_m, d, k, structure, filename='new_grid_sbpss3', ot_filename='new_grid_sbpss_ot3'):

    compatability_matrix, g = generate_grid_compatability_matrix(sqrt_m, d)
    m, n = compatability_matrix.shape

    valid = False
    np.random.seed(k)
    v = np.random.randint(1, 10**6)
    np.random.seed(v)
    aux_exp_data = {'size': str(sqrt_m) + 'x' + str(sqrt_m), 'arc_dist': d, 'structure': structure, 'exp_no': k, 'seed': v}    

    while not valid:
        alpha = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized customer frequency       
        beta = np.random.exponential(scale=1, size=sqrt_m**2) # obtain values for non normelized server frecquency
        alpha = alpha/alpha.sum()
        beta = beta/beta.sum()
        valid = False
        valid, _ = verify_crp_condition(compatability_matrix, alpha, beta)

    if valid:
        arc_dist=d
        print(k-1, str(sqrt_m) + 'x' + str(sqrt_m), 'd=', arc_dist)
        print('-'*75)
        aux_exp_data['exp_no'] = k

    timestamp = dt.datetime.now()

    s = np.ones(m)
    c = np.zeros((m, n))
    nnz = compatability_matrix.nonzero()
    pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
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

    def log_ot_data(res, c, w , q, gamma, policy, rho, c_type):

        res['mat']['c'] = c
        res['mat']['w'] = w
        res['mat']['q'] = q 
        res['aux']['gamma'] = gamma
        res['aux']['policy'] = policy
        res['aux']['rho'] = rho
        res['aux']['c_type'] = c_type

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

         
            w_greedy = np.divide(np.ones(c.shape), c, out=np.zeros_like(c), where=(c != 0))
            sim_res_greedy = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_greedy, w_only=True,  prt_all=True, prt=True)
            sim_res_greedy = log_ot_data(sim_res_greedy, c, c , 0 * compatability_matrix, 1, 'greedy', rho, c_type)

            df_greedy = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_greedy, timestamp, aux_exp_data)
            write_df_to_file(filename, df_greedy)
            
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
                    sim_res_fcfs_alis_ot = log_ot_data(sim_res_fcfs_alis_ot, c, w_fcfs_ot , q_fcfs_ot, gamma, 'fcfs_alis_ot', rho, c_type)
                    df_fcfs_alis_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_ot, timestamp, aux_exp_data)
                    write_df_to_file(filename, df_fcfs_alis_ot)

                    r_fcfs_weighted_ot = r_fcfs_weighted_ot[:m, :]
                    q_fcfs_weighted_ot = r_fcfs_weighted_ot * (1./mu - r_fcfs_weighted_ot.sum(axis=0))
                    q_fcfs_weighted_ot = q_fcfs_weighted_ot/q_fcfs_weighted_ot.sum(axis=0)
                    w_fcfs_weighted_ot  = np.divide(q_fcfs_weighted_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_weighted_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_weighted_ot = log_ot_data(sim_res_fcfs_alis_weighted_ot, c, w_fcfs_weighted_ot,  q_fcfs_weighted_ot, gamma, 'weighted_fcfs_alis_ot', rho, c_type,)
                    df_fcfs_alis_weighted_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_weighted_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename, df_fcfs_alis_weighted_ot)
    
    gc.collect()

    return None
   

# def sbpss_exp(compatability_matrix, alpha, beta, timestamp=None, aux_exp_data=None, filename='new_grid_sbpss'):


#     m, n = compatability_matrix.shape
#     s = np.ones(m)
#     c = np.zeros((m, n))
#     nnz = compatability_matrix.nonzero()
#     pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
#     no_of_edges = len(nnz[0])    
#     exact = n <= 10

#     timestamp = timestamp if timestamp is not None else dt.datetime.now()

#     res_df = []


#     for rho in [0.6, 0.7, 0.8, 0.9] + [.95, .99, 1] + [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

#         st = time()
#         lamda = alpha * rho
#         mu = beta
#         pad_lamda = np.append(alpha*rho, 1. - rho)

#         if rho == 1:
#             exp_res = simulate_matching_sequance(compatability_matrix, alpha, beta, prt_all=True, prt=True)
#         else:
#             exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, prt_all=True, prt=True)
        
#         fcfs_approx = fast_entropy_approximation(compatability_matrix, lamda, mu, pad=(rho < 1))
#         q_fcfs = fcfs_approx * (1./mu - fcfs_approx.sum(axis=0))
#         q_fcfs = q_fcfs/q_fcfs.sum(axis=0)
        
#         if rho < 1:
#             fcfs_approx = fcfs_approx[:m]
        
#         exp_res['mat']['fcfs_approx'] = fcfs_approx
#         alis_approx = fast_alis_approximation(1. * compatability_matrix, alpha, beta, rho)
#         exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
#         exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * exp_res['mat']['alis_approx'] + (rho) * exp_res['mat']['fcfs_approx']
        
#         exp_res['aux']['rho'] = rho
#         exp_res['aux']['gamma'] = 0
#         exp_res['aux']['policy'] = 'fcfs_alis'

#         sbpss_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, exp_res, timestamp, aux_exp_data)
#         write_df_to_file(filename, sbpss_df)

#         if rho >= 0.6 and rho < 1:
            
#             r_fcfs_weighted, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, 0, weighted=True)
#             if r_fcfs_weighted is not None:
#                 r_fcfs_weighted = r_fcfs_weighted[:m, :]
#                 q_fcfs_weighted = r_fcfs_weighted * (1./mu - r_fcfs_weighted.sum(axis=0))
#                 q_fcfs_weighted = q_fcfs_weighted/q_fcfs_weighted.sum(axis=0)
#                 w_fcfs_weighted  = np.divide(q_fcfs_weighted, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))
#                 w_exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted)

#                 w_exp_res['mat']['fcfs_approx'] = r_fcfs_weighted
#                 w_exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
#                 w_exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * w_exp_res['mat']['alis_approx'] + (rho) * w_exp_res['mat']['fcfs_approx']

#                 w_exp_res['aux']['rho'] = rho
#                 w_exp_res['aux']['gamma'] = 0
#                 w_exp_res['aux']['policy'] = 'weighted_fcfs_alis'

#         print('ending - structure: ', aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'], ' rho: ', rho, ' duration: ', time() - st)
#         print('pct_error_fcfs_alis_approx:'  , np.abs(exp_res['mat']['sim_matching_rates'] - exp_res['mat']['fcfs_alis_approx']).sum()/lamda.sum())
        
#         gc.collect()

#     return None
    

# def spbss_ot_exp(compatability_matrix, alpha, beta, timestamp=None, aux_exp_data=None, filename='new_grid_sbpss_ot'):


#     dfs = []
#     m, n = compatability_matrix.shape


#     def log_ot_data(res, c, w , q, gamma, policy, rho, c_type):

#         res['mat']['c'] = c
#         res['mat']['w'] = w
#         res['mat']['q'] = q 
#         res['aux']['gamma'] = gamma
#         res['aux']['policy'] = policy
#         res['aux']['rho'] = rho
#         res['aux']['c_type'] = c_type

#         return res

#     for c_type in ['dist', 'rand']:

#         for rho in [.6, .8, .9, .95]:

#             lamda = rho * alpha
#             mu = beta 
#             s = np.ones(m)

#             if c_type == 'rand':
#                 c = np.random.exponential(1, (m, n)) * compatability_matrix
#             else:
#                 c = np.zeros((m, n))
#                 for i in range(m):
#                     for j in range(n):
#                         c[i,j] = 1 + abs(i - j)
#                 c = c * compatability_matrix
#                 c = c/c.sum()

#             lamda_pad = np.append(lamda, mu.sum() - lamda.sum())
#             c_pad = np.vstack([c, np.zeros((1, n))])
#             compatability_matrix_pad = np.vstack([compatability_matrix, np.ones((1, n))])

#             r_pad = sinkhorn_stabilized(c_pad, lamda_pad, mu, compatability_matrix_pad, 0.01)
#             min_c = (c_pad * r_pad).sum()
#             r_pad = sinkhorn_stabilized(-1*c_pad, lamda_pad, mu, compatability_matrix_pad, 0.01)
#             max_c = (c_pad * r_pad).sum()
#             c_diff = max_c - min_c

#             r_fcfs = entropy_approximation(compatability_matrix, lamda, mu, pad=True)
#             q_fcfs = r_fcfs * (1./(mu - r_fcfs.sum(axis=0)))
#             q_fcfs = q_fcfs/q_fcfs.sum(axis=0)

#             min_ent = (-1 * lamda * np.log(lamda)).sum()
#             max_ent = (-1 * r_fcfs * np.log(r_fcfs, out=np.zeros_like(r_fcfs), where=(r_fcfs != 0))).sum()

#             ent_diff = max_ent - min_ent
#             c = (ent_diff/c_diff) * c

         
#             w_greedy = np.divide(np.ones(c.shape), c, out=np.zeros_like(c), where=(c != 0))
#             sim_res_greedy = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_greedy, w_only=True,  prt_all=True, prt=True)
#             sim_res_greedy = log_ot_data(sim_res_greedy, c, c , 0 * compatability_matrix, 1, 'greedy', rho, c_type)

#             df_greedy = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_greedy, timestamp, aux_exp_data)
#             write_df_to_file(filename, df_greedy)
            
#             for gamma in [0.1 * i for i in range(1, 10, 1)]:
#             # for gamma in [0.1] + [0.2 * i for i in range(1, 5, 1)]:

#                 print('gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])

#                 r_fcfs_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=False)
#                 if r_fcfs_ot is None:
#                     print('failed')
#                 r_fcfs_weighted_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=True)
#                 if r_fcfs_weighted_ot is None:
#                     print('failed2')

#                 if r_fcfs_ot is not None and r_fcfs_weighted_ot is not None:

#                     r_fcfs_ot = r_fcfs_ot[:m, :]
#                     q_fcfs_ot = r_fcfs_ot * (1./mu - r_fcfs_ot.sum(axis=0))
#                     q_fcfs_ot = q_fcfs_ot/q_fcfs_ot.sum(axis=0)
#                     w_fcfs_ot = np.divide(q_fcfs_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

#                     sim_res_fcfs_alis_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_ot, prt_all=True, prt=True)
#                     sim_res_fcfs_alis_ot = log_ot_data(sim_res_fcfs_alis_ot, c, w_fcfs_ot , q_fcfs_ot, gamma, 'fcfs_alis_ot', rho, c_type)
#                     df_fcfs_alis_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_ot, timestamp, aux_exp_data)
#                     write_df_to_file(filename, df_fcfs_alis_ot)

#                     r_fcfs_weighted_ot = r_fcfs_weighted_ot[:m, :]
#                     q_fcfs_weighted_ot = r_fcfs_weighted_ot * (1./mu - r_fcfs_weighted_ot.sum(axis=0))
#                     q_fcfs_weighted_ot = q_fcfs_weighted_ot/q_fcfs_weighted_ot.sum(axis=0)
#                     w_fcfs_weighted_ot  = np.divide(q_fcfs_weighted_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

#                     sim_res_fcfs_alis_weighted_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted_ot, prt_all=True, prt=True)
#                     sim_res_fcfs_alis_weighted_ot = log_ot_data(sim_res_fcfs_alis_weighted_ot, c, w_fcfs_weighted_ot,  q_fcfs_weighted_ot, gamma, 'weighted_fcfs_alis_ot', rho, c_type,)
#                     df_fcfs_alis_weighted_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_weighted_ot, timestamp, aux_exp_data)
#                     write_df_to_file(filename, df_fcfs_alis_weighted_ot)
    
#     gc.collect()
#     return None


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    grids_exp_for_parallel(30)