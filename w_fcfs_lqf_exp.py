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


        
def w_spbss(exp, timestamp, policy):

    exp_data = exp[['timestamp','m', 'n' ,'exp_num', 'density_level', 'beta_dist', 'graph_no']].drop_duplicates()
    alpha_data = exp[['i', 'alpha']].drop_duplicates()
    beta_data = exp[['j', 'beta']].drop_duplicates()
    
    m = exp_data['m'].iloc[0]
    n = exp_data['n'].iloc[0]
    exp_no = exp_data['exp_num'].iloc[0]
    graph_no = exp_data['graph_no'].iloc[0]
    density_level = exp_data['density_level'].iloc[0]
    beta_dist = exp_data['beta_dist'].iloc[0]

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

    edge_count = compatability_matrix.sum()

    for rho in [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, .95, .99]:

        st = time()

        lamda = alpha * rho
        mu = beta
        pad_lamda = np.append(alpha*rho, 1. - rho)

        fcfs_approx = fast_entropy_approximation(compatability_matrix, lamda, mu, pad=(rho < 1))
        if rho < 1:
            fcfs_approx = fcfs_approx[:m]
        try:
            alis_approx = fast_alis_approximation(1. * compatability_matrix, alpha, beta, rho)
        except:
            alis_approx = np.zeros((m, n))
        
        q = fcfs_approx * (1./(mu - fcfs_approx.sum(axis=0)))
        q = q/q.sum(axis=0)

        r_weighted, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, 0, weighted=True)
        
        r_weighted = r_weighted[:m, :]
        q_weighted = r_weighted * (1./(mu - r_weighted.sum(axis=0)))
        q_weighted = q_weighted/q_weighted.sum(axis=0)
        w  = np.divide(q_weighted, q_fcfs, out=np.zeros_like(q), where=(q != 0))

        w_exp = simulate_queueing_system(compatability_matrix, lamda, mu, s, w, prt=True)
        w_exp_res['mat']['fcfs_approx'] = r_weighted
        w_exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
        w_exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * w_exp_res_fcfs['mat']['alis_approx'] + (rho) * w_exp_res_fcfs['mat']['fcfs_approx']
        w_exp_res['aux']['rho'] = rho
        w_exp_res['aux']['gamma'] = 0
        w_exp_res['aux']['policy'] = 'weighted_fcfs_alis'
        w_exp_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, w, timestamp, aux_exp_data)
        write_df_to_file(filename, w_exp_df)

        exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, s, prt=True)
        exp_res['mat']['fcfs_approx'] = fcfs_approx
        exp_res['mat']['alis_approx'] = alis_approx if alis_approx is not None else np.zeros((m, n))
        exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * exp_res_fcfs['mat']['alis_approx'] + (rho) * exp_res_fcfs['mat']['fcfs_approx']
        exp_res['aux']['rho'] = rho
        exp_res['aux']['gamma'] = 0
        exp_res['aux']['policy'] = 'fcfs_alis'

        exp_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, exp_res_fcfs, timestamp, aux_exp_data)
        write_df_to_file(filename, exp_df)

        print('ending - structure: ', aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'], ' rho: ', rho, ' duration: ', time() - st)
        print('pct_error_fcfs_alis_approx:'  , np.abs(exp_res_fcfs['mat']['sim_matching_rates'] - exp_res_fcfs['mat']['fcfs_alis_approx']).sum()/lamda.sum())
        
        gc.collect()

    return None
   

if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    