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


def compare_w_policy(newfilename, filename='FZ_final_w_qp', p=30, lqf=True):

    df = pd.read_csv(filename + '.csv')
    pool = mp.Pool(processes=p)

    for n in range(7,11,1):
        exps = []
        for timestamp, exp in df[df['n'] == n].groupby(by=['timestamp'], as_index=False):
            exps.append([exp, timestamp, newfilename, lqf])
            if len(exps) == p:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                print(len(exps[0]))
                sbpss_dfs = pool.starmap(w_spbss, exps)
                exps = []
        else:
            if len(exps) > 0:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(w_spbss, exps)
                exps = []

        
def w_spbss(exp, timestamp, filename, lqf=True):

    exp_data = exp[['timestamp','m', 'n' ,'exp_num', 'density_level', 'beta_dist', 'graph_no']].drop_duplicates()
    alpha_data = exp[['i', 'alpha']].drop_duplicates()
    beta_data = exp[['j', 'beta']].drop_duplicates()

    policy_name = 'lqf_alis' if lqf else 'fcfs_alis' 
    
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

    c = np.zeros((m,n))

    for split in ['zero', 'one', 'half', 'rand']:

        if split == 'zero':
            theta = np.zeros(m)
        elif split == 'one':
            theta = np.ones(m)
        elif split == 'half':
            theta = 0.5 * np.ones(m)
        else:
            theta = np.random.uniform(0.1, 0.9, m)

        for rho in [.99, 0.6, 0.7, 0.8, 0.9, .95]:

            st = time()
            eta = alpha * rho
            lamda = eta**(1.- theta)
            s = eta**theta        
            mu = beta

            try:
                fcfs_eta_approx = fast_entropy_approximation(compatability_matrix, eta, mu, pad=True)
                fcfs_eta_approx = fcfs_eta_approx[:m]
            except:
                fcfs_eta_approx = -1 * np.ones((m,n))

            r = np.dot(np.diag(1./s), fcfs_eta_approx)
            alis_approx = fast_alis_approximation(1. * compatability_matrix, alpha, beta, rho)
            q = r * (1./(mu - r.sum(axis=0)))
            q = q/q.sum(axis=0)
            
            # try:
            r_weighted, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, 0, weighted=True)
            r_weighted = r_weighted[:m, :]

            q_weighted = r_weighted * (1./(mu - r_weighted.sum(axis=0)))
            q_weighted = q_weighted/q_weighted.sum(axis=0)
            w  = np.divide(q_weighted, q, out=np.zeros_like(q), where=(q != 0))

            exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, s, prt=True, lqf=lqf, per_edge=5000)
            exp_res['mat']['fcfs_approx'] = r
            exp_res['mat']['alis_approx'] = alis_approx 
            exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * exp_res['mat']['alis_approx'] + (rho) * exp_res['mat']['fcfs_approx']
            exp_res['aux']['rho'] = rho
            exp_res['aux']['gamma'] = 0
            exp_res['aux']['policy'] = policy_name
            exp_res['aux']['split'] = split
            exp_res['aux']['w'] = 1.
            exp_res['aux']['exp_no'] = exp_no
            exp_res['aux']['grpah_no'] = graph_no
            exp_res['aux']['beta_dist'] = beta_dist
            exp_res['aux']['density_level'] = density_level
            exp_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, exp_res, timestamp)
            write_df_to_file(filename, exp_df)

            print('policy: ', policy_name, 'lqf: ', lqf, ' split: ', split, ' graph_no: ', graph_no, ' exp_no: ', exp_no, ' rho: ', rho)

            w_exp_res = simulate_queueing_system(compatability_matrix, lamda, mu, s, w, prt=True, lqf=lqf, per_edge=5000)
            w_exp_res['mat']['fcfs_approx'] = r_weighted
            w_exp_res['mat']['alis_approx'] = alis_approx 
            w_exp_res['mat']['fcfs_alis_approx'] = (1. - rho) * w_exp_res['mat']['alis_approx'] + (rho) * w_exp_res['mat']['fcfs_approx']
            w_exp_res['aux']['rho'] = rho
            w_exp_res['aux']['gamma'] = 0
            w_exp_res['aux']['policy'] = 'weighted_' + policy_name
            w_exp_res['aux']['split'] = split
            w_exp_res['mat']['w'] = w
            w_exp_res['aux']['exp_no'] = exp_no
            w_exp_res['aux']['grpah_no'] = graph_no
            w_exp_res['aux']['beta_dist'] = beta_dist
            w_exp_res['aux']['density_level'] = density_level
            w_exp_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, w_exp_res, timestamp)
            write_df_to_file(filename, w_exp_df)
            print('policy: ', 'weighted_' + policy_name, 'lqf: ', lqf, ' split: ', split, ' graph_no: ', graph_no,' exp_no: ', exp_no, ' rho: ', rho)
            # except:
            #     print('could not solve for policy: ', 'weighted_' + policy_name, 'lqf: ', lqf, ' split: ', split, ' graph_no: ', graph_no,' exp_no: ', exp_no, ' rho: ', rho )
            gc.collect()

        return None
   

if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    compare_w_policy(newfilename='sbpss_w_compare_lqf', filename='FZ_final_w_qp', p=30, lqf=True)

    