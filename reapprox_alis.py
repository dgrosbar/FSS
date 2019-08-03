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

import gc

def go_back_and_approximate_alis(filename, p=30):

    main_df = pd.read_csv(filename + '.csv')
    main_df = main_df[(main_df['gamma'] == 0) & (main_df['policy'] == 'fcfs_alis')]
    pool = mp.Pool(processes=p)
    exps = []
    newfilename = filename + '_alis'
    z = 1
    for (size, exp_no), exp in main_df.groupby(by=['size','exp_no'], as_index=False):

        
        exps.append([exp, size, exp_no, newfilename, z])
        z = z+1
    if p > 1:
        exps_res = pool.starmap(approximate_alis_exp, exps)
    else:
        approximate_alis_exp(*exps[0])

            
def approximate_alis_exp(exp_df, size, exp_no, newfilename, z):

    alis_rho_dfs = []

    for rho, rho_exp_df in exp_df.groupby(by='rho'):

        # print(rho_exp_df)

        exp_data = rho_exp_df[['m', 'n']].drop_duplicates()
        alpha_data = rho_exp_df[['i', 'alpha']].drop_duplicates()
        beta_data = rho_exp_df[['j', 'beta']].drop_duplicates()
        m = int(exp_data['m'].iloc[0])
        n = int(exp_data['n'].iloc[0])
        print(m, n)
        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m,n))
        
        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in rho_exp_df.iterrows():
            compatability_matrix[int(row['i']), int(row['j'])] = 1.
        
        # print(alpha.sum())
        # print(beta.sum())
        # print(compatability_matrix.sum())
        print('z: ',z, ' size: ',size, ' exp_no: ',exp_no, 'rho: ',rho)
        alis_approx = fast_sparse_alis_approximation(compatability_matrix, alpha, beta, rho, check_every=10, max_time=6000)

        res_dict = {'mat': {'alis_approx': alis_approx}, 'aux': {'exp_no': exp_no, 'size': size, 'rho': rho}}
        alis_rho_df = log_res_to_df(compatability_matrix, result_dict=res_dict)
        # print(alis_rho_df)
        alis_rho_dfs.append(alis_rho_df)

    alis_exp_df = pd.concat(alis_rho_dfs, axis=0)
    # print(alis_exp_df)
    exp_df = exp_df.drop(columns=['alis_approx'])
    exp_df = pd.merge(left=exp_df, right=alis_exp_df[['exp_no', 'size', 'rho', 'i', 'j', 'alis_approx']], on=['exp_no', 'size', 'rho', 'i', 'j'], how='left')
    exp_df.loc[:, 'fcfs_alis_approx'] = exp_df['rho'] * exp_df['fcfs_approx'] + (1. - exp_df['rho']) * exp_df['alis_approx']
    # print(exp_df)
    write_df_to_file(newfilename, exp_df)

if __name__ == '__main__':


    go_back_and_approximate_alis('erdos_renyi_sbpss_uni_mu_comp')














