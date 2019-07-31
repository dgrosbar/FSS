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


def create_maps(m, d, zeta, max_max_wls, min_count_wls, max_max_rho):

    k = 0
    size = str(int(m)) +'x' + str(int(m))
    while k < 30:

        try:
            compatability_matrix, g, alpha, beta, workload_sets, rho_m, rho_n, node_map = generate_grid_compatability_matrix_with_map(m, d , zeta)
            c_i = -1* np.ones(m**2)
            c_j = -1* np.ones(m**2)
            wq_i = -1* np.ones(m**2)
            wq_j = -1* np.ones(m**2)        
            max_wls = 0
            max_rho = workload_sets[0]['rho']
            count_wls = len(workload_sets)
            for workload_set in workload_sets.values():
                # for key, val in workload_set.items():
                #     print(key, val)
                c = len(workload_set['supply_nodes'])
                max_wls = max(max_wls, len(workload_set['demand_nodes']))
                for node in workload_set['demand_nodes']:
                    c_i[node] = c
                for node in workload_set['supply_nodes']:
                    c_j[node] = c
                
            print('max_wls: ', max_wls, 'count_wls: ', count_wls, 'max_rho: ', max_rho)

            if max_wls <= max_max_wls and count_wls > min_count_wls and max_rho < max_max_rho:

                k +=1
                print(k , ' SBPSSs found')
                node_map = pd.DataFrame(data=node_map, columns=['node','x','y'])
                res_dict = {'col': dict(), 'row': dict(), 'mat': dict(), 'aux': dict()}
                res_dict['col']['eta_j_MMF'] = rho_n
                res_dict['row']['eta_i_MMF'] = rho_m
                res_dict['row']['c_i_MMF'] = c_i
                res_dict['col']['c_j_MMF'] = c_j
                res_dict['aux']['exp_no'] = k
                res_dict['aux']['arc_dist'] = d
                res_dict['aux']['zeta'] = zeta
                res_dict['aux']['structure'] = 'grid'
                res_dict['aux']['size'] = size

                map_df = log_res_to_df(compatability_matrix, alpha=alpha, beta=alpha, result_dict=res_dict)
                map_df = pd.merge(left=map_df, right=node_map.rename(columns={'x': 'x_i', 'y': 'y_i', 'node': 'i'}), on='i')
                map_df = pd.merge(left=map_df, right=node_map.rename(columns={'x': 'x_j', 'y': 'y_j', 'node': 'j'}), on='j')
                write_df_to_file('map_exps_' + size, map_df)
        except:
            continue


def map_exp_for_parallel(p=30, filename='map_exps_30x30'):

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


def sbpss_exp(timestamp, exp, filename='map_exp_sbpss', ot_filename='map_exp_sbpss_ot'):

    exp_data = exp[['m', 'n', 'exp_no', 'arc_dist', 'size']].drop_duplicates()
    alpha_data = exp[['i', 'alpha', 'eta_i_MMF']].drop_duplicates()
    beta_data = exp[['j', 'beta', 'eta_j_MMF']].drop_duplicates()
    m = exp_data['m'].iloc[0]
    n = exp_data['n'].iloc[0]
    d = exp_data['arc_dist'].iloc[0]
    exp_no = exp_data['exp_no'].iloc[0]
    size = exp_data['size'].iloc[0]
    alpha = np.zeros(m)
    beta = np.zeros(n)
    eta_i = np.zeros(m)
    eta_j = np.zeros(n)
    compatability_matrix = np.zeros((m,n))
    
    rho_mmf_max = exp['eta_j_MMF'].max() 

    for k, row in alpha_data.iterrows():
        alpha[int(row['i'])] = float(row['alpha'])

    for k, row in beta_data.iterrows():
        beta[int(row['j'])] = float(row['beta'])

    for k, row in alpha_data.iterrows():
        eta_i[int(row['i'])] = float(row['eta_i_MMF'])

    for k, row in beta_data.iterrows():
        eta_j[int(row['j'])] = float(row['eta_j_MMF'])

    for k, row in exp.iterrows():
        compatability_matrix[int(row['i']), int(row['j'])] = 1.



    valid = False
    v = np.random.randint(1, 10**6)
    np.random.seed(v)
    aux_exp_data = {'size': '10x10', 'arc_dist': d, 'structure': '', 'exp_no': k, 'seed': v}    
    s = np.ones(m)
    c = np.zeros((m, n))
    nnz = compatability_matrix.nonzero()
    pad_compatability_matrix = np.vstack([compatability_matrix, np.ones(n)])
    no_of_edges = len(nnz[0])    
    exact = n <= 10
    org_compatability_matrix = compatability_matrix

    print(exp_no, ': all data read in')

    for rho in [0.6, 0.7, 0.8, 0.9] + [.95, .99] + [0.01, 0.05, 0.1, 0.2, 0.4]:

        st = time()
        print('alpha.sum(): ',alpha.sum())
        print('beta.sum(): ',beta.sum())
        lamda = alpha * (rho/rho_mmf_max) 
        mu = beta
        rho_j = (rho/rho_mmf_max) * eta_j
        print('lamda.sum(): ',lamda.sum())
        print('mu.sum(): ',mu.sum())

        pad_lamda = np.append(lamda, 1. - lamda.sum())
        print('lamda_pad.sum()', pad_lamda.sum())
        exp.loc[:, 'rho_i_MMF'] = (rho / rho_mmf_max) * exp['eta_i_MMF']
        exp.loc[:, 'rho_j_MMF'] = (rho / rho_mmf_max) * exp['eta_j_MMF']
        exp.loc[: ,'Wq_i_MMF'] = exp['rho_i_MMF']**(((2.0*(exp['c_i_MMF']+1.0))**0.5)-1)/(exp['c_i_MMF']*(1-exp['rho_i_MMF']))
        exp.loc[: ,'Wq_j_MMF'] = exp['rho_j_MMF']**(((2.0*(exp['c_j_MMF']+1.0))**0.5)-1)/(exp['c_j_MMF']*(1-exp['rho_j_MMF']))
        exp.loc[: ,'kill'] = exp['Wq_i_MMF'] < exp['Wq_j_MMF']

        for arc_kill in [False]:
            
            if arc_kill:
                print(compatability_matrix.sum(axis=1))
                print(compatability_matrix.sum(axis=0))
                for k, row in exp.iterrows():
                    if row['kill']:
                        compatability_matrix[int(row['i']), int(row['j'])] = 0
                print(compatability_matrix.sum(axis=1))
                print(compatability_matrix.sum(axis=0))
            else:
                compatability_matrix = org_compatability_matrix

            # compatability_matrix = sps.csr_matrix(compatability_matrix)

            print(exp_no, rho, ': starting fast_entropy_approximation')
            fcfs_approx = fast_entropy_approximation(compatability_matrix, lamda, mu, pad=(rho < 1))
            
            print(exp_no, rho, ': ending fast_entropy_approximation')
            if rho < 1:
                fcfs_approx = fcfs_approx[:m]
            try:
                print(exp_no, rho, ': starting fast_alis_approximation')
                alis_approx = fast_alis_approximation(1. * compatability_matrix, alpha, beta, rho) if m < 100 else np.zeros((m, n))
                print(exp_no, rho, ': ending fast_alis_approximation')
            except:
                alis_approx = np.zeros((m, n))
            
            q_fcfs = fcfs_approx * (1./mu - fcfs_approx.sum(axis=0))
            q_fcfs = q_fcfs/q_fcfs.sum(axis=0)

            if rho >= 0.6 and rho < 1:

                print(exp_no, ': starting weight_calcs')
                print(exp_no, rho, ': starting weighted_entropy_regulerized_ot')
                r_fcfs_weighted, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho_j, 0, weighted=True)
                print(exp_no, rho, ': ending weighted_entropy_regulerized_ot')
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
                    add_sbpss_df = exp[['i','j','rho_j_MMF', 'rho_j_MMF', 'c_i_MMF', 'c_j_MMF', 'Wq_i_MMF', 'Wq_j_MMF', 'kill']]
                    sbpss_df = pd.merge(left=sbpss_df, right=add_sbpss_df, on=['i', 'j'], how='left') 
                    write_df_to_file(filename + '_' + size, sbpss_df)
            
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
            exp_res['aux']['arc_kill'] = True

            sbpss_df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, exp_res, timestamp, aux_exp_data)
            write_df_to_file(filename + '_' + size, sbpss_df)

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
    compatability_matrix = org_compatability_matrix
    for c_type in ['dist']:

        for rho in [.4, .6, .8, .9, .95, .99]:

            lamda = alpha * (rho/rho_mmf_max) 
            mu = beta 
            s = np.ones(m)
            rho_j = (rho/rho_mmf_max) * eta_j

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

                r_fcfs_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho_j, gamma, weighted=False)
                if r_fcfs_ot is None:
                    print('failed unweighted ', 'gamma:', gamma, ' rho:', rho, ' c_type:', c_type, aux_exp_data['structure'], ' exp_no: ', aux_exp_data['exp_no'])
                r_fcfs_weighted_ot, _ = weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho_j, gamma, weighted=True)
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
                    write_df_to_file(ot_filename + '_' + size, df_fcfs_alis_ot)

                    r_fcfs_weighted_ot = r_fcfs_weighted_ot[:m, :]
                    q_fcfs_weighted_ot = r_fcfs_weighted_ot * (1./mu - r_fcfs_weighted_ot.sum(axis=0))
                    q_fcfs_weighted_ot = q_fcfs_weighted_ot/q_fcfs_weighted_ot.sum(axis=0)
                    w_fcfs_weighted_ot  = np.divide(q_fcfs_weighted_ot, q_fcfs, out=np.zeros_like(q_fcfs), where=(q_fcfs != 0))

                    sim_res_fcfs_alis_weighted_ot = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_fcfs_weighted_ot, prt_all=True, prt=True)
                    sim_res_fcfs_alis_weighted_ot = log_ot_data(sim_res_fcfs_alis_weighted_ot, c, w_fcfs_weighted_ot,  q_fcfs_weighted_ot, gamma, 'weighted_fcfs_alis_ot', rho, c_type,)
                    df_fcfs_alis_weighted_ot = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_fcfs_alis_weighted_ot, timestamp, aux_exp_data)
                    write_df_to_file(ot_filename + '_' + size, df_fcfs_alis_weighted_ot)

            w_greedy = np.divide(np.ones(c.shape), c, out=np.zeros_like(c), where=(c != 0))
            sim_res_greedy = simulate_queueing_system(compatability_matrix, lamda, mu, s, w_greedy, w_only=True,  prt_all=True, prt=True)
            sim_res_greedy = log_ot_data(sim_res_greedy, c, c , 0 * compatability_matrix, 1, 'greedy', rho, c_type)

            df_greedy = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, sim_res_greedy, timestamp, aux_exp_data)
            write_df_to_file(ot_filename + '_' + size, df_greedy)

    gc.collect()

    return None
   


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    # create_maps(30, 3, 0.2, 450, 5, 1.2)
    map_exp_for_parallel(30)