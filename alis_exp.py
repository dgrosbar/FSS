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
from utilities import *
from mr_calc_and_approx import *
import gc


def go_back_and_approximate_alis(filename='FZ_final_w_qp', p=30):

	try:
		df = pd.read_csv(filename + '.csv')
	except:
		df = pd.read_csv('./Results/' + filename + '.csv')
	pool = mp.Pool(processes=p)

	for n in range(7,11,1):
		exps = []
		for timestamp, exp in df[df['n'] == n].groupby(by=['timestamp'], as_index=False):
			exps.append([exp, timestamp])
			if len(exps) == p:
				print('no_of_exps:', len(exps), 'n:', n)
				print('starting work with {} cpus'.format(p))
				if p > 1:
					sbpss_dfs = pool.starmap(approximate_sbpss_pure_alis, exps)
				else:
					approximate_sbpss_pure_alis(*exps[0])
				
				exps = []
		else:
			if len(exps) > 0:
				print('no_of_exps:', len(exps), 'n:', n)
				print('starting work with {} cpus'.format(p))
				sbpss_dfs = pool.starmap(approximate_sbpss_pure_alis, exps)
				exps = [] 

def approximate_sbpss_pure_alis(exp, timestamp):

	st = time()
	exp_data = exp[['m', 'n', 'density_level', 'graph_no', 'exp_num', 'beta_dist']].drop_duplicates()
	alpha_data = exp[['i', 'alpha']].drop_duplicates()
	beta_data = exp[['j', 'beta']].drop_duplicates()
	
	m = exp_data['m'].iloc[0]
	n = exp_data['n'].iloc[0]
	graph_no = exp_data['graph_no'].iloc[0]
	exp_no = exp_data['exp_num'].iloc[0]
	beta_dist = exp_data['beta_dist'].iloc[0]
	density_level = exp_data['density_level'].iloc[0]

	print('density_level:', density_level, 'graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist)

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

	no_of_edges = len(nnz[0])

	alis_df = []

	exp_res = simulate_queueing_system(compatability_matrix, alpha, beta, sims=30, per_edge=10000, alis=True)

	alis_approx = fast_sparse_alis_approximation(compatability_matrix, alpha, beta, 0, check_every=10, max_time=600)
	exp_res['mat']['alis_approx'] = alis_approx
	
	print('ending - density_level: ', density_level, ' graph_no: ', graph_no, ' exp_no: ', exp_no, ' beta_dist: ', beta_dist, ' rho: ', 1, ' duration: ', time() - st, 'pct_error', 
		np.abs(exp_res['mat']['sim_matching_rates'] - exp_res['mat']['alis_approx']).sum())

	exp_res['aux']['graph_no'] = graph_no
	exp_res['aux']['exp_no'] = exp_no
	exp_res['aux']['beta_dist'] = beta_dist
	exp_res['aux']['density_level'] = density_level
	exp_res['aux']['rho'] = 0
	exp_res['aux']['policy'] = 'alis'
	# exp_res['mat']['alis_sim_rates'] = exp_res2['mat']['sim_matching_rates']
	# exp_res['aux']['sim_adj'] = sim_adj
	# exp_res['aux']['sim_rate_gap'] = sim_rate_gap
	alis_exp_df = log_res_to_df(compatability_matrix, alpha, beta, alpha, np.zeros(m), beta, result_dict=exp_res, timestamp=timestamp)
	# print(alis_exp_df[['i', 'j', 'alpha', 'beta', 'sim_matching_rates', 'alis_approx', 'alis_sim_rates']])
	write_df_to_file('FZ_Kaplan_exp_pure_alis', alis_exp_df)

	return None


if __name__ == '__main__':


	go_back_and_approximate_alis()



