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

def go_back_and_do_matching_sequence(filename, newfilename, p):

	df = pd.read_csv(filename + '.csv')
	df = df[df['rho'] == 0.1]
	exps = []
	for timestamp, exp in df.groupby(by='timestamp'):

		exp = exp[['i','j','alpha','beta', 'exp_no','size', 'm','n']]
		exps.append([exp, timestamp, newfilename])

	pool =mp.Pool(processes=p)

	dfs = pool.starmap(simulate_and_approximate_matching_sequence, exps)


def simulate_and_approximate_matching_sequence(exp, timestamp, newfilename):

		exp_data = exp[['m', 'n', 'exp_no','size']].drop_duplicates()
		alpha_data = exp[['i', 'alpha']].drop_duplicates()
		beta_data = exp[['j', 'beta']].drop_duplicates()
		m = int(exp_data['m'].iloc[0])
		n = int(exp_data['n'].iloc[0])
		exp_no = exp_data['exp_no'].iloc[0]
		size = exp_data['size'].iloc[0]
		alpha = np.zeros(m)
		beta = np.zeros(n)
		compatability_matrix = np.zeros((m,n))

		for k, row in alpha_data.iterrows():
			alpha[int(row['i'])] = float(row['alpha'])

		for k, row in beta_data.iterrows():
			beta[int(row['j'])] = float(row['beta'])

		for k, row in exp.iterrows():
			compatability_matrix[int(row['i']), int(row['j'])] = 1.
		
		res_dict = simulate_matching_sequance(compatability_matrix, alpha, beta, prt_all=True)
		res_df = log_res_to_df(compatability_matrix, alpha=alpha, beta=beta, result_dict=res_dict, add_aux_data={'exp_no': exp_no, 'size': size})

		write_df_to_file(newfilename, res_df)

		return True


if __name__ == '__main__':

	try:
		print('erdos_renyi_sbpss')
		go_back_and_do_matching_sequence('erdos_renyi_sbpss', 'erdos_renyi_sbpss_ms', p=30)
	except:
		print('new_grid_sbpss3')
		go_back_and_do_matching_sequence('new_grid_sbpss3', 'new_grid_sbpss3_ms', p=30)		

