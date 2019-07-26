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

def arc_softnes_exp(n, ul, uh):

	compatability_matrix = np.hstack([np.zeros((n-1, 1)), np.ones((n-1, n-1))])
	compatability_matrix = np.vstack([np.array([1,1] + [0]*(n-2)), compatability_matrix])

	mu = np.ones(n)
	lamda = np.array([ul] + [uh]*(n-1))
	beta = mu/mu.sum()
	alpha = lamda/lamda.sum()
	s = np.ones(n)
	sim_res = simulate_queueing_system(compatability_matrix, lamda, mu, s)
	sim_res['aux']['ul'] = ul
	sim_res['aux']['uh'] = uh
	sim_res['row']['i_type'] = np.array([1] +[0]*(n-1))
	sim_res['col']['j_type'] = np.array([1] +[0]*(n-1))
	df = log_res_to_df(compatability_matrix, alpha, beta, lamda, s, mu, result_dict=sim_res, timestamp=dt.datetime.now())
	df_i = df[['i_type','ul','uh','n', 'sim_waiting_times']].drop_duplicates()
	df_i = df_i.groupby(by=['i_type','ul','uh','n'], as_index=False).mean()
	return df_i

	

if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    df1 = pd.concat([arc_softnes_exp(n, 0.8, 0.9) for n in [2,3,5,10,15,20,30,50,100]], axis=0)
    df2 = pd.concat([arc_softnes_exp(n, 0.8, 0.95) for n in [2,3,5,10,15,20,30,50,100]], axis=0)
    df3 = pd.concat([arc_softnes_exp(n, 0.8, 0.99) for n in [2,3,5,10,15,20,30,50,100]], axis=0)
    df = pd.concat([df1, df2, df3], axis=0)
    write_to_file('arc_softness2.csv', df)
    print(df.pivot(index='n', columns='i_type', values='sim_waiting_times'))

   	