import numpy as np
from numba import jit
import cplex
from cplex.exceptions import CplexError
from math import exp
from time import time
from numpy import ma
from utilities import printarr
from scipy import sparse as sps
import sys
import subprocess

# from gurobipy import *


def adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta, jt_perms=None, print_progress=False, pad=False):


	# Modified Adan_Weiss Matching Rate Fromula:
	#                           __                                                                                        __
	#                          |                J                                                                           |
	#                          |    ______    ______                                                                        |
	#                          |    \         \          k                                  -1     J-1                -1    |              
	#                          |     \         \       -----                /               \     -----  /           \      | 
	#  r_ij = B x a_i x b_j x  |     /         /        | | I(c in u(l)) x | b(l) - a(l)X(l) |  x  | |  | b(l) - a(l) |     | 
	#                          |    /         /         l=1                 \               /      l=k   \           /      |
	#                          |   /______   /______                                                                        |
	#                          |    p in P     k=1                                                                          |
	#                          |__                                                                                        __|

	if pad:
		compatability_matrix = np.vstack([compatability_matrix, np.ones(len(beta))])
		alpha = np.append(alpha, beta.sum() - alpha.sum())


	start_time = time()

	m = len(alpha) # m customer classes 
	n = len(beta) # n server classes (n = J)

	s_adj = dict((i, set(np.nonzero(compatability_matrix[:, i])[0])) for i in range(n))
	c_adj = dict((j, set(np.nonzero(compatability_matrix[j, :])[0])) for j in range(m))

	
	# 2^n x n of all n length words over [0,1] representing server subsets
	server_combination_matrix = np.array(np.meshgrid(*[[0,1]] * n)).T.reshape(-1, n)[1:, :]
	
	# 2^n x 1 matrix of the sum of server capacities for every subset of servers
	beta_l = np.dot(server_combination_matrix, beta)
	# 2^n x m of the customers only served by the servers of the subset per subset
	phi_x_alpha = 1 - 1 * (np.dot(1 - server_combination_matrix, compatability_matrix.T) > 0) # 2^n x m matrix
	# 2^n x 1 matrix of the sum of arrival rates of uncompatability_matrixiuely served customer classes 
	alpha_l = np.dot(phi_x_alpha, alpha) # 2^n x 1 matrix
	# number of subsets
	subset_count, _ = server_combination_matrix.shape
	
	beta_alpha_arr = beta_l - alpha_l

	inv_beta_alpha = dict(
		(
			frozenset(server_combination_matrix[subset_num, :].nonzero()[0]),
			1./beta_alpha_arr[subset_num]
		)
		for subset_num in range(subset_count-1))

	inv_beta_alpha[frozenset(range(n))] = 1.

	beta_alpha = dict(
		(
			frozenset(server_combination_matrix[subset_num, :].nonzero()[0]),
			beta_alpha_arr[subset_num]
		)
		for subset_num in range(subset_count))

	beta_alpha[frozenset([])] = 1.
	
	# ---------------------------------------------------------------------------------------------------------------------------------------------------
	
	inv_beta_alpha_x_xhi_arr = np.zeros((subset_count, n))

	for j in range(n):

		inv_beta_alpha_x_xhi_arr[:, j] = 1./(beta_l - alpha_l + np.dot(compatability_matrix.T[j] * phi_x_alpha, alpha))

	inv_beta_alpha_x_xhi = dict(
		(
			frozenset(server_combination_matrix[subset_num, :].nonzero()[0]),
			inv_beta_alpha_x_xhi_arr[subset_num, :].reshape((1,n))
		)
		for subset_num in range(subset_count))

	# ---------------------------------------------------------------------------------------------------------------------------------------------------
	
	phi_x_alpha = dict(
		(
			frozenset(server_combination_matrix[subset_num, :].nonzero()[0]),
			phi_x_alpha[subset_num, :].reshape((m,1)) * compatability_matrix
		)
		for subset_num in range(subset_count) if phi_x_alpha[subset_num, :].sum() > 0)

	# ---------------------------------------------------------------------------------------------------------------------------------------------------


	prod = np.prod
	compatability_matrix = compatability_matrix.astype(float)



	# print('x'*40)
	# print('x'*40)
	# print('x'*40)
	jt_perms = jt_perms if jt_perms is not None else jpermute(range(n))

	# Modified Adan_Weiss Matching Rate Frobetala
	#                           __                                                                                      __
	#                          |                J                                                                        |
	#                          |    ______    ______                                                                     |
	#                          |    \         \          k                                -1    J-1                 -1   |              
	#                          |     \         \       -----             /               \     -----  /           \      | 
	#  r_ij = B x a_i x b_j x  |     /         /        | |  o(l)a(l) x | b(l) - a(l)X(l) |  x  | |  | b(l) - a(l) |     | 
	#                          |    /         /         l=1              \               /      l=k   \           /      |
	#                          |   -------   -------                                                                     |
	#                          |    p in P     k=1                                                                       |
	#                          |__                                                                                     __|

	def sum_over_permutations(jt_perms, inv_beta_alpha_x_xhi, inv_beta_alpha, phi_x_alpha, m, n):


		matching_rates = np.zeros((n, m, n))

		num_of_perms = np.prod(range(1, n+1, 1))

		five_pct = int(num_of_perms/20.)
		
		perm_count = 0
		report_count = 0
		report_time = time()

		vstack = np.vstack
		array = np.array
		cumprod = np.cumprod
		rollaxis = np.rollaxis
		dstack = np.dstack
		hstack = np.hstack
		flip = np.flip

		for perm, swap_idx in jt_perms:
			#print(perm, swap_idx)

			
			if perm_count % five_pct == 0 and print_progress:
				print('progress: {} "%"pct complerte {} {}'.format(report_count * 5, time() - start_time, time() - report_time))
				report_count +=1
				report_time = time()

			if swap_idx == -1:

				new_subset={}
				sub_perms = [frozenset(perm[:k]) for k in range(1, n + 1, 1)]

				inv_beta_alpha_xhi_1_k = array([inv_beta_alpha_x_xhi[sub_perm] for sub_perm in sub_perms])
				prod_inv_beta_alpha_xhi_1_k = cumprod(inv_beta_alpha_xhi_1_k, axis=0) # n x n matrix

				inv_beta_alpha_k_j = [inv_beta_alpha[sub_perm] for sub_perm in sub_perms[:n-1]] + [1]
				prod_inv_beta_alpha_k_j = flip(cumprod(flip(inv_beta_alpha_k_j, 0)), 0) # first idndex to u customers


				phi_x_alpha_1_j = [phi_x_alpha[sub_perm] for sub_perm in sub_perms if sub_perm in phi_x_alpha]
				arr_phi_x_alpha_1_j = array(phi_x_alpha_1_j)
				cutoff = n - len(phi_x_alpha_1_j) - 1
			
			else:

				new_subset = frozenset(perm[:swap_idx + 1])

				inv_beta_alpha_x_xhi_new = inv_beta_alpha_x_xhi[new_subset]
				#prod_inv_beta_alpha_xhi_1_k[swap_idx:] *= inv_beta_alpha_x_xhi_new/inv_beta_alpha_xhi_1_k[swap_idx]
				prod_inv_beta_alpha_xhi_1_k = update_prod_inv_beta_alpha_xhi_1_k(
					prod_inv_beta_alpha_xhi_1_k,
					inv_beta_alpha_x_xhi_new,
					inv_beta_alpha_xhi_1_k[swap_idx],
					swap_idx
					) 

				inv_beta_alpha_xhi_1_k[swap_idx] = inv_beta_alpha_x_xhi_new

				inv_beta_alpha_new = inv_beta_alpha[new_subset]
				# prod_inv_beta_alpha_k_j[:swap_idx + 1] *= inv_beta_alpha_new/inv_beta_alpha_k_j[swap_idx]
				prod_inv_beta_alpha_k_j = update_prod_inv_beta_alpha_k_j(
					prod_inv_beta_alpha_k_j,
					inv_beta_alpha_new,
					inv_beta_alpha_k_j[swap_idx],
					swap_idx
					)

				inv_beta_alpha_k_j[swap_idx] = inv_beta_alpha_new

				if swap_idx > cutoff:
					
					phi_x_alpha_new = phi_x_alpha.get(new_subset, None)

					if phi_x_alpha_new is not None:

						phi_x_alpha_1_j[swap_idx - cutoff -1] = phi_x_alpha_new
						arr_phi_x_alpha_1_j = array(phi_x_alpha_1_j)


					else:
						phi_x_alpha_1_j = phi_x_alpha_1_j[1: ]
						arr_phi_x_alpha_1_j = array(phi_x_alpha_1_j)
						cutoff = n - len(phi_x_alpha_1_j) - 1


				elif swap_idx == cutoff:
					
					phi_x_alpha_new = phi_x_alpha.get(new_subset, None)
					
					if phi_x_alpha_new is not None:
						phi_x_alpha_1_j.insert(0, phi_x_alpha_new)
						arr_phi_x_alpha_1_j = array(phi_x_alpha_1_j)
						cutoff = n - len(phi_x_alpha_1_j) - 1

			
			d, _, _ =  arr_phi_x_alpha_1_j.shape
			
			#matching_rates[n-d:, :, :] += arr_phi_x_alpha_1_j * prod_inv_beta_alpha_xhi_1_k[n-d:, :].reshape((d, 1, n)) * prod_inv_beta_alpha_k_j[n-d:].reshape((d, 1, 1))
			matching_rates = matching_rates_add(
				matching_rates,
				arr_phi_x_alpha_1_j,
				prod_inv_beta_alpha_xhi_1_k[n-d:, :].reshape((d, 1, n)),
				prod_inv_beta_alpha_k_j[n-d:].reshape((d, 1, 1)), n, d)

			perm_count += 1

		matching_rates = matching_rates.sum(axis=0)
		matching_rates = np.dot(alpha.reshape((m,1)), beta.reshape((1,n))) * matching_rates
		matching_rates = matching_rates/matching_rates.sum()

		if pad:
			return matching_rates[: m-1, :]
		else:
			return matching_rates

	return(sum_over_permutations(jt_perms, inv_beta_alpha_x_xhi, inv_beta_alpha, phi_x_alpha, m, n))


def jpermute(iterable):
	"""
	Use the Johnson-Trotter algorithm to return all permutations of iterable.

	The algorithm is applied to a 1-based set of integers representing the indices
	of the given iterable, then a shallow copy of iterable is mutated and returned
	for each successive permutation.
	"""
	# A shallow copy of 'iterable'. This is what is mutated and yielded for each perm.
	sequence = list(iterable)
	length = len(sequence)
	indices = range(1, length+1)

	# The list of directed integers: [-1, 1], [-1, 2], ...
	state = [[-1, idx] for idx in indices]

	# Add sentinels at the beginning and end
	state = [[-1, length+1]] + state + [[-1, length+1]]

	# The first permutation is the sequence itself

	yield list(sequence), -1

	mobile_index = mobile_direction = direction = value = None
	while True:
		# 1. Find the highest mobile
		mobile = -1
		for idx in indices:
			direction, value = state[idx]
			if value > mobile and value > state[idx+direction][1]:
				# value is mobile and greater than the previous mobile
				mobile = value
				mobile_index = idx
				mobile_direction = direction
				if mobile == length:
					# no point in continuing as mobile is as large as it can be.
					break
		if mobile == -1:
			break

		# 2. Swap the mobile with the element it 'sees'
		sees = mobile_index + mobile_direction
		# ... first update the state
		state[mobile_index], state[sees] = state[sees], state[mobile_index]
		# ... then update the sequence
		sequence[mobile_index-1], sequence[sees-1] = sequence[sees-1], sequence[mobile_index-1]

		# 3. Switch the direction of elements greater than mobile
		if mobile < length:
			for idx in indices:
				if state[idx][1] > mobile:
					state[idx][0] = -state[idx][0]

		yield list(sequence), min(mobile_index-1, sees-1)


@jit(nopython=True)
def matching_rates_add(matching_rates, arr_phi_x_alpha_1_j, prod_inv_beta_alpha_xhi_1_k, prod_inv_beta_alpha_k_j, n, d):
	matching_rates[n-d:, :, :] += arr_phi_x_alpha_1_j * prod_inv_beta_alpha_xhi_1_k * prod_inv_beta_alpha_k_j
	return matching_rates


@jit(nopython=True)    
def update_prod_inv_beta_alpha_xhi_1_k(prod_inv_beta_alpha_xhi_1_k, inv_beta_alpha_x_xhi_new, inv_beta_alpha_xhi_1_k, swap_idx):
	prod_inv_beta_alpha_xhi_1_k[swap_idx:] *= inv_beta_alpha_x_xhi_new/inv_beta_alpha_xhi_1_k
	return prod_inv_beta_alpha_xhi_1_k


@jit(nopython=True)
def update_prod_inv_beta_alpha_k_j(prod_inv_beta_alpha_k_j, inv_beta_alpha_new, inv_beta_alpha_k_j, swap_idx):
	prod_inv_beta_alpha_k_j[:swap_idx + 1] *= inv_beta_alpha_new/inv_beta_alpha_k_j
	return prod_inv_beta_alpha_k_j


def quadratic_approximation_cplex(compatability_matrix, alpha, beta, prt=False):

	col_names = []
	qual_names = []
	obj = []
	ub = []
	lb = []
	constraints_coeff = []
	qmat = []
	row_names = []
	rhs = []
	senses = ''
	types = ''
	edge_counter = 0
	translator = dict()



	for i,j in zip(*compatability_matrix.nonzero()):

		col_names.append('r_' + str(i) + str(j))
		# qual_names.append((qual.customer.name, qual.server.name))
		# translator[(qual.customer.name, qual.server.name)] = qual
		obj.append(0.0)
		lb.append(0.0)
		ub.append(1.0)
		constraints_coeff.append([['c_' + str(i), 's_' + str(j)],
								  [1.0, 1.0]])
		qmat.append([[edge_counter], [1/(alpha[i]*beta[j])]])
		edge_counter += 1

	for j, beta_j in enumerate(beta):
		row_names.append('s_' + str(j))
		rhs.append(beta_j)
		senses = senses + 'E'

	for i, alpha_i in enumerate(alpha):
		row_names.append('c_' + str(i))
		rhs.append(alpha_i)
		senses = senses + 'E'

	qp = cplex.Cplex()
	qp.objective.set_sense(qp.objective.sense.minimize)
	qp.linear_constraints.add(rhs=rhs, senses=senses, names=row_names)

	if prt:
		print('obj:', obj)
		print('lb:', lb)
		print('ub:', ub)
		print('col_names:', col_names)
		print('type:', types)
		print('const_coeff:', constraints_coeff)
		print('row_names:', row_names)
		print('rhs:', rhs)

	qp.variables.add(obj=obj, lb=lb, ub=ub, names=col_names, types=types, columns=constraints_coeff)

	if prt:
		print('row_names:')
		print(row_names)
		print('constraints:')
		print(constraints_coeff)

	qp.objective.set_quadratic(qmat)

	if prt:
		print(qmat)
		print('problem type:')
		print(qp.problem_type.QP)
		print(qp.get_problem_type())

	try:
		qp.solve()
	except CplexError as exc:
		print(exc)
		return

	slack = qp.solution.get_linear_slacks()
	x = qp.solution.get_values()


	matching_rates = np.zeros(compatability_matrix.shape)

	matching_rates[compatability_matrix.nonzero()] = x

	return matching_rates


def entropy_approximation_w_log(compatability_matrix, z, q, lamda, mu, check_every=10**2, max_iter=10**7, epsilon=10**-7, pad=False, ret_all=False):

	k = 0
	within_epsilon = True
	converge = False
		

	matching_rates = z
	printarr(matching_rates, 'initial')
	for k in range(max_iter):

		# printarr(lamda/((matching_rates * q).sum(axis=1)), 'lamda/((matching_rates * q).sum(axis=1))')


		h_lam = np.log((compatability_matrix.transpose() * lamda/(matching_rates * q).sum(axis=1)).transpose(),out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		log_matching_rates = np.log(matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		log_matching_rates = log_matching_rates + h_lam * q
		matching_rates = np.exp(log_matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		
		h_mu = np.log(compatability_matrix * mu/(matching_rates * q).sum(axis=0), out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		log_matching_rates = np.log(matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		log_matching_rates = log_matching_rates + h_mu * q
		matching_rates = np.exp(log_matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		# matching_rates = matching_rates * mu/(matching_rates * q).sum(axis=0)
		# printarr(matching_rates, str(k))
		if k > 0 and k % check_every == 0 or k == max_iter - 1:
			cur_iter, gap_pct = (k,
				max(
					max(abs(mu - (matching_rates * q).sum(axis=0))/mu),
					max(abs(lamda - (matching_rates * q).sum(axis=1))/lamda)
					)
				)

			if gap_pct < epsilon:
				converge = True
				break

	return matching_rates


def entropy_approximation_w(compatability_matrix, z, q, lamda, mu, check_every=10**2, max_iter=10**7, epsilon=10**-7, pad=False, ret_all=False):

	k = 0
	within_epsilon = True
	converge = False
		

	matching_rates = z
	printarr(matching_rates, 'initial')
	for k in range(max_iter):

		# printarr(lamda/((matching_rates * q).sum(axis=1)), 'lamda/((matching_rates * q).sum(axis=1))')


		h_lam = (compatability_matrix.transpose() * lamda/(matching_rates * q).sum(axis=1)).transpose()
		# log_matching_rates = np.log(matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		matching_rates = matching_rates * np.power(h_lam, q)
		# matching_rates = np.exp(log_matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		
		h_mu = compatability_matrix * mu/(matching_rates * q).sum(axis=0)
		# log_matching_rates = np.log(matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		matching_rates = matching_rates * np.power(h_mu, q)
		# matching_rates = np.exp(log_matching_rates, out=np.zeros_like(compatability_matrix), where=(compatability_matrix != 0))
		# matching_rates = matching_rates * mu/(matching_rates * q).sum(axis=0)
		# printarr(matching_rates, str(k))
		if k > 0 and k % check_every == 0 or k == max_iter - 1:
			cur_iter, gap_pct = (k,
				max(
					max(abs(mu - (matching_rates * q).sum(axis=0))/mu),
					max(abs(lamda - (matching_rates * q).sum(axis=1))/lamda)
					)
				)

			if gap_pct < epsilon:
				converge = True
				break

	return matching_rates


def entropy_approximation(compatability_matrix, lamda, mu, check_every=10**2, max_iter=10**7, epsilon=10**-7, pad=False, ret_all=False):

	k = 0
	within_epsilon = True
	converge = False

	if  sps.isspmatrix(compatability_matrix):
		is_sps = True
	else:
		is_sps = False

	if pad:
		if is_sps:
			compatability_matrix = sps.vstack([compatability_matrix, np.ones(len(mu))])
			lamda = np.append(lamda, mu.sum() - lamda.sum())
		else:
			compatability_matrix = np.vstack([compatability_matrix, np.ones(len(mu))])
			lamda = np.append(lamda, mu.sum() - lamda.sum())			

	matching_rates = compatability_matrix

	for k in range(max_iter):

		if is_sps:
			matching_rates = sps.diags(lamda/matching_rates.sum(axis=1).A.ravel()).dot(matching_rates)
			matching_rates = matching_rates.dot(sps.diags(mu/matching_rates.sum(axis=0).A.ravel()))
			if k > 0 and k % check_every == 0:
				cur_iter, gap_pct =  (k, 
					max(
						np.max(np.abs((mu - matching_rates.sum(axis=0)))/mu[:np.newaxis]),
						np.max(np.abs((matching_rates.sum(axis=1).T - lamda))/lamda[:np.newaxis])
						)
					)
				if gap_pct < epsilon:
					converge = True
					break
		else:
			matching_rates = (matching_rates.transpose() * lamda/matching_rates.sum(axis=1)).transpose()
			matching_rates = matching_rates * mu/matching_rates.sum(axis=0)
			if k > 0 and k % check_every == 0 or k == max_iter - 1:
				cur_iter, gap_pct = (k,
					max(
						max(abs(mu - matching_rates.sum(axis=0))/mu),
						max(abs(lamda - matching_rates.sum(axis=1))/lamda)
						)
					)

				if gap_pct < epsilon:
					converge = True
					break

	if converge:
		if pad:
			if ret_all:
				return True, matching_rates[:-1, :], gap_pct
			else:
				return matching_rates[:-1, :]
		else:
			if ret_all:
				return True, matching_rates, gap_pct
			else:
				return matching_rates	
	else:
		if ret_all:
			return False, None, gap_pct
		else:
			return None


def fast_entropy_approximation(compatability_matrix, lamda, mu, check_every=10**2, max_iter=10**7, epsilon=10**-7, pad=False, ret_all=False):

	k = 0
	within_epsilon = True
	converge = False
	compatability_matrix = compatability_matrix.astype(int)
	m, n = compatability_matrix.shape
	if sps.isspmatrix(compatability_matrix):
		compatability_matrix = compatability_matrix.todense().A

	if pad:

		compatability_matrix = np.vstack([compatability_matrix, np.ones(len(mu)).astype(int)])
		compatability_matrix = compatability_matrix.astype(int)
		lamda = np.append(lamda, mu.sum() - lamda.sum())	
		m = m + 1		

	matching_rates = fast_matrix_scaling(compatability_matrix.astype(float), lamda, mu, m ,n) 

	if matching_rates is not None:
		converge = True

	if converge:
		if pad:
			if ret_all:
				return True, matching_rates[:-1, :]
			else:
				return matching_rates[:-1, :]
		else:
			if ret_all:
				return True, matching_rates, gap_pct
			else:
				return matching_rates	
	else:
		if ret_all:
			return False, None, gap_pct
		else:
			return None


@jit(nopython=True, cache=True)
def fast_matrix_scaling(compatability_matrix, lamda, mu, m, n):

	max_iter = 10**7
	check_every = 10**2
	epsilon = 10**-7
	matching_rates = compatability_matrix

	for k in range(max_iter):

		matching_rates = (matching_rates.transpose() * lamda/matching_rates.sum(axis=1)).transpose()
		matching_rates = matching_rates * mu/matching_rates.sum(axis=0)

		if k > 0 and k % check_every == 0 or k == max_iter - 1:
			
			cur_iter = k 
			gap_pct_cols = (np.abs(mu - matching_rates.sum(axis=0))/mu).max()
			gap_pct_rows = (np.abs(lamda - matching_rates.sum(axis=1))/lamda).max()
			if gap_pct_cols < epsilon and gap_pct_rows < epsilon:
				converge = True
				break

	if converge:
		return matching_rates
	else:
		return None


def ohm_law_approximation(compatability_matrix, alpha, beta):

	m,n = compatability_matrix.shape
	alpha_beta = (np.diag(alpha).dot(compatability_matrix)).dot(np.diag(beta))

	bottom_left = alpha_beta.T
	upper_right = -1 * alpha_beta
	upper_left = np.diag(alpha_beta.sum(axis=1))
	bottom_right = -1 * np.diag(alpha_beta.sum(axis=0))

	A = np.vstack([
		np.hstack([upper_left, upper_right]), 
		np.hstack([bottom_left, bottom_right]),
		])

	b = np.vstack([alpha.reshape((m, 1)), beta.reshape((n, 1))])
	w_v = np.linalg.solve(A, b)
	v = np.squeeze(w_v[:m,:])
	w = np.squeeze(w_v[m:,:])

	
	delta_v_w = (np.diag(v).dot(compatability_matrix) - compatability_matrix.dot(np.diag(w)))
	matching_rates = delta_v_w * alpha_beta

	return matching_rates, v, w


def local_entropy(compatability_matrix, lamda, mu, s=None, prt=False):

	m, n  = compatability_matrix.shape
	k = m + n
	l = (m + 1) * n

	rows = []
	cols = []
	data = []
	if s is None:
		s = np.ones(m)
	col_set = set()

	for i, j in zip(*compatability_matrix.nonzero()):
			if prt:
				print((i, j),'-->', (i, i * n + j), (m + j, i * n + j))

			rows.append(i)
			cols.append(i * n + j)
			data.append(1)

			rows.append(m + j)
			cols.append(i * n + j)
			data.append(lamda[i] * s[i]/mu[j])

			# col_set.add(i * n + j)

	for j in range(n):

		rows.append(m + j)
		cols.append(m * n + j)
		data.append(1)

	rows = np.array(rows)
	cols = np.array(cols)
	data = np.array(data)
	
	A = np.zeros((m + n, (m + 1) * n))
	A[rows, cols] = data
	b = np.ones(m + n)
	
	z = np.vstack([compatability_matrix, np.ones(n)]).ravel()
	
	pi_hat, duals = fast_primal_dual_algorithm(compatability_matrix, A, b, z, m, n, prt=True)
	pi_hat = pi_hat.reshape((m + 1, n))
	return(np.dot(np.diag(lamda), pi_hat[:m, :]))


def rho_entropy(compatability_matrix, lamda, mu, s=None, prt=False):

	m, n  = compatability_matrix.shape
	k = m + n
	l = (m + 1) * n

	rows = []
	cols = []
	data = []
	if s is None:
		s = np.ones(m)
	col_set = set()

	for i, j in zip(*compatability_matrix.nonzero()):

			if prt:
				print((i, j),'-->', (i, i * n + j), (m + j, i * n + j))

			rows.append(i)
			cols.append(i * n + j)
			data.append(1)

			rows.append(m + j)
			cols.append(i * n + j)
			data.append(lamda[i] * s[i])

	for j in range(n):

		rows.append(m + j)
		cols.append(m * n + j)
		data.append(-mu[j])

	rows = np.array(rows)
	cols = np.array(cols)
	data = np.array(data)
	
	A = np.zeros((m + n, (m + 1) * n))
	A[rows, cols] = data
	b = np.hstack((np.ones(m), np.zeros(n)))
	
	z = np.vstack([compatability_matrix, np.ones(n)]).ravel()
	
	pi_hat, duals = fast_primal_dual_algorithm(A, b, z, m, n, prt=True)
	pi_hat = pi_hat.reshape((m + 1, n))
	return(np.dot(np.diag(lamda), pi_hat[:m, :]))


def node_entropy(compatability_matrix, lamda, mu, prt=False):
	

	compatability_matrix = np.vstack([compatability_matrix, np.ones(len(mu))])
	lamda = np.append(lamda, mu.sum() - lamda.sum())
	m, n = compatability_matrix.shape
	k = m + n 
	l = (m * n) + (n * m) 

	rows = []
	cols = []
	data = []
	col_set = set()

	for i, j in zip(*compatability_matrix.nonzero()):
		if prt:
			print((i, j),'-->', (i, i * n + j), (m + j, i * n + j))

		rows.append(i)
		cols.append(i * n + j)
		data.append(1)

		rows.append(m + j)
		cols.append((m * n) + j * m + i)
		data.append(1)

		rows.append(k)
		cols.append(i * n + j)
		data.append(lamda[i])
		rows.append(k)
		cols.append((m * n) + j * m + i)
		data.append(-mu[j])

		k += 1

	rows = np.array(rows)
	cols = np.array(cols)
	data = np.array(data)

	A = np.zeros((k, l))
	A[rows, cols] = data
	b = np.concatenate([np.ones(m + n), np.zeros(k - (m + n))], axis=0) 
	z = np.concatenate([compatability_matrix.ravel(), compatability_matrix.T.ravel()], axis=0)

	pi_hat, duals = fast_primal_dual_algorithm(A, b, z, m, n)
	pi_hat = pi_hat[:m * n].reshape((m, n))
	return(np.dot(np.diag(lamda[: m - 1]), pi_hat[: m-1, :]))


def fast_primal_dual_algorithm(compatability_matrix, A, b, z, m, n, pi0=None, act_rows=None , check_every=10**3, max_iter=10**7, max_time=600, epsilon=10**-6, prt=True, prtall=True):

	start_time = time()

	def f(pi):

		res = np.divide(pi, z, out=np.zeros_like(pi), where= z!=0)
		res = pi * ma.log(res).filled(0)

		return res.sum()

	def check_optimality_gap():
			log_exp_A_eta = -At.dot(eta)
			pi_eta = np.exp(ze + log_exp_A_eta)
			f_pi_eta = f(pi_eta) + eta.dot(A.dot(pi_eta) - b)
			gap_k = f(pi_hat) - f_pi_eta
			gap_k_pct = gap_k/np.abs(f_pi_eta)
			return gap_k, gap_k_pct

	def check_feasibility_gap():

		gap_k = b - A.dot(pi_hat)

		return ((gap_k*gap_k).sum())**0.5

	def check_stop(i, prt=False):

		opt_gap, opt_gap_pct = check_optimality_gap()
		feas_gap = check_feasibility_gap()
		if prtall or ((i % 10**5 == 0) and prt):
			print('iteration',  i)
			print('optimality gap is:', opt_gap)
			print('optimality gap pct is: ', opt_gap_pct)
			print('feasibility gap is: ', feas_gap)
		if feas_gap > 3:
			return False, True, False
		if opt_gap_pct < epsilon:
			if feas_gap < epsilon:
				return True, False, False
		if time() - start_time > max_time:
			return False, False, True

		return False, False, False

	# ze = z * exp(-1.0)
	# v = np.amin(z[np.where(z > 0)])
	L = 2 #min(((1.0/v) * (np.amax(np.abs(A[:m].sum(axis=1))) + np.amax(np.abs(A[m:].sum(axis=1))))), 3)

	if prt or True:
		print('L', L)

	flag = False

	try:
		while not flag:

			m_p_n_p_1, m_p_1_t_n = A.shape
			At = A.transpose()
			pi_k = np.zeros((m_p_1_t_n, ))
			pi_hat = np.zeros((m_p_1_t_n, ))
			prev_pi_hat = np.zeros((m_p_1_t_n, ))
			prev_gap = np.zeros((m_p_n_p_1,))
			lamda = np.zeros((m_p_n_p_1, ))
			prev_lamda = np.zeros((m_p_n_p_1, ))
			zeta = np.zeros((m_p_n_p_1, ))
			ze = ma.log(z).filled(0) - 1.0
			v = np.amin(z[np.where(z > 0)])

			for i in np.arange(max_iter):

				alpha = (i + 1.0)/2.0
				tau = 2.0/(i+3.0)

				if i == 0 and prt:
					print('starting fast primal dual gradient descent')
					s = time()

					if i == 0 and pi0 is not None:
						pi_k = pi0
						pi_hat = pi0
						gap = b - A.dot(pi_k)
						prev_gap = gap

					else:
						log_exp_A_lamda = -At.dot(lamda)
						pi_k = np.exp(ze + log_exp_A_lamda)
				
				if i > 0:
					log_exp_A_lamda = -At.dot(lamda)
					pi_k = np.exp(ze + log_exp_A_lamda)
				pi_hat = tau * pi_k + (1.0 - tau) * pi_hat

				if (i > 0 and i % check_every == 0):
					converged, oob, time_violation = check_stop(i, prt)
					if converged:
						flag=True
						break
					elif oob:
						flag=False
						print('oob feasibility gap halving step size')
						L = L * 2
						print('new L', L)
						break
					elif time_violation:
						print('time exceed aborting algorithm')
						return None, None

				gap = b - A.dot(pi_k)
				eta = lamda - (1.0/L) * gap
				zeta = zeta - (alpha/L) * gap
				lamda = (tau * zeta) + (1.0 - tau) * eta

		if prt:
			print('ended fast primal-dual algorithm after ' + str(i) + ' iterations')
			print('run time:', time() - s, 'seconds')
		return pi_hat, lamda
	except:
		print('failed to solve')
		return None, None


def quadratic_approximation(compatability_matrix, alpha, beta, prt=False, pad=False):

	if pad:
		compatability_matrix = np.vstack((compatability_matrix, np.ones(len(beta))))
		alpha = np.append(alpha, beta.sum() - alpha.sum())

	qp = Model()
	obj = QuadExpr()

	m, n = compatability_matrix.shape

	matching_rates = np.zeros((m, n))

	customers = range(m)
	servers = range(n)

	edges = list(zip(*compatability_matrix.nonzero()))
	flow = qp.addVars(edges, name="flow", lb=0, ub=1, vtype=GRB.CONTINUOUS)

	min_a_b = 1
	for i, j in zip(*compatability_matrix.nonzero()):
		if (alpha[i]*beta[j])< min_a_b:
			min_a_b = alpha[i]*beta[j]
		obj += flow[i,j] * flow[i,j] * (1/(alpha[i]*beta[j]))

	print('max_quad_coeff:', 1/min_a_b)

	qp.addConstrs((flow.sum(i, '*') == alpha[i] for i in customers), 'row_sums')
	qp.addConstrs((flow.sum('*', j) == beta[j] for j in servers), 'col_sums')

	qp.setObjective(obj)
	qp.optimize()

	if qp.status == GRB.Status.OPTIMAL:
		x = qp.getAttr('x', flow)

	for i, j in edges:
		matching_rates[i, j] = x[i, j]

	if pad:
		return matching_rates[:m, :]
	else:
		return matching_rates


def alis_approximation(compatability_matrix, alpha, beta, rho):


	m, n = compatability_matrix.shape

	adj_beta = np.ones(n)/n * (1 - rho) + beta * rho

	# p = np.ones((n,n)) *(1/n)
	p = np.vstack([beta for _ in range(n)])
	s = time()
	p = entropy_approximation(p, np.ones(n), n*adj_beta)

	def p_to_r(p, alpha):

		q = (1. - np.dot(compatability_matrix, p.T)).T
		# printarr(q,'q')
		q = np.cumprod(q, axis=0)
		# printarr(q,'q')
		c = 1. / (np.ones(m) - q[-1, :])
		# printarr(c,'c')
		q = np.vstack((np.ones((1, m)), q[:-1, :]))
		# printarr(q,'q')
		q = q * c
		# printarr(q,'q')
		try:
			q = q * alpha
		except:
			print('q', q.shape)
			print('alpha', alpha.shape)

		# printarr(q,'q')
		r = np.zeros((n, m, n))
		for k in range(n):
			for i,j in zip(*compatability_matrix.nonzero()):
				r[k,i,j] = q[k, i] * p[k, j]
		
		return r, p[0, :]
	
	def r_to_p(r, p_one):

		p = np.zeros((n, n))

		for k in range(1, n, 1):
			for j in range(n):
				p[k, j] = r[:k, :, j].sum()/r[:k, :, :].sum()

		p = np.vstack((p_one, p[1:, :]))

		p = entropy_approximation(p, np.ones(n), n*adj_beta)

		return p

	converge = False
	iter_k = 0

	s = time()

	while not converge and iter_k < 100000:

		prev_p = p 
		r , p_one = p_to_r(p, alpha)
		p = r_to_p(r, p_one)
		
		# print(iter_k, np.abs(p - prev_p).sum())
		if np.abs(p - prev_p).sum() < 10**-9:
			converge = True

		iter_k += 1

	# print('time to converge:',  time() - s, 'iterations:', iter_k)

	# printarr(p, 'last p reg')
	# print(p.sum(axis=0))
	# print(p.sum(axis=1))
	
	r, _ = p_to_r(p, alpha)

	Q = compatability_matrix
	# for k in range(n):
	# 	for i in range(m):
	# 		for j in range(n):
	# 			if Q[i,j] == 1.:
	# 				ratio = p[k, j]/sum(p[k,h]*Q[i,h] for h in range(n))
	# 				h_ratio = (1/p[k, j])/sum(Q[i,h]/p[k,h] for h in range(n) if Q[i,h]>0)
	# 				print(k, i, j, ratio, h_ratio)

	r = rho * r.sum(axis=0)

	return r


def fast_alis_approximation(compatability_matrix, alpha, beta, rho, check_every=10, max_time=600):


	m, n = compatability_matrix.shape

	col_sums = n * (np.ones(n)/n * (1 - rho) + beta * rho)

	p = np.vstack([beta for _ in range(n)])
	p = fast_matrix_scaling(p, np.ones(n), col_sums, m, n)

	converge = False
	timed_out = False

	iter_k = 1
	start_time = time()

	while not converge and not timed_out:
		prev_p = p
		q = p_to_q(p, compatability_matrix, alpha, m, n)
		qq = q[..., None] 
		pp = p[:, None, :]
		p = r_to_p(compatability_matrix, pp, qq, p, n)
		p = fast_matrix_scaling(p, np.ones(n), col_sums, m, n)
		if iter_k > 0 and iter_k % check_every == 0:
			if np.abs(prev_p - p).sum() < 10**-6:
				converge = True
			if time() - start_time > max_time:
				timed_out = True
		iter_k = iter_k + 1

	q = p_to_q(p, compatability_matrix, alpha, m, n)

	r = p[:, None, :] * q[..., None] 
	r = r * compatability_matrix
	r = rho * r.sum(axis=0)

	return r

	
@jit(nopython=True, cache=True)
def p_to_q(p, compatability_matrix, alpha, m, n):


	q = (np.ones((n, m)) - np.dot(compatability_matrix, p.T).T)
	
	for ell in range(1, n, 1):
		q[ell, :] = q[ell, :] * q[ell - 1, :] 

	c = 1. / (np.ones((1, m)) - q[-1, :])

	q = np.vstack((np.ones((1, m)), q[:-1, :]))

	q = q * c * alpha

	return q

# @jit(nopython=True, cache=True)
def r_to_p(compatability_matrix, pp, qq, p, n):

	r = qq * pp
	r = r * compatability_matrix
	r = r.sum(axis=1)
	r_ell = r[0]
	for ell in range(1, n, 1):
		p[ell, :] = r_ell/r_ell.sum()
		r_ell = r_ell + r[ell]

	return p


def convert_to_normal_form(c, z, w, q):

	m, n  = q.shape
	
	c_w = np.divide(-1*c, w, out=np.zeros_like(q), where=(w != 0))
	exp_c_w = np.exp(c_w, out=np.zeros_like(q), where=(q != 0))
	z = z * w * exp_c_w
	q = np.divide(q, w, out=np.zeros_like(q), where=(q != 0))

	return z, q


def metrize_constraintes(q, row_sums, col_sums, z, pi_0=None, prt=False):
	
	m, n  = q.shape
	# print('q.shape: ', q.shape)
	
	k = m + n
	l = len(q.nonzero()[0])

	# print('k,l: ', k, l)

	new_z = []
	rows = []
	cols = []
	data = []
	if pi_0 is not None:
		new_pi_0 = []

	for col, (i, j) in enumerate(zip(*q.nonzero())):
			
		if prt:
				print((i, j),'-->', (i, i * n + j), (m + j, i * n + j))

		rows.append(i)
		cols.append(col)
		data.append(q[i, j])

		rows.append(m + j)
		cols.append(col)
		data.append(q[i, j])

		new_z.append(z[i,j])

		if pi_0 is not None:
			new_pi_0.append(pi_0[i,j])

	rows = np.array(rows)
	cols = np.array(cols)
	data = np.array(data)
	
	A = np.zeros((k, l))
	A[rows, cols] = data
	b = np.hstack([row_sums, col_sums])
	new_z = np.array(new_z)

	if pi_0 is not None:
		new_pi_0 = np.array(new_pi_0)
		return A, b, new_z, new_pi_0
	else:
		return A, b, new_z, None


def weighted_entropy_regulerized_ot(compatability_matrix, c, lamda, s, mu, rho, gamma, weighted=False):

	m, n = compatability_matrix.shape
	
	c = gamma * np.vstack([c, np.zeros((1, n))])

	if weighted:
		w = np.vstack([(1. - rho) * np.ones(n) * compatability_matrix, rho * np.ones(n)])
		w = (np.vstack([compatability_matrix, np.ones(n)]).sum()/w.sum()) * w
		# print('w_max weighted: ', w.max())
	else:
		w = np.vstack([compatability_matrix, np.ones(n)])
		# print('w_max not weighted: ', w.max())
	if gamma > 0:
		w = (1 - gamma) * w

	compatability_matrix = np.vstack([compatability_matrix, np.ones((1, n))])
	
	eta = lamda * s * rho
	eta = np.append(eta, mu.sum() - eta.sum())
	s = np.append(s, 1)
	
	z, q = convert_to_normal_form(c, compatability_matrix, w, compatability_matrix)

	if weighted:
		pi_0 = entropy_approximation(compatability_matrix, eta, mu)
		pi_0 = pi_0 * w
	else:
		pi_0 = None

	A, b, z , pi_0 = metrize_constraintes(q, eta, mu, z, pi_0)

	if A.shape[0] > 10:
		print(A.shape[0], 'x',  A.shape[1], ' matrix going sparse')
		A = sps.csr_matrix(A)

	eta_w, _ = fast_primal_dual_algorithm(compatability_matrix, A, b, z, m + 1, n, pi0=pi_0, act_rows=None , check_every=10**3, max_iter=10**8, epsilon=10**-6, prt=True, prtall=False)
	if eta_w is not None:
		m_n_eta_w = np.zeros((m + 1, n))
		for col, (i,j) in enumerate(zip(*compatability_matrix.nonzero())):
			m_n_eta_w[i,j] = eta_w[col]
		eta_w = m_n_eta_w
		eta = np.divide(eta_w, w, out=np.zeros_like(compatability_matrix), where= w!= 0)
		r = np.divide(eta, np.dot(np.diag(s), compatability_matrix), out=np.zeros_like(compatability_matrix), where= w!= 0)
		# printarr(r.sum(axis=0), 'r.sum(axis=0)')
		# printarr(r.sum(axis=1), 'r.sum(axis=1)')
		return r, w
	else:
		return None, None


def sinkhorn_knopp(M, a, b, compatability_matrix, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, **kwargs):
	"""
	Solve the entropic regularization optimal transport problem and return the OT matrix

	The function solves the following optimization problem:

	.. math::
		\gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

		s.t. \gamma 1 = a

			 \gamma^T 1= b

			 \gamma\geq 0
	where :

	- M is the (ns,nt) metric cost matrix
	- :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
	- a and b are source and target weights (sum to 1)

	The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [2]_


	Parameters
	----------
	a : np.ndarray (ns,)
		samples weights in the source domain
	b : np.ndarray (nt,) or np.ndarray (nt,nbb)
		samples in the target domain, compute sinkhorn with multiple targets
		and fixed M if b is a matrix (return OT loss + dual variables in log)
	M : np.ndarray (ns,nt)
		loss matrix
	reg : float
		Regularization term >0
	numItermax : int, optional
		Max number of iterations
	stopThr : float, optional
		Stop threshol on error (>0)
	verbose : bool, optional
		Print information along iterations
	log : bool, optional
		record log if True


	Returns
	-------
	gamma : (ns x nt) ndarray
		Optimal transportation matrix for the given parameters
	log : dict
		log dictionary return only if log==True in parameters

	Examples
	--------

	>>> import ot
	>>> a=[.5,.5]
	>>> b=[.5,.5]
	>>> M=[[0.,1.],[1.,0.]]
	>>> ot.sinkhorn(a,b,M,1)
	array([[ 0.36552929,  0.13447071],
		   [ 0.13447071,  0.36552929]])


	References
	----------

	.. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013


	See Also
	--------
	ot.lp.emd : Unregularized OT
	ot.optim.cg : General regularized OT

	"""

	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	M = np.asarray(M, dtype=np.float64)

	if len(a) == 0:
		a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
	if len(b) == 0:
		b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

	# init data
	Nini = len(a)
	Nfin = len(b)

	if len(b.shape) > 1:
		nbb = b.shape[1]
	else:
		nbb = 0

	if log:
		log = {'err': []}

	# we assume that no distances are null except those of the diagonal of
	# distances
	if nbb:
		u = np.ones((Nini, nbb)) / Nini
		v = np.ones((Nfin, nbb)) / Nfin
	else:
		u = np.ones(Nini) / Nini
		v = np.ones(Nfin) / Nfin

	# print(reg)

	# Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
	K = np.empty(M.shape, dtype=M.dtype)
	np.divide(M, -reg, out=K)
	np.exp(K, out=K)
	K = K * compatability_matrix

	# print(np.min(K))
	tmp2 = np.empty(b.shape, dtype=M.dtype)

	Kp = (1 / a).reshape(-1, 1) * K
	cpt = 0
	err = 1
	while (err > stopThr and cpt < numItermax):
		uprev = u
		vprev = v

		KtransposeU = np.dot(K.T, u)
		v = np.divide(b, KtransposeU)
		u = 1. / np.dot(Kp, v)

		if (np.any(KtransposeU == 0) or
				np.any(np.isnan(u)) or np.any(np.isnan(v)) or
				np.any(np.isinf(u)) or np.any(np.isinf(v))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print('Warning: numerical errors at iteration', cpt)
			u = uprev
			v = vprev
			break
		if cpt % 10 == 0:
			# we can speed up the process by checking for the error only all
			# the 10th iterations
			if nbb:
				err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
					np.sum((v - vprev)**2) / np.sum((v)**2)
			else:
				# compute right marginal tmp2= (diag(u)Kdiag(v))^T1
				np.einsum('i,ij,j->j', u, K, v, out=tmp2)
				err = np.linalg.norm(tmp2 - b)**2  # violation of marginal
			if log:
				log['err'].append(err)

			if verbose:
				if cpt % 200 == 0:
					print(
						'{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))
		cpt = cpt + 1
	if log:
		log['u'] = u
		log['v'] = v

	if nbb:  # return only loss
		res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
		if log:
			return res, log
		else:
			return res

	else:  # return OT matrix

		if log:
			return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
		else:
			return u.reshape((-1, 1)) * K * v.reshape((1, -1))


def sinkhorn_stabilized(M, a, b, compatability_matrix, reg, numItermax=1000, tau=1e3, stopThr=1e-9, warmstart=None, verbose=False, print_period=20, log=False, **kwargs):
	"""
	Solve the entropic regularization OT problem with log stabilization

	The function solves the following optimization problem:

	.. math::
		\gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

		s.t. \gamma 1 = a

			 \gamma^T 1= b

			 \gamma\geq 0
	where :

	- M is the (ns,nt) metric cost matrix
	- :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
	- a and b are source and target weights (sum to 1)

	The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
	scaling algorithm as proposed in [2]_ but with the log stabilization
	proposed in [10]_ an defined in [9]_ (Algo 3.1) .


	Parameters
	----------
	a : np.ndarray (ns,)
		samples weights in the source domain
	b : np.ndarray (nt,)
		samples in the target domain
	M : np.ndarray (ns,nt)
		loss matrix
	reg : float
		Regularization term >0
	tau : float
		thershold for max value in u or v for log scaling
	warmstart : tible of vectors
		if given then sarting values for alpha an beta log scalings
	numItermax : int, optional
		Max number of iterations
	stopThr : float, optional
		Stop threshol on error (>0)
	verbose : bool, optional
		Print information along iterations
	log : bool, optional
		record log if True


	Returns
	-------
	gamma : (ns x nt) ndarray
		Optimal transportation matrix for the given parameters
	log : dict
		log dictionary return only if log==True in parameters

	Examples
	--------

	>>> import ot
	>>> a=[.5,.5]
	>>> b=[.5,.5]
	>>> M=[[0.,1.],[1.,0.]]
	>>> ot.bregman.sinkhorn_stabilized(a,b,M,1)
	array([[ 0.36552929,  0.13447071],
		   [ 0.13447071,  0.36552929]])


	References
	----------

	.. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

	.. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

	.. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.


	See Also
	--------
	ot.lp.emd : Unregularized OT
	ot.optim.cg : General regularized OT

	"""

	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	M = np.asarray(M, dtype=np.float64)

	if len(a) == 0:
		a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
	if len(b) == 0:
		b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

	# test if multiple target
	if len(b.shape) > 1:
		nbb = b.shape[1]
		a = a[:, np.newaxis]
	else:
		nbb = 0

	# init data
	na = len(a)
	nb = len(b)

	cpt = 0
	if log:
		log = {'err': []}

	# we assume that no distances are null except those of the diagonal of
	# distances
	if warmstart is None:
		alpha, beta = np.zeros(na), np.zeros(nb)
	else:
		alpha, beta = warmstart

	if nbb:
		u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
	else:
		u, v = np.ones(na) / na, np.ones(nb) / nb

	def get_K(alpha, beta):
		"""log space computation"""
		return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg) * compatability_matrix

	def get_Gamma(alpha, beta, u, v):
		"""log space gamma computation"""
		return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg + np.log(u.reshape((na, 1))) + np.log(v.reshape((1, nb)))) * compatability_matrix

	# print(np.min(K))

	K = get_K(alpha, beta)
	transp = K
	loop = 1
	cpt = 0
	err = 1
	while loop:

		uprev = u
		vprev = v

		# sinkhorn update
		v = b / (np.dot(K.T, u) + 1e-16)
		u = a / (np.dot(K, v) + 1e-16)

		# remove numerical problems and store them in K
		if np.abs(u).max() > tau or np.abs(v).max() > tau:
			if nbb:
				alpha, beta = alpha + reg * \
					np.max(np.log(u), 1), beta + reg * np.max(np.log(v))
			else:
				alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
				if nbb:
					u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
				else:
					u, v = np.ones(na) / na, np.ones(nb) / nb
			K = get_K(alpha, beta)

		if cpt % print_period == 0:
			# we can speed up the process by checking for the error only all
			# the 10th iterations
			if nbb:
				err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
					np.sum((v - vprev)**2) / np.sum((v)**2)
			else:
				transp = get_Gamma(alpha, beta, u, v)
				err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
			if log:
				log['err'].append(err)

			if verbose:
				if cpt % (print_period * 20) == 0:
					print(
						'{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
				print('{:5d}|{:8e}|'.format(cpt, err))

		if err <= stopThr:
			loop = False

		if cpt >= numItermax:
			loop = False

		if np.any(np.isnan(u)) or np.any(np.isnan(v)):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print('Warning: numerical errors at iteration', cpt)
			u = uprev
			v = vprev
			break

		cpt = cpt + 1

	# print('err=',err,' cpt=',cpt)
	if log:
		log['logu'] = alpha / reg + np.log(u)
		log['logv'] = beta / reg + np.log(v)
		log['alpha'] = alpha + reg * np.log(u)
		log['beta'] = beta + reg * np.log(v)
		log['warmstart'] = (log['alpha'], log['beta'])
		if nbb:
			res = np.zeros((nbb))
			for i in range(nbb):
				res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
			return res, log

		else:
			return get_Gamma(alpha, beta, u, v), log
	else:
		if nbb:
			res = np.zeros((nbb))
			for i in range(nbb):
				res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
			return res
		else:
			return get_Gamma(alpha, beta, u, v)





					# pi_k = pi0 if i == 0 and pi0 is not None else ze*np.exp(At.dot(lamda))

				
				# if 0 <= i < 10:
				# 	print('m: ', m)
				# 	print('n: ', n)
				# 	print('m + n + 1: ', m + n + 1, 'len(b): ', len(b))

				# 	print(i)
				# 	print('-------------------------------')

				# 	prev_pi_hat_mat = np.zeros((m+1, n))
				# 	for col, (i,j) in enumerate(zip(*compatability_matrix.nonzero())):
				# 		prev_pi_hat_mat[i,j] = prev_pi_hat[col]
					
				# 	printarr(prev_pi_hat_mat, 'prev_pi_hat')

				# 	pi_hat_mat = np.zeros((m+1, n))
				# 	for col, (i,j) in enumerate(zip(*compatability_matrix.nonzero())):
				# 		pi_hat_mat[i,j] = pi_hat[col]
					
				# 	printarr(pi_hat_mat, 'pi_hat')

				# 	printarr(np.dot(A, prev_pi_hat)[m: ], 'prev_col_sum')
				# 	printarr(np.dot(A, pi_hat)[m: ], 'cur_col_sum')
				# 	printarr(np.dot(A, prev_pi_hat)[ :m], 'prev_row_sum')
				# 	printarr(np.dot(A, pi_hat)[ :m], 'cur_row_sum')
					
				# 	printarr(b[m: ], 'req_col_sum')
				# 	printarr(b[:m], 'req_row_sum')

				# 	printarr(prev_gap[m: ], 'prev_col_violation')
				# 	printarr(gap[m: ], 'cur_col_violation')

				# 	printarr(prev_gap[ :m], 'prev_row_violation')
				# 	printarr(gap[ :m], 'cur_row_violation')

				# 	printarr(lamda[m: ], 'lamda_cols')
				# 	printarr(prev_lamda[m: ], 'prev_lamda_cols')

				# 	printarr(lamda[ :m], 'lamda_rows')
				# 	printarr(prev_lamda[ :m], 'prev_lamda_rows')
		 
				# 	prev_pi_hat = pi_hat
				# 	prev_gap = b - A.dot(pi_hat)
				# 	prev_lamda = lamda


def bipartite_workload_decomposition(Q, lamda, mu, path=None):

    m = len(lamda)
    n = len(mu)

    if path is None:
        path = '\\Users\\dean.grosbard\\Dropbox\\Software3.0\\fss'

    lamda_sum = np.asscalar(lamda.sum())
    inputf = open('inputHPF.txt', 'w', 1)
    theta_max = 1
    theta_min = -100
    edges = set(zip(*Q.nonzero()))
    num_nodes = m + n + 2
    num_edges = (num_nodes-2) + len(edges)
    rn = range(0, m, 1)
    rm = range(m, m + n, 1)

    inputf.write('p ' + str(num_nodes) +
                 ' ' + str(num_edges) +
                 ' ' + str(theta_min) +
                 ' ' + str(theta_max) +
                 ' 0'+'\n')
    inputf.write('n 0 s'+'\n')
    inputf.write('n ' + str(num_nodes-1) + ' t'+'\n')

    for i in rn:
        ub = lamda[i-1]
        inputf.write('a ' + '0' + ' ' + str(i) + ' ' + str(ub) + ' ' + '0.0' + '\n')

    for edge in sorted(list(edges)):

        i = edge[0] + 1
        j = edge[1] + m + 1

        inputf.write('a ' + str(i) + ' ' + str(j) + ' ' + str(lamda_sum) + ' ' + '0.0' + '\n')

    for j in rm:
        ub = float(mu[j - m - 1])
        coefficient = float(-1 * mu[j - m - 1])
        inputf.write('a ' + str(j) + ' ' + str(num_nodes-1) + ' ' + str(ub) + ' ' + str(coefficient) + '\n')

    inputf.close()

    _ = subprocess.call('./hpf inputHPF.txt outputHPF.txt', shell=True)
    outputf = open('outputHPF.txt', 'r+', 1)

    rho_m = [0.0] * m
    rho_n = [0.0] * n

    workload_sets = dict()
    bps = []

    for line in outputf:

        data = line.split()

        if data[0] == 'l':
            rank = 0
            for bp in data[1:-1]:
                workload_sets[rank] = \
                    {'rho': 1-float(bp),  'demnand_nodes': set(), 'supply_nodes': set()}
                bps.append(1-float(bp))
                rank += 1

        elif data[0] == 'n':
            node = int(data[1]) - 1

            if 0 < int(data[1]) < num_nodes - 1:
                singleton = True
                for i in range(3, len(data), 1):
                    if data[i] == '1':  # Check if at sum point it moves to the source set if not it is a singleton
                        rho = bps[i - 3]
                        # print(node, rho)
                        if node < m:
                            rho_m[node] = rho
                        else:
                            rho_n[node - m] = rho
                        if node in rn:
                            workload_sets[i-3]['demnand_nodes'].add(node)
                        if node in rm:
                            workload_sets[i-3]['supply_nodes'].add(node - m)
                        singleton = False
                        break
                if singleton:
                    print( node, 'single')
                    return None, None, None
    print( '-----------------')
    print( 'rho_n', rho_n)
    print( '-----------------')

    return workload_sets, np.array(rho_m), np.array(rho_n)



































