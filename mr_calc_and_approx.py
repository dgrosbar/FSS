import numpy as np
from numba import jit
import cplex
from cplex.exceptions import CplexError
from math import exp
from time import time
from numpy import ma
from utilities import printarr
from scipy import sparse as sps


def adan_weiss_fcfs_alis_matching_rates(compatability_matrix, alpha, beta, jt_perms=None, print_progress=False):


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

def quadratic_approximation(compatability_matrix, alpha, beta, prt=False):

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


def entropy_approximation(compatability_matrix, lamda, mu, check_every=10**2, max_iter=10**7, epsilon=10**-9, pad=False):

	k = 0
	within_epsilon = True


	if  sps.isspmatrix(compatability_matrix):

		if pad:
			compatability_matrix = sps.vstack([compatability_matrix, np.ones(len(mu))])
			lamda = np.append(lamda, mu.sum() - lamda.sum())

		matching_rates = compatability_matrix

		for k in range(max_iter):
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
					if pad:
						return True, matching_rates[:-1, :], gap_pct
					else:
						return True, matching_rates, gap_pct

	else:
		if pad:
			compatability_matrix = np.vstack([compatability_matrix, np.ones(len(mu))])
			lamda = np.append(lamda, mu.sum() - lamda.sum())
		
		matching_rates = compatability_matrix

		for k in range(max_iter):
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
					if pad:
						return True, matching_rates[:-1, :], gap_pct
					else:
						return True, matching_rates, gap_pct

		else:
			return False, matching_rates, gap_pct 


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


def local_entropy(compatability_matrix, lamda, mu, prt=False):


	m, n  = compatability_matrix.shape
	k = m + n
	l = (m + 1) * n

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
			cols.append(i * n + j)
			data.append(lamda[i]/mu[j])

			col_set.add(i * n + j)

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


def fast_primal_dual_algorithm(A, b, z, m, n, pi0=None, act_rows=None , check_every=10**3, max_iter=10**6, epsilon=10**-6, prt=True, prtall=False):

	m_p_n_p_1, m_p_1_t_n = A.shape
	pi_k = np.zeros((m_p_1_t_n, ))
	pi_hat = np.zeros((m_p_1_t_n, ))
	prev_pi_hat = np.zeros((m_p_1_t_n, ))
	prev_gap = np.zeros((m_p_n_p_1,))
	lamda = np.zeros((m_p_n_p_1, ))
	prev_lamda = np.zeros((m_p_n_p_1, ))
	zeta = np.zeros((m_p_n_p_1, ))
	ze = z * exp(-1.0)
	v = np.amin(z[np.where(z > 0)])

	def f(pi):

		res = np.divide(pi, z, out=np.zeros_like(pi), where= z!=0)
		res = pi * ma.log(res).filled(0)

		return res.sum()

	def check_optimality_gap():

			pi_eta = ze*np.exp(-1*At.dot(eta))
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
			return False, True
		if opt_gap_pct < epsilon:
			if feas_gap < epsilon:
				return True, False

		return False, False


	
	L = ((1.0/v) * (np.amax(np.abs(A[:m].sum(axis=1))) + np.amax(np.abs(A[m:].sum(axis=1)))))

	if prt:
		print('L', L)

	flag = False

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
		ze = z * exp(-1.0)
		v = np.amin(z[np.where(z > 0)])

		for i in np.arange(max_iter):

			alpha = (i + 1.0)/2.0
			tau = 2.0/(i+3.0)

			if i == 0 and prt:
				print('starting fast primal dual gradient descent')
				s = time()

			pi_k = pi0 if i == 0 and pi0 is not None else ze*np.exp(At.dot(lamda))
			
		# 	if 0 < i < 60:
		# 	    print('m: ', m)
		# 	    print('n: ', n)
		# 	    print('m + n + 1: ', m + n + 1, 'len(b): ', len(b))

		# 	    print(i)
		# 	    print('-------------------------------')
		# 	    printarr(prev_pi_hat.reshape((m + 1, n)), 'prev_pi_hat')
		# 	    printarr(pi_hat.reshape((m + 1, n)), 'pi_hat')

		# 	    printarr(np.dot(A, prev_pi_hat)[m: ], 'prev_col_sum')
		# 	    printarr(np.dot(A, pi_hat)[m: ], 'cur_col_sum')
		# 	    printarr(np.dot(A, prev_pi_hat)[ :m], 'prev_row_sum')
		# 	    printarr(np.dot(A, pi_hat)[ :m], 'cur_row_sum')
				
		# 	    # printarr(b[m: ], 'req_col_sum')
		# 	    # printarr(b[:m], 'req_row_sum')

		# 	    # printarr(prev_gap[m: ], 'prev_col_violation')
		# 	    # printarr(gap[m: ], 'cur_col_violation')

		# 	    # printarr(prev_gap[ :m], 'prev_row_violation')
		# 	    # printarr(gap[ :m], 'cur_row_violation')

		# 	    printarr(lamda[m: ], 'lamda_cols')
		# 	    printarr(prev_lamda[m: ], 'prev_lamda_cols')

		# 	    printarr(lamda[ :m], 'lamda_rows')
		# 	    printarr(prev_lamda[ :m], 'prev_lamda_rows')
	 
		# 	    prev_pi_hat = pi_hat
		# 	    prev_gap = b - A.dot(pi_hat)
		# 	    prev_lamda = lamda

			pi_k = ze * np.exp(-At.dot(lamda))
			pi_hat = tau * pi_k + (1.0 - tau) * pi_hat

			if (i > 0 and i % check_every == 0):
				converged, oob = check_stop(i, prt)
				if converged:
					flag=True
					break
				elif oob:
					flag=False
					print('oob feasibility gap halving step size')
					L = L*2
					break

			gap = b - A.dot(pi_k)

			eta = lamda - (1.0/L) * gap
			zeta = zeta - (alpha/L) * gap
			lamda = (tau * zeta) + (1.0 - tau) * eta

	if prt:
		print('ended fast primal-dual algorithm after ' + str(i) + ' iterations')
		print('run time:', time() - s, 'seconds')
	return pi_hat, lamda


# def get_pi_hat(compatability_matrix, lamda, mu, rho, c=None):

#     q = compatability_matrix

#     '''
#         lamda is a vector of length m
#         mu is a vector of length n
#         rho is a vector of length n or a scalar
#         q is a matrix of size m x n 
#     '''
#     def transform_to_normal_form(M, W, Q, Z, row_sum, col_sum):

#         if sps.isspmatrix(Q):
#             print('in sparse')
#             # print(Z.shape)
#             # print(M.shape)
#             # print(type(M))
#             # print(W.shape)
#             # print(sps.csr_matrix((np.exp(-1*M.data/W.data), M.indices, M.indptr)).shape)
#             Z_hat = Z.multiply(W).multiply(sps.csr_matrix((np.exp(-1*M.data/W.data), M.indices, M.indptr)))
#             Q_hat = sps.csr_matrix((Q.data/W.data, Q.indices, Q.indptr))
#             z = Z_hat.todense().A.ravel()

#         else:

#             Z_hat = Z * W * np.exp(-M/W)
#             Q_hat = Q/W
#             z = Z_hat.ravel()

#         A, b, col_set = metrize_constranits(Q_hat, row_sum, col_sum)

#         return A, b, z, list(col_set)


#     def metrize_constranits(Q, row_sum, col_sum, eq_double_vars=False, prt=False):

#         m = Q.shape[0]
#         n = Q.shape[1]

#         k = m + n
#         l = m * n

#         rows = []
#         cols = []
#         data = []
#         col_set = set()

#         for i, j in zip(*Q.nonzero()):
#                 if prt:
#                     print((i, j),'-->', (i, i * n + j), (m + j, i * n + j))
#                 rows.append(i)
#                 rows.append(m + j)
#                 cols.append(i * n + j)
#                 cols.append(i * n + j)
#                 col_set.add(i * n + j)
#                 data.append(Q[i, j])
#                 data.append(Q[i, j])

#         rows = np.array(rows)
#         cols = np.array(cols)
#         data = np.array(data)
#         A = sps.coo_matrix((data, (rows, cols)), shape=(k, l)).tocsr()
#         b = np.concatenate((row_sum, col_sum))

#         return A, b, col_set

#     m, n = q.shape
#     q_pad_sps = sps.csr_matrix(np.vstack([q, np.ones((1, n))]))
#     lamda_pad = np.append(lamda*rho, mu.sum() - rho*lamda.sum())
#     printarr(lamda_pad, 'lamda_pad')
#     w = q.dot(np.diag(np.ones(n) - rho))
#     w_pad = np.vstack((w, np.ones((1, n))*rho))
#     printarr(w_pad, 'w_pad')
#     w_pad_sps = sps.csr_matrix(w_pad)
#     c_pad_sps = sps.csr_matrix(np.vstack([c, np.zeros((1, n))])) if c is not None else 0*sps.csr_matrix(q_pad_sps) 
#     z_pad_sps = q_pad_sps.dot(sps.diags(mu))    
#     a,b,z, cols = transform_to_normal_form(c_pad_sps, w_pad_sps, q_pad_sps, z_pad_sps, lamda_pad, mu)
#     pi_hat, duals = fast_primal_dual_algorithm(a,b,z)
#     pi_hat = pi_hat.reshape((m+1, n))
#     pi_hat = np.divide(pi_hat, w_pad, out=np.zeros_like(pi_hat), where=w_pad != 0)

#     rho_hat = np.dot(pi_hat[:m, :].sum(axis=0), np.diag(1./mu))
#     lq_hat = pi_hat[:m, :]/pi_hat[m, :]

#     return pi_hat[:m, :], rho_hat, lq_hat