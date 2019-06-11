import numpy as np
import pandas as pd
from numba import jit
from numba.extending import overload, register_jitable
from numba import types
from numba.compiler import Flags
import heapq as hq
from time import time
from copy import copy
from scipy import sparse as sps
import atexit
from jit_heap import *



# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# @profile
SIM_LENGTHS = {
    6 : 10**5,
    100: 2*(10**6),
    300: 7*(10**6),
    1000: 30*(10**6),
}


def simulate_matching_sequance(compatability_matrix, alpha, beta, prt=True, sims=30, sim_len=None, seed=None, sim_name='sim'):


    m = len(alpha)  # m- number of servers
    n = len(beta)  # n - number of customers 
    print(m)
    print('alpha.sum():',alpha.sum())
    print('beta.sum():', beta.sum())   

    if sim_len is None:
            sim_len = SIM_LENGTHS[m]

    print('sim_len:', sim_len)
    # compatability_matrix = compatability_matrix.todense().A

    sparse = sps.issparse(compatability_matrix)

    matching_rates = []
    if sparse:
        s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))
        c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[1]) for i in range(m))  # adjacency list for customers
    else:        
        s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
        c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    for k in range(sims):

        if seed is not None:
            np.random.seed(seed+k)

        start_time = time()
        if prt:
            print('starting_sim ', k)
        
        customer_sizes = np.array([int((sim_len + sim_len//5) * alpha[i]) for i in range(m)], dtype=np.int32)
        server_sizes = np.array([int((sim_len + sim_len//5) * beta[j]) for j in range(n)], dtype=np.int32)
        customer_stream = np.hstack([i * np.ones(customer_sizes[i], dtype=np.int32) for i in range(m)])
        server_stream = np.hstack([j * np.ones(server_sizes[j], dtype=np.int32) for j in range(n)])
        np.random.shuffle(customer_stream)
        np.random.shuffle(server_stream)
        full_len = min(len(customer_stream), len(server_stream))
        server_stream = np.array(server_stream[ : full_len])
        customer_stream = np.array(customer_stream[: full_len])
        event_stream = np.vstack([customer_stream, server_stream]).T

        server_queues = tuple([int(1)] for i in range(m))
        customer_queues = tuple([int(1)] for j in range(n))

        matching_rates_k = matching_sim_loop_pairs(customer_queues, server_queues, event_stream, s_adj, c_adj, m, n, sim_len)    
        
        matching_rates.append(matching_rates_k)

        if prt:
            print('ending sim ', k, 'duration:', time() - start_time)

    nnz = compatability_matrix.nonzero()        
    matching_rates_mean = sum(matching_rates)*(1.0/sims)
    if sims > 1:
        matching_rates_stdev = (sum((matching_rates[k]-matching_rates_mean)**2 for k in range(sims))*(1.0/(sims-1)))**0.5
    else:
        matching_rates_stdev = 0 * matching_rates_mean
    return matching_rates_mean, matching_rates_stdev

#@jit(nopython=True)

#@profile
# jit(nopython=True)
def create_queues(event_stream, customer_queues, server_queues, m, n, sim_len):

    

    for i in range(m):
        customer_queues[i].pop()
    for j in range(n):
        server_queues[j].pop()

    for k in range(sim_len):

        i = event_stream[k, 0]
        j = event_stream[k, 1]
        # print(i,j)
        customer_queues[i].append(k)
        server_queues[j].append(k)
        # print(customer_queues)
        # print(server_queues)

    for i in range(m):
        customer_queues[i].append(sim_len + 1)
    for j in range(n):
        server_queues[j].append(sim_len + 1)

    return customer_queues, server_queues

# jit(nopython=True)
def create_queues_heap(event_stream, customer_queues, server_queues, m, n, sim_len):

    

    for i in range(m):
        heapify(customer_queues[i])
        heappop(customer_queues[i])
    for j in range(n):
        heapify(server_queues[j])
        heappop(server_queues[j])

    for k in range(sim_len):

        i = event_stream[k, 0]
        j = event_stream[k, 1]
        # print(i,j)
        heappush(customer_queues[i],k)
        heappush(server_queues[j], k)


    for i in range(m):
        heappush(customer_queues[i], sim_len + 1)
    for j in range(n):
        heappush(server_queues[j], sim_len + 1)


@jit(nopython=True, cache=True)
def matching_sim_loop_pairs(customer_queues, server_queues, event_stream, s_adj, c_adj, m, n, sim_len):


    matching_counter = np.zeros((m,n))


    for i in range(m):

        customer_queues[i].pop(0)

    for j in range(n):

        server_queues[j].pop(0)
    
    for cur_time in range(sim_len):

        
        i = event_stream[cur_time, 0]
        j = event_stream[cur_time, 1]
        # print(i,j)

        if customer_queues[i]:
            customer_queues[i].append(cur_time)
        
        if server_queues[j]:
            server_queues[j].append(cur_time)
        
        if not customer_queues[i]:

            last_arrived = cur_time
            m_j = -1

            for s_j in c_adj[i]:
                if server_queues[s_j]:
                    if server_queues[s_j][0] <= last_arrived:
                        m_j = s_j
                        last_arrived = server_queues[s_j][0]
            if m_j >= 0:
                matching_counter[i, m_j] = matching_counter[i, m_j] + 1
                server_queues[m_j].pop(0)
            else:
                customer_queues[i].append(cur_time)

        if not server_queues[j]:

            last_arrived = cur_time
            m_i = -1

            for c_i in s_adj[j]:
                if customer_queues[c_i]:
                    if customer_queues[c_i][0] <= last_arrived:
                        m_i = c_i
                        last_arrived = customer_queues[c_i][0]

            if m_i >= 0:
                matching_counter[m_i, j] = matching_counter[m_i, j] + 1
                customer_queues[m_i].pop(0)
            else:
                server_queues[j].append(cur_time)

        cur_time = cur_time + 1

    print(matching_counter.sum())

    return matching_counter/matching_counter.sum()


@jit(nopython=True)
def matching_sim_loop_pairs2(customer_queues, server_queues, event_stream, s_adj, c_adj, m, n, sim_len):

    server_arrived_idx = [-1] * n
    customer_arrived_idx = [-1] * m
    server_match_idx = [0] * n
    customer_match_idx = [0] * m

    matching_counter = np.zeros((m,n))
    
    for cur_time in range(sim_len):

        i = event_stream[cur_time, 0]
        j = event_stream[cur_time, 1]
        # print('arrival is ', i, j)

        customer_arrived_idx[i] = customer_arrived_idx[i] + 1
        server_arrived_idx[j] = server_arrived_idx[j] + 1
        # print('customer_arrived_idx[',i,']: ', customer_arrived_idx[i], ' customer_match_idx[',i,']: ', customer_match_idx[i])
        if customer_arrived_idx[i] == customer_match_idx[i]:

            last_arrived = cur_time
            m_j = -1

            for s_j in c_adj[i]:
                # print('s_',s_j, ' last arrived at ', server_queues[s_j][server_match_idx[s_j]])
                s_j_last_arrived = server_queues[s_j][server_match_idx[s_j]]
                if s_j_last_arrived <= last_arrived:
                    m_j = s_j
                    last_arrived = s_j_last_arrived

            if m_j >= 0:
                # print('match: ', m_j, j)
                matching_counter[i, m_j] = matching_counter[i, m_j] + 1
                customer_match_idx[i] = customer_match_idx[i] + 1
                server_match_idx[m_j] = server_match_idx[m_j] + 1


        # print('server_arrived_idx[',j,']: ', server_arrived_idx[j], ' server_match_idx[',j,']: ', server_match_idx[j])
        if server_arrived_idx[j] == server_match_idx[j]:

            last_arrived = cur_time
            m_i = -1

            for c_i in s_adj[j]:
                # print('c_',c_i, ' last arrived at ', customer_queues[c_i][customer_match_idx[c_i]])
                c_i_last_arrived = customer_queues[c_i][customer_match_idx[c_i]]
                if c_i_last_arrived <= last_arrived:
                    m_i = c_i
                    last_arrived = c_i_last_arrived

            if m_i >= 0:
                # print('match: ', m_i, j)
                matching_counter[m_i, j] = matching_counter[m_i, j] + 1
                server_match_idx[j] = server_match_idx[j] + 1
                customer_match_idx[m_i] = customer_match_idx[m_i] + 1

    # print(matching_counter)
    return matching_counter/matching_counter.sum()


@jit(nopython=True)
def matching_sim_loop(customer_stream, server_stream, s_adj, m, n):

    matching_rates = np.zeros((m, n))
    start_at_idx = [0]*n

    for s in server_stream:
        cur_idx = start_at_idx[s]
        for c in customer_stream[cur_idx:]:
            if c in s_adj[s]:
                matching_rates[c, s] = matching_rates[c, s] + 1
                customer_stream[cur_idx] = -1
                start_at_idx[s] = cur_idx + 1
                break
            cur_idx = cur_idx + 1

    matching_rates = matching_rates/matching_rates.sum()

    return matching_rates


def simulate_queueing_system(compatability_matrix, lamda, mu, s=None,  prt=False, sims=30, sim_len=None, warm_up=None, seed=None, sim_name='sim'):


    m = len(lamda)  # m- number of servers
    n = len(mu)  # n - number of customers    
    matching_rates = []
    waiting_times = []
    waiting_times_stdev = []

    if sim_len is None:
            sim_len = 1000 * m
            warm_up = 100 * m 

    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    if s is None:
        s = np.ones(m)
    
    service_rates = np.dot(np.dot(np.diag(s), compatability_matrix), np.diag(1./mu))

    start_time = time()
    
    for k in range(sims):

        if seed is not None:
            np.random.seed(seed + k)

        
        customer_classes = np.random.choice(a=range(m), size=sim_len, p=lamda/lamda.sum())
        arrival_times = np.random.exponential(scale=1./(lamda.sum()), size=sim_len).cumsum()
        event_stream = list(zip(arrival_times, customer_classes, (-1 * np.ones(sim_len)).astype(int)))
        service_times = np.random.exponential(service_rates, size=(sim_len, m, n))
        customer_queues = tuple([-1.] for i in range(m))

        matching_rates_k, waiting_k = queueing_sim_loop(customer_queues, event_stream, service_times, s_adj, c_adj, m, n, warm_up)

        waiting_k_mean = waiting_k[:, 1]/waiting_k[:, 0]
        waiting_k_stdev = ((waiting_k[:, 2] - ((waiting_k[:, 1]**2)/waiting_k[:, 0]))**0.5)/(waiting_k[:, 0] - 1)
        matching_rates.append(matching_rates_k)
        waiting_times.append(waiting_k_mean)
        waiting_times_stdev.append(waiting_k_stdev)

    if prt:
        print('ending sim ', k, 'duration:', time() - start_time)

    results = dict()
    nnz = compatability_matrix.nonzero()        
    results['matching_rates_mean'] = sum(matching_rates)*(1./sims)
    results['waiting_times_mean'] = sum(waiting_times)*(1./sims)
    results['waiting_times_stdev_mean'] = sum(waiting_times_stdev)*(1./sims)
    matching_rates_mean = results['matching_rates_mean']
    waiting_times_mean = results['waiting_times_mean']
    waiting_times_stdev_mean = results['waiting_times_stdev_mean']
    if sims > 1:
        results['matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5
        results['waiting_times_stdev'] = (sum((waiting_times[k] - waiting_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5
        results['waiting_times_stdev_stdev'] = (sum((waiting_times_stdev[k] - waiting_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5
    else:
        results['matching_rates_stdev'] = 0 * matching_rates_mean
        results['waiting_times_stdev'] = 0 * matching_rates_mean
        results['waiting_times_stdev_stdev'] = 0 * matching_rates_mean
    return results


@jit(nopython=True)
def queueing_sim_loop(customer_queues, event_stream, service_times, s_adj, c_adj, m, n, warm_up):

    matching_counter = np.zeros((m,n))
    idle_times = np.zeros((n, 3))
    waiting_times = np.zeros((m, 3))
    server_states = np.zeros(n, dtype=np.int8)
    server_idled_at = np.zeros(n, dtype=np.float64)
    service_time_idx = np.zeros((m, n), dtype=np.int32)
    matches = 0
    record = False
    record_stat_time = 0

    for i in range(m):
        heapify(customer_queues[i])
        heappop(customer_queues[i])

    heapify(event_stream)

    while event_stream:

        event = heappop(event_stream)


        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            if customer_queues[i]:
                customer_queues[i].append(cur_time)

            else:

                j = -1
                idled_time = cur_time

                for s_j in c_adj[i]:
                    if server_states[s_j] == 0:
                        if server_idled_at[s_j] < idled_time:
                            j = s_j
                            idled_time = server_idled_at[s_j]

                if j >= 0:

                    matches = matches + 1
                    if matches > warm_up:
                        if not record:
                            record = True
                            record_stat_time = cur_time
                        matching_counter[i, j] = matching_counter[i, j] + 1
                        waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
                        idle_time = cur_time - idled_time
                        idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
                    server_states[j] = 1
                    service_time = service_times[service_time_idx[i, j], i, j]
                    service_time_idx[i, j] = service_time_idx[i, j] + 1

                    heappush(event_stream, (cur_time + service_time, i, j))
                else:
                    customer_queues[i].append(cur_time)
        else:

            i = -1
            time = cur_time

            for c_i in s_adj[j]:
                if customer_queues[c_i]:
                    if customer_queues[c_i][0] < time:
                        i = c_i
                        time = customer_queues[c_i][0]

            if i >= 0:


                arrival_time = heappop(customer_queues[i])

                matches = matches + 1
                if matches > warm_up:
                    if not record:
                        record = True
                        record_stat_time = cur_time
                    matching_counter[i, j] = matching_counter[i,j] + 1
                    waiting_time = cur_time - arrival_time
                    waiting_times[i, :] = waiting_times[i, :] + np.array([1, waiting_time, waiting_time**2])
                service_time = service_times[service_time_idx[i, j], i, j]
                service_time_idx[i, j] = service_time_idx[i, j] + 1

                heappush(event_stream, (cur_time + service_time, i, j))
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/(cur_time - record_stat_time), waiting_times



