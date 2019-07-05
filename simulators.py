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
from utilities import printarr
import multiprocessing as mp

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

# @profile
SIM_LENGTHS = {
    6 : 10**5,
    10: 5*10**5,
    100: 2*(10**6),
    300: 7*(10**6),
    1000: 30*(10**6),
}


def simulate_matching_sequance(compatability_matrix, alpha, beta, prt=True, sims=30, sim_len=None, seed=None,  p=None, per_edge=1000):


    m = len(alpha)  # m- number of servers
    n = len(beta)  # n - number of customers 
    print(m)
    print('alpha.sum():',alpha.sum())
    print('beta.sum():', beta.sum())   

    edge_count = int(np.asscalar(compatability_matrix.sum()))

    if sim_len is None:
            sim_len = per_edge * edge_count
            sim_len = sim_len + int(sim_len // 10) 

    warm_up = int(sim_len // 10) 

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

    start_time = time()

    if p is None:
        for k in range(sims):

            if seed is not None:
                np.random.seed(seed + k)

            if prt:
                print('starting_sim ', k)
            
            # customer_sizes = np.array([int((sim_len + sim_len//5) * alpha[i]) for i in range(m)], dtype=np.int32)
            # server_sizes = np.array([int((sim_len + sim_len//5) * beta[j]) for j in range(n)], dtype=np.int32)
            # customer_stream = np.hstack([i * np.ones(customer_sizes[i], dtype=np.int32) for i in range(m)])
            # server_stream = np.hstack([j * np.ones(server_sizes[j], dtype=np.int32) for j in range(n)])
            # np.random.shuffle(customer_stream)
            # np.random.shuffle(server_stream)
            # full_len = min(len(customer_stream), len(server_stream))
            # server_stream = np.array(server_stream[ : full_len])
            # customer_stream = np.array(customer_stream[: full_len])
            # event_stream = np.vstack([customer_stream, server_stream]).T

            server_queues = tuple([int(1)] for i in range(m))
            customer_queues = tuple([int(1)] for j in range(n))

            matching_rates_k = matching_sim_loop_pairs3(customer_queues, server_queues, alpha, beta, s_adj, c_adj, m, n, sim_len, warm_up)    
            
            matching_rates.append(matching_rates_k)

            if prt:
                print('ending sim ', k, 'duration:', time() - start_time)
    else:

        print('starting parallel simulation')
        exps = []
        for k in range(sims):
            exps.append([tuple([int(1)] for i in range(m)), tuple([int(1)] for j in range(n)), alpha, beta, s_adj, c_adj, m, n, sim_len, warm_up])
        pool = mp.Pool(processes=p)
        matching_rates = pool.starmap(matching_sim_loop_pairs3, exps)
        print('ending parallel simulations after: ', time() - start_time)
        pool.close()
        pool.terminate()

    total_duration = time() - start_time

    nnz = compatability_matrix.nonzero()        

    results = {'mat': dict(), 'col': dict(), 'row': dict(), 'aux': dict()}

    nnz = compatability_matrix.nonzero()        
    
    results['mat']['sim_matching_rates'] = sum(matching_rates)*(1./sims)
    # results['row']['sim_waiting_times'] = sum(waiting_times)*(1./sims)
    # results['row']['sig_sim_waiting_times'] = sum(waiting_times_stdev)*(1./sims)
    # results['col']['sim_idle_times'] = sum(idle_times)*(1./sims)
    # results['col']['sig_sim_idle_times'] = sum(idle_times_stdev)*(1./sims)

    matching_rates_mean = results['mat']['sim_matching_rates']
    # waiting_times_mean = results['row']['sim_waiting_times']
    # waiting_times_stdev_mean = results['row']['sig_sim_waiting_times']
    # idle_times_mean = results['col']['sim_idle_times']
    # idle_times_stdev_mean = results['col']['sig_sim_idle_times']

    results['mat']['sim_matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    # results['row']['sim_waiting_times_stdev'] = (sum((waiting_times[k] - waiting_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    # results['row']['sig_sim_waiting_times_stdev'] = (sum((waiting_times_stdev[k] - waiting_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    # results['col']['sim_idle_times_stdev'] = (sum((idle_times[k] - idle_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    # results['col']['sig_sim_idle_times_stdev'] = (sum((idle_times_stdev[k] - idle_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    
    results['aux']['no_of_sims'] = sims
    results['aux']['sim_duration'] = total_duration
    results['aux']['sim_len'] = sim_len
    results['aux']['warm_up'] = warm_up
    results['aux']['seed'] = seed

    return results

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


@jit(nopython=True, cache=True)
def matching_sim_loop_pairs3(customer_queues, server_queues, alpha, beta, s_adj, c_adj, m, n, sim_len, warm_up):


    matching_counter = np.zeros((m,n))
    cur_time = 0
    r_n = np.arange(n)
    r_m = np.arange(m)


    for c_i in r_m:

        customer_queues[c_i].pop(0)

    for s_j in r_n:

        server_queues[s_j].pop(0)
    
    for cur_time in range(sim_len):

        
        i = np.random.choice(a=r_m)
        j = np.random.choice(a=r_n)
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
                if cur_time > warm_up:
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
                if cur_time > warm_up:
                    matching_counter[m_i, j] = matching_counter[m_i, j] + 1
                customer_queues[m_i].pop(0)
            else:
                server_queues[j].append(cur_time)

        cur_time = cur_time + 1

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


def simulate_queueing_system(compatability_matrix, lamda, mu, s=None,  prt=False, sims=30, sim_len=None, warm_up=None, seed=None, sim_name='sim', per_edge=1000, prt_all=False):


    m = len(lamda)  # m- number of servers
    n = len(mu)  # n - number of customers    
    matching_rates = []
    waiting_times = []
    idle_times = []
    waiting_times_stdev = []
    idle_times_stdev = []
    edge_count = int(np.asscalar(compatability_matrix.sum()))

    if sim_len is None:
            sim_len = per_edge * edge_count
            sim_len = sim_len + int(sim_len // 10) 

    warm_up = int(sim_len // 10) 

    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    if s is None:
        s = np.ones(m)
    
    service_rates = np.dot(np.dot(np.diag(s), compatability_matrix), np.diag(1./mu))

    start_time = time()

    if prt:
        print('sim length: ', sim_len, ' warm_up_period: ', warm_up)    
    
    for k in range(1, sims+1, 1):


        customer_queues = tuple([-1.] for i in range(m))
        if prt_all:
            print('starting sim ', k, ' {:.4f} '.format(time() - start_time))

        start_time_k = time()
        
        if seed is not None:
            np.random.seed(seed + k)
        
        event_stream = [(-1., -1, -1)]
        matching_rates_k, waiting_k, idle_k = queueing_sim_loop(customer_queues, event_stream, lamda, s, mu, s_adj, c_adj, m, n, warm_up, sim_len)

        matching_rates.append(matching_rates_k)

        for data_name, data_k, data_sum, data_stdev  in zip(['waiting', 'idle'], [waiting_k, idle_k], [waiting_times, idle_times], [waiting_times_stdev, idle_times_stdev]):
            data_k_mean = np.divide(data_k[:, 1], data_k[:, 0], out=np.zeros_like(data_k[:, 1]), where=data_k[:, 0]!=0)
            sq_data_k_mean = np.divide(data_k[:, 1]**2, data_k[:, 0], out=np.zeros_like(data_k[:, 1]), where=data_k[:, 0]!=0)
            data_k_n_m_1 = data_k[:, 0] - 1. 
            data_k_stdev = np.divide((data_k[:, 2] - sq_data_k_mean)**0.5, data_k_n_m_1, out=np.zeros_like(data_k_n_m_1), where=data_k_n_m_1!=0)
            data_sum.append(data_k_mean)
            data_stdev.append(data_k_stdev)


        if prt_all:
            print('ending sim ', k, 'duration: {:.4f} , {:.4f}'.format(time() - start_time_k ,time() - start_time))
    
    total_duration = time() - start_time
    if prt:

        print('ending ',k, ' sims. duration: {:.4f}'.format(total_duration))

    results = {'mat': dict(), 'col': dict(), 'row': dict(), 'aux': dict()}

    nnz = compatability_matrix.nonzero()        
    
    results['mat']['sim_matching_rates'] = sum(matching_rates)*(1./sims)
    results['row']['sim_waiting_times'] = sum(waiting_times)*(1./sims)
    results['row']['sig_sim_waiting_times'] = sum(waiting_times_stdev)*(1./sims)
    results['col']['sim_idle_times'] = sum(idle_times)*(1./sims)
    results['col']['sig_sim_idle_times'] = sum(idle_times_stdev)*(1./sims)

    matching_rates_mean = results['mat']['sim_matching_rates']
    waiting_times_mean = results['row']['sim_waiting_times']
    waiting_times_stdev_mean = results['row']['sig_sim_waiting_times']
    idle_times_mean = results['col']['sim_idle_times']
    idle_times_stdev_mean = results['col']['sig_sim_idle_times']

    results['mat']['sim_matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    results['row']['sim_waiting_times_stdev'] = (sum((waiting_times[k] - waiting_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    results['row']['sig_sim_waiting_times_stdev'] = (sum((waiting_times_stdev[k] - waiting_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    results['col']['sim_idle_times_stdev'] = (sum((idle_times[k] - idle_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    results['col']['sig_sim_idle_times_stdev'] = (sum((idle_times_stdev[k] - idle_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    
    results['aux']['no_of_sims'] = sims
    results['aux']['sim_duration'] = total_duration
    results['aux']['sim_len'] = sim_len
    results['aux']['warm_up'] = warm_up
    results['aux']['seed'] = seed


    return results


@jit(nopython=True, cache=True)
def queueing_sim_loop(customer_queues, event_stream, lamda, s, mu, s_adj, c_adj, m, n, warm_up, sim_len):

    # heapify = hq.heapify
    # heappop = hq.heappop
    # heappush = hq.heappush


    matching_counter = np.zeros((m,n))
    idle_times = np.zeros((n, 3))
    waiting_times = np.zeros((m, 3))
    server_states = np.zeros(n, dtype=np.int8)
    server_idled_at = np.zeros(n, dtype=np.float64)
    interarrival = 1./lamda
    # service_time_idx = np.zeros((m, n), dtype=np.int32)
    matches = 0
    arrivals = 0
    record = False
    record_stat_time = 0

    heapify(event_stream)
    heappop(event_stream)

    for i in range(m):
        heapify(customer_queues[i])
        heappop(customer_queues[i])

    for i in range(m):
        heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    while arrivals < sim_len:

        event = heappop(event_stream)


        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            arrivals = arrivals + 1

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
                    service_time = np.random.exponential(s[i]/mu[j])
                    heappush(event_stream, (cur_time + service_time, i, j))
                
                else:
                    customer_queues[i].append(cur_time)

            heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, -1))
        
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

                service_time = np.random.exponential(s[i]/mu[j])
                heappush(event_stream, (cur_time + service_time, i, j))
            
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/(cur_time - record_stat_time), waiting_times, idle_times


def ball_stacks(compatability_matrix, alpha, stacks, sim_len, warm_up):

    m, n = compatability_matrix.shape

    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    matching_rates = np.zeros((m,n))

    state = np.vstack([np.random.permutation(n) for _ in range(stacks)]).astype(int)

    state = np.random.permutation(state.ravel()).reshape((stacks, n))

    for k in range(sim_len):

        # print('state')
        # printarr(state)

        unmatched = True
        i = np.random.choice(n, p=alpha)

        while unmatched:
            
            stack = np.random.randint(0, stacks)
            # print('customer: ', i, ' stack: ', stack)
            for idx in range(n):
                if state[stack, idx] in c_adj[i]:
                    j = state[stack, idx]
                    unmatched = False
                    # print('matched customer: ', i, 'with server: ', j)
                    if k > warm_up:
                        matching_rates[i, j] = matching_rates[i, j] + 1 
                    if idx < n-1:
                        state[stack, idx:n-1] = state[stack, idx + 1:n]
                        state[stack, -1] = j
                    break

    return matching_rates/matching_rates.sum()


def ball_stacks2(compatability_matrix, alpha, stacks, sim_len, warm_up, start=None):

    m, n = compatability_matrix.shape

    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    matching_rates = np.zeros((m,n))

    if start is None:
        state = np.vstack([np.random.permutation(n) for _ in range(stacks)]).astype(int)
        state = np.random.permutation(state.ravel()).reshape((n, stacks))
        state = np.vstack(((state == j).sum(axis=1) for j in range(n)))
    elif start == 'uniform':
        state = np.ones((n,n)) * (stacks//n)

    else:
        state = np.eye(n) * stacks



    # printarr(state, 'initial state is:')

    sum_state = np.zeros((n,n))
    sum_state_sq = np.zeros((n,n))

    matches = 0
    ss = time()
    for k in range(sim_len):

        if k % 100000 == 0:
            print('at iteration {} after {}'.format(k, time() - ss))
        # print('state at time {}'.format(k))
        # printarr(state, 'state')
        if k > warm_up:
            sum_state += state
            sum_state_sq += state * state
        # printarr(sum_state,'mean_state')
        # printarr(sum_state_sq,'mean_state_sq')

        matched = False
        i = np.random.choice(n, p=alpha)
        # print('customer: ', i)
        while not matched:
            for l in range(n):
                j = np.random.choice(n , p=state[l, :]/stacks)
                
                # print('at layer {} server choosen is {}'.format(l, j))
                
                if j in c_adj[i]:

                    # print('match customer {} with server {} at level {}'.format(i,j,l))
                    if k > warm_up:
                        matching_rates[i, j] += 1
                    # if k%100 == 0 and k > warm_up:
                    
                    
                    state[l, j] -= 1
                    # print('server {} is removed form layer {}'.format(j, l))
                    if l < n-1:

                        up = [(np.random.choice(n, p=state[h,:]/stacks), h) for h in range(n-1, l, -1)]
                        # for v,h in up:
                            # print('one server {} is moving up from level {} to level {}'.format(v, h, h-1))
                        
                        for v,h in up:

                            state[h, v] -= 1
                            state[h-1, v] += 1

                    # print('server {} is added to layer {}'.format(j, n-1))
                    state[n-1, j] += 1
                    matched=True
                    break
    w = sim_len - warm_up
    mean_state = sum_state/w
    mean_state_stdev = ((sum_state_sq - (sum_state*sum_state)/w)/(w -1))**0.5

    return matching_rates/matching_rates.sum(), mean_state, mean_state_stdev


def ball_stacks3(compatability_matrix, lamda, mu, stacks, sim_len, warm_up, s=None):

    m, n = compatability_matrix.shape

    if s is None:
        s = np.ones(n)
    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m)) 

    matching_rates = np.zeros((m,n))

    state = np.vstack([np.random.permutation(n) for _ in range(stacks)]).astype(int)

    state = np.random.permutation(state.ravel()).reshape((n, stacks))

    state = np.vstack(((state == j).sum(axis=1) for j in range(n)))
    state = np.hstack((state, np.zeros((n, n))))

    sum_state = np.zeros((n,2*n))
    sum_state_sq = np.zeros((n,2*n))

    lost = np.zeros(m)

    interarrival = 1./lamda

    event_stream = []    

    for i in range(m):
        hq.heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    matches = 0
    ss = time()

    while matches < sim_len:

        event = hq.heappop(event_stream)
        cur_time = event[0]
        i = event[1]
        j = event[2]


        if j == -1:

            matched = False

        # print('customer: ', i)
            for _ in range(100):

                for l in range(n):

                    j = np.random.choice(2*n, p=state[l, :]/stacks)
                    
                    if j in c_adj[i]:

                        # print('match customer {} with server {} at level {}'.format(i,j,l))
                        matches += 1
                        if matches > warm_up:
                            matching_rates[i, j] += 1
                            sum_state += state
                            sum_state_sq += state * state
                        # if k%100 == 0 and k > warm_up:
                        
                        state[l, j] -= 1
                        state[l, n + j] += 1
                        service_time = np.random.exponential(s[i]/mu[j])
                        hq.heappush(event_stream, (cur_time + service_time, i, j))
                        
                        if matches % 10000 == 0:
                            print('at iteration {} after {}'.format(matches, time() - ss))
                        
                        # print('state at time {}'.format(k))
                        # printarr(state, 'state')
                        # printarr(sum_state,'mean_state')
                        # printarr(sum_state_sq,'mean_state_sq')

                        # print('server {} is removed form layer {}'.format(j, l))
                        break
                        
            else:
                lost[i] +=1

            hq.heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, -1))

        else:
            
            l = np.random.choice(n , p=state[:, n + j]/state[:, n + j].sum())
            

            if l < n-1:

                up = [(np.random.choice(2*n, p=state[h, :]/stacks), h) for h in range(n-1, l, -1)]
                # for v,h in up:
                    # print('one server {} is moving up from level {} to level {}'.format(v, h, h-1))
                
                for v,h in up:

                    state[h, v] -= 1
                    state[h-1, v] += 1

            state[l, n + j] -= 1
            state[n-1, j] += 1

            # print('server {} is added to layer {}'.format(j, n-1))
        
    w = sim_len - warm_up
    mean_state = sum_state/w
    mean_state_stdev = ((sum_state_sq - (sum_state*sum_state)/w)/(w -1))**0.5

    print(lost)

    return matching_rates/matching_rates.sum(), mean_state, mean_state_stdev
            


def ball_stacks4(compatability_matrix, lamda, mu, bpl, sim_len, warm_up, s=None):

    m, n = compatability_matrix.shape

    matching_rates = np.zeros((m, n))

    if s is None:
        s = np.ones(n)
    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m))

    servers = [j for j in range(n) for _ in range(bpl)]
    server_ids = [j_id for _ in range(n) for j_id in range(bpl)]
    # print(servers)
    # print(server_ids)

    levels = [l for l in range(n) for _ in range(bpl)]
    lev_loc = [l_id for _ in range(n) for l_id in range(bpl)]
    # print(levels)
    # print(lev_loc)

    servers = list(zip(servers, server_ids))
    locations = np.random.permutation(list(zip(levels, lev_loc)))

    server_states = dict()
    layer_states = dict((l, dict()) for l in range(n))

    for (j, j_id), (l, l_id)  in zip(servers, locations):
        # print((j, j_id), (l, l_id))
        server_states[j, j_id] = (l, l_id, -1)
        layer_states[l][l_id] = (j, j_id, -1)
    #     for l, layer in layer_states.items():
    #         print(l, ': ', [loc for loc in layer])
    # for l, layer in layer_states.items():
    #     print(l, ': ', [loc for loc in layer])

    lost = np.zeros(m)

    interarrival = 1./lamda

    event_stream = []    

    for i in range(m):
        hq.heappush(event_stream, (np.random.exponential(interarrival[i]), i, (-1, -1)))

    matches = 0
    ss = time()

    while matches < sim_len:

        event = hq.heappop(event_stream)
        print(event)
        cur_time = event[0]
        i = event[1]
        j, j_id = event[2]

        if j == -1:

            matched = False

            for _ in range(10):

                for l in range(n):

                    l_id = np.random.randint(bpl)
                    j, j_id, j_state = layer_states[l][l_id]
                    
                    print('customer {} at layer {} got server {}, {} at state {} at location {}'.format(i, l, j, j_id, j_state, l_id))
                    if j in c_adj[i]:
                        if j_state == -1:
                            matched = True
                            print('match customer {} with server {} at level {}'.format(i,j,l))
                            matches += 1
                            if matches > warm_up:
                                matching_rates[i, j] += 1

                            j_in, j_in_id, j_in_state = j, j_id, i

                            for k in range(n-1, l, -1):

                                k_id = np.random.randint(bpl)
                                j_out, j_out_id, j_out_state = layer_states[k][k_id]

                                layer_states[k][k_id] = (j_in, j_in_id, j_in_state)
                                server_states[j_in, j_in_id] = (k, k_id, j_in_state)

                                j_in, j_in_id, j_in_state = j_out, j_out_id, j_out_state

                            
                            layer_states[l][l_id] = (j, j_id, i)
                            server_states[j, j_id] = (l, l_id, i)

                            service_time = np.random.exponential(s[i]/mu[j])
                            print(service_time)
                            print('server {} {} will finish serving customer {} at time {}'.format(j, j_id, i, cur_time + service_time,))
                            hq.heappush(event_stream, (cur_time + service_time, -1, (j, j_id)))
                            if matches % 10000 == 0:
                                print('at iteration {} after {}'.format(matches, time() - ss))
                        
                        # print('state at time {}'.format(k))
                        # printarr(state, 'state')
                        # printarr(sum_state,'mean_state')
                        # printarr(sum_state_sq,'mean_state_sq')

                        # print('server {} is removed form layer {}'.format(j, l))
                            break
                if matched:
                    break
                        
            else:
                lost[i] +=1

            hq.heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, (-1, -1)))

        else:
            
            l, l_id, j_state = server_states[j, j_id]
            server_states[j, j_id] = (l, l_id, -1)
            layer_states[l, l_id] = (j, j_id, -1)

            # print('server {} is added to layer {}'.format(j, n-1))
        
    # w = sim_len - warm_up
    # mean_state = sum_state/w
    # mean_state_stdev = ((sum_state_sq - (sum_state*sum_state)/w)/(w -1))**0.5

    print(lost)

    return matching_rates/matching_rates.sum()


def ball_stacks5(compatability_matrix, lamda, mu, bpl, sim_len, warm_up, s=None):

    m, n = compatability_matrix.shape

    matching_rates = np.zeros((m, n))

    if s is None:
        s = np.ones(n)
    s_adj = tuple(set(np.nonzero(compatability_matrix[:, j])[0]) for j in range(n))  # adjacency list for servers
    c_adj = tuple(set(np.nonzero(compatability_matrix[i, :])[0]) for i in range(m))

    servers = [j for j in range(n) for _ in range(bpl)]
    server_ids = [j_id for _ in range(n) for j_id in range(bpl)]
    # print(servers)
    # print(server_ids)

    levels = [l for l in range(n) for _ in range(bpl)]
    lev_loc = [l_id for _ in range(n) for l_id in range(bpl)]
    # print(levels)
    # print(lev_loc)

    servers = list(zip(servers, server_ids))
    locations = np.random.permutation(list(zip(levels, lev_loc)))

    server_states = dict()
    layer_states = dict((l, dict()) for l in range(n))

    for (j, j_id), (l, l_id)  in zip(servers, locations):
        # print((j, j_id), (l, l_id))
        server_states[j, j_id] = (l, l_id, -1)
        layer_states[l][l_id] = (j, j_id, -1)
    #     for l, layer in layer_states.items():
    #         print(l, ': ', [loc for loc in layer])
    # for l, layer in layer_states.items():
    #     print(l, ': ', [loc for loc in layer])

    lost = np.zeros(m)

    interarrival = 1./lamda

    event_stream = []    

    for i in range(m):
        hq.heappush(event_stream, (np.random.exponential(interarrival[i]), i, (-1, -1)))

    matches = 0
    ss = time()

    while matches < sim_len:

        event = hq.heappop(event_stream)
        print(event)
        cur_time = event[0]
        i = event[1]
        j, j_id = event[2]

        if j == -1:

            matched = False

            for _ in range(10):

                for l in range(n):

                    l_id = np.random.randint(bpl)
                    j, j_id, j_state = layer_states[l][l_id]
                    
                    print('customer {} at layer {} got server {}, {} at state {} at location {}'.format(i, l, j, j_id, j_state, l_id))
                    if j in c_adj[i]:
                        if j_state == -1:
                            matched = True
                            print('match customer {} with server {} at level {}'.format(i,j,l))
                            matches += 1
                            if matches > warm_up:
                                matching_rates[i, j] += 1

                            j_in, j_in_id, j_in_state = j, j_id, i

                            for k in range(n-1, l, -1):

                                k_id = np.random.randint(bpl)
                                j_out, j_out_id, j_out_state = layer_states[k][k_id]

                                layer_states[k][k_id] = (j_in, j_in_id, j_in_state)
                                server_states[j_in, j_in_id] = (k, k_id, j_in_state)

                                j_in, j_in_id, j_in_state = j_out, j_out_id, j_out_state

                            
                            layer_states[l][l_id] = (j, j_id, i)
                            server_states[j, j_id] = (l, l_id, i)

                            service_time = np.random.exponential(s[i]/mu[j])
                            print(service_time)
                            print('server {} {} will finish serving customer {} at time {}'.format(j, j_id, i, cur_time + service_time,))
                            hq.heappush(event_stream, (cur_time + service_time, -1, (j, j_id)))
                            if matches % 10000 == 0:
                                print('at iteration {} after {}'.format(matches, time() - ss))
                        
                        # print('state at time {}'.format(k))
                        # printarr(state, 'state')
                        # printarr(sum_state,'mean_state')
                        # printarr(sum_state_sq,'mean_state_sq')

                        # print('server {} is removed form layer {}'.format(j, l))
                            break
                if matched:
                    break
                        
            else:
                lost[i] +=1

            hq.heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, (-1, -1)))

        else:
            
            l, l_id, j_state = server_states[j, j_id]
            server_states[j, j_id] = (l, l_id, -1)
            layer_states[l, l_id] = (j, j_id, -1)

            # print('server {} is added to layer {}'.format(j, n-1))
        
    # w = sim_len - warm_up
    # mean_state = sum_state/w
    # mean_state_stdev = ((sum_state_sq - (sum_state*sum_state)/w)/(w -1))**0.5

    print(lost)

    return matching_rates/matching_rates.sum()





# @jit(nopython=True, cache=True)
# def queueing_sim_loop_old(customer_queues, event_stream, service_times, s_adj, c_adj, m, n, warm_up):

#     matching_counter = np.zeros((m,n))
#     idle_times = np.zeros((n, 3))
#     waiting_times = np.zeros((m, 3))
#     server_states = np.zeros(n, dtype=np.int8)
#     server_idled_at = np.zeros(n, dtype=np.float64)
#     service_time_idx = np.zeros((m, n), dtype=np.int32)
#     matches = 0
#     record = False
#     record_stat_time = 0

#     for i in range(m):
#         heapify(customer_queues[i])
#         heappop(customer_queues[i])

#     heapify(event_stream)

#     while event_stream:

#         event = heappop(event_stream)


#         cur_time = event[0]
#         i = event[1]
#         j = event[2]

#         if j == -1:

#             if customer_queues[i]:
#                 customer_queues[i].append(cur_time)

#             else:

#                 j = -1
#                 idled_time = cur_time

#                 for s_j in c_adj[i]:
#                     if server_states[s_j] == 0:
#                         if server_idled_at[s_j] < idled_time:
#                             j = s_j
#                             idled_time = server_idled_at[s_j]

#                 if j >= 0:

#                     matches = matches + 1
#                     if matches > warm_up:
#                         if not record:
#                             record = True
#                             record_stat_time = cur_time
#                         matching_counter[i, j] = matching_counter[i, j] + 1
#                         waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
#                         idle_time = cur_time - idled_time
#                         idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
#                     server_states[j] = 1
#                     service_time = service_times[service_time_idx[i, j], i, j]
#                     service_time_idx[i, j] = service_time_idx[i, j] + 1

#                     heappush(event_stream, (cur_time + service_time, i, j))
#                 else:
#                     customer_queues[i].append(cur_time)
#         else:

#             i = -1
#             time = cur_time

#             for c_i in s_adj[j]:
#                 if customer_queues[c_i]:
#                     if customer_queues[c_i][0] < time:
#                         i = c_i
#                         time = customer_queues[c_i][0]

#             if i >= 0:


#                 arrival_time = heappop(customer_queues[i])

#                 matches = matches + 1
#                 if matches > warm_up:
#                     if not record:
#                         record = True
#                         record_stat_time = cur_time
#                     matching_counter[i, j] = matching_counter[i,j] + 1
#                     waiting_time = cur_time - arrival_time
#                     waiting_times[i, :] = waiting_times[i, :] + np.array([1, waiting_time, waiting_time**2])
#                 service_time = service_times[service_time_idx[i, j], i, j]
#                 service_time_idx[i, j] = service_time_idx[i, j] + 1

#                 heappush(event_stream, (cur_time + service_time, i, j))
#             else:
#                 server_states[j] = 0
#                 server_idled_at[j] = cur_time
#     print(matching_counter.sum())
#     return matching_counter/(cur_time - record_stat_time), waiting_times, idle_times