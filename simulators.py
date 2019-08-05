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
from math import ceil
import numba
from numba import types
from numba.typed import Dict, List


def simulate_matching_sequance(compatability_matrix, alpha, beta, prt=True, prt_all=False,sims=30, sim_len=None, seed=None,  p=None, per_edge=500):

    m = len(alpha)  # m- number of servers
    n = len(beta)  # n - number of customers 
    print(m)
    print('alpha.sum():',alpha.sum())
    print('beta.sum():', beta.sum())
    cum_alpha = alpha.cumsum()
    cum_beta = beta.cumsum()   

    edge_count = int(np.asscalar(compatability_matrix.sum()))
    print('sim_len: ', sim_len)
    if sim_len is None:
        sim_len = per_edge * edge_count
        sim_len = sim_len + int(sim_len // 10) 
        print(sim_len)

    warm_up = int(sim_len // 10) 

    print('sim_len:', sim_len)
    sparse = sps.issparse(compatability_matrix)
    matching_rates = []

    s_adj = tuple(List() for j in range(n)) 
    c_adj = tuple(List() for i in range(m))

    for i, j in zip(*compatability_matrix.nonzero()):
        s_adj[j].append(np.int16(i))
        c_adj[i].append(np.int16(j))

    start_time = time()
    if prt:
        print('starting matching simulations')
    if p is None:
        for k in range(sims):

            server_queues = tuple(List() for j in range(n))
            for j in range(n):
                server_queues[j].append(-1.)
            customer_queues = tuple(List() for i in range(m))
            for i in range(m):
                customer_queues[i].append(-1.)

            if seed is not None:
                np.random.seed(seed + k)

            if prt_all:
                print('starting_sim ', k)
            
            matching_rates_k = matching_sim_loop(customer_queues, server_queues, cum_alpha, cum_beta, s_adj, c_adj, m, n, sim_len, warm_up)    
            
            matching_rates.append(matching_rates_k)

            if prt_all:
                print('ending sim ', k, 'duration:', time() - start_time)
    else:
        if prt:
            print('starting parallel simulation')
        exps = []
        for k in range(sims):
            exps.append([customer_queues, server_queues, cum_alpha, cum_beta, s_adj, c_adj, m, n, sim_len, warm_up])
        pool = mp.Pool(processes=p)
        matching_rates = pool.starmap(matching_sim_loop_pairs, exps)
        if prt:
            print('ending parallel simulations after: ', time() - start_time)
        pool.close()
        pool.terminate()
    if prt:
        print('ending matching simulations: ', time() - start_time)

    total_duration = time() - start_time

    nnz = compatability_matrix.nonzero()        

    results = {'mat': dict(), 'col': dict(), 'row': dict(), 'aux': dict()}

    nnz = compatability_matrix.nonzero()        
    
    results['mat']['sim_matching_rates'] = sum(matching_rates)*(1./sims)


    matching_rates_mean = results['mat']['sim_matching_rates']

    results['mat']['sim_matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean


    aux_sim_data = {'no_of_sims' : sims, 'sim_duration' : total_duration, 'sim_len' : sim_len, 'warm_up' : warm_up, 'seed' : seed}

    return parse_sim_data(compatability_matrix, matching_rates, aux_sim_data=aux_sim_data)


@jit(nopython=True, cache=True)
def matching_sim_loop(customer_queues, server_queues, cum_alpha, cum_beta, s_adj, c_adj, m, n, sim_len, warm_up):


    matching_counter = np.zeros((m,n))
    cur_time = 0

    for i in range(m):

        customer_queues[i].pop(0)

    for j in range(n):

        server_queues[j].pop(0)
    
    for cur_time in range(sim_len):

        u_i = np.random.uniform(0, 1)
        u_j = np.random.uniform(0, 1)
        i = np.searchsorted(cum_alpha, u_i, 'left')
        j = np.searchsorted(cum_beta, u_j, 'left')

        if len(customer_queues[i]) > 0:
            customer_queues[i].append(cur_time)
        
        if len(server_queues[j]) > 0:
            server_queues[j].append(cur_time)
        
        if len(customer_queues[i]) == 0:

            last_arrived = cur_time
            m_j = -1

            for s_j in c_adj[i]:
                if len(server_queues[s_j]) > 0:
                    if server_queues[s_j][0] <= last_arrived:
                        m_j = s_j
                        last_arrived = server_queues[s_j][0]
            if m_j >= 0:
                if cur_time > warm_up:
                    matching_counter[i, m_j] = matching_counter[i, m_j] + 1
                server_queues[m_j].pop(0)
            else:
                customer_queues[i].append(cur_time)

        if len(server_queues[j])== 0:

            last_arrived = cur_time
            m_i = -1

            for c_i in s_adj[j]:
                if len(customer_queues[c_i])>0:
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


def simulate_queueing_system(compatability_matrix, lamda, mu, s=None, w=None, w_only=False, prt=False, sims=30, sim_len=None, warm_up=None, seed=None, per_edge=1000, prt_all=False, p=None, lqf=False):


    m = len(lamda)  # m- number of servers
    n = len(mu)  # n - number of customers    
    match_rates = []
    wait_times = []
    idle_times = []
    wait_times_stdev = []
    idle_times_stdev = []
    match_ratios = []
    edge_count = int(np.asscalar(compatability_matrix.sum()))

    if sim_len is None:
            sim_len = per_edge * edge_count
            sim_len = sim_len + int(sim_len // 10) 

    warm_up = int(sim_len // 10) 

    s_adj = tuple(List() for j in range(n)) 
    c_adj = tuple(List() for i in range(m))

    for i, j in zip(*compatability_matrix.nonzero()):
        s_adj[j].append(np.int16(i))
        c_adj[i].append(np.int16(j))

    if s is None:
        s = np.ones(m)

    
    service_rates = np.dot(np.dot(np.diag(s), compatability_matrix), np.diag(1./mu))

    start_time = time()

    if prt:
        print('sim length: ', sim_len, ' warm_up_period: ', warm_up)    
    
    if p is None:
        
        for k in range(1, sims + 1, 1):

            event_stream = [(-1., np.int16(-1), np.int16(-1))]
            customer_queues = tuple(List() for i in range(m))
            for i in range(m):
                customer_queues[np.int16(i)].append(-1.)
            
            if prt_all:
                print('starting sim ', k, ' {:.4f} '.format(time() - start_time))

            start_time_k = time()

            if seed is not None:
                np.random.seed(seed + k)
            if w is None:
                w = compatability_matrix
            
            if lqf: 
                match_ratios_k, match_rates_k, wait_k, idle_k = queueing_sim_loop_lqf(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len, w_only)
            else:
                match_ratios_k, match_rates_k, wait_k, idle_k = queueing_sim_loop(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len, w_only)


            match_rates.append(match_rates_k)
            match_ratios.append(match_ratios_k)
            wait_times, wait_times_stdev = get_mean_and_stdev(wait_times, wait_times_stdev, wait_k)
            idle_times, idle_times_stdev = get_mean_and_stdev(idle_times, idle_times_stdev, idle_k)

            if prt_all:
                print('ending sim ', k, 'duration: {:.4f} , {:.4f}'.format(time() - start_time_k ,time() - start_time))
        
        total_duration = time() - start_time
        
        if prt:
            print('ending ',k, ' sims. duration: {:.4f}'.format(total_duration))

    else:

        pool = mp.Pool(processes=p)

        if w is None:
            for k in range(sims):
                exps.append([customer_queues, event_stream, lamda, s, mu, s_adj, c_adj, m, n, warm_up, sim_len])
                reses = pool.starmap(queueing_sim_loop, exps)
        elif only_w:
            for k in range(sims):
                exps.append([customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len])
                reses = pool.starmap(queueing_sim_loop_with_only_w, exps)
        else:
            for k in range(sims):
                exps.append([customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len])
                reses = pool.starmap(queueing_sim_loop_with_w, exps)

        pool.terminate()


    aux_sim_data = {'no_of_sims' : sims, 'sim_duration' : total_duration, 'sim_len' : sim_len, 'warm_up' : warm_up, 'seed' : seed}
    
    return parse_sim_data(compatability_matrix, match_ratios, match_rates, wait_times, wait_times_stdev, idle_times, idle_times_stdev, aux_sim_data)


@jit(nopython=True, cache=True)
def queueing_sim_loop(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len, w_only):

    heapify = hq.heapify
    heappop = hq.heappop
    heappush = hq.heappush

    matching_counter = np.zeros((m,n))
    idle_times = np.zeros((n, 3))
    waiting_times = np.zeros((m, 3))
    server_states = np.zeros(n, dtype=np.int8)
    server_idled_at = np.zeros(n, dtype=np.float64)
    interarrival = 1./lamda
    matches = 0
    record = False
    record_stat_time = 0

    heapify(event_stream)
    heappop(event_stream)

    for i in range(m):
        customer_queues[i].pop()
        heappush(event_stream, (np.random.exponential(interarrival[i]), np.int16(i), np.int16(-1)))

    while matches < sim_len:

        event = heappop(event_stream)

        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            if len(customer_queues[i]) > 0:
                customer_queues[i].append(cur_time)

            else:

                j = np.int16(-1)
                longest_idle = 0
                for s_j in c_adj[i]:
                    if server_states[s_j] == 0:
                        if w_only:
                            if w[i, s_j] > longest_idle:
                                j = s_j
                                longest_idle = w[i, s_j]
                        else:
                            if w[i, s_j] * (cur_time - server_idled_at[s_j]) > longest_idle:
                                j = s_j
                                longest_idle = w[i, s_j] * (cur_time - server_idled_at[s_j])

                if j >= 0:

                    matches = matches + 1
                    if matches > warm_up:
                        if not record:
                            record = True
                            record_stat_time = cur_time
                        matching_counter[i, j] = matching_counter[i, j] + 1
                        waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
                        idle_time = cur_time - server_idled_at[s_j]
                        idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
                    server_states[j] = 1
                    service_time = np.random.exponential(s[i]/mu[j])
                    heappush(event_stream, (cur_time + service_time, np.int16(i), np.int16(j)))
                
                else:
                    customer_queues[i].append(cur_time)

            heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), np.int16(i), np.int16(-1)))
        
        else:

            i = np.int16(-1)
            longest_waiting = 0

            for c_i in s_adj[j]:
                if len(customer_queues[c_i]) > 0:
                    if w_only:
                        if w[c_i, j] > longest_waiting:
                            i = c_i
                            longest_waiting = w[c_i, j]                      
                    else:
                        if w[c_i, j] * (cur_time - customer_queues[c_i][0]) > longest_waiting:
                            i = c_i
                            longest_waiting = w[c_i, j] * (cur_time - customer_queues[c_i][0])

            if i >= 0:

                arrival_time = customer_queues[i].pop()
                matches = matches + 1
                if matches > warm_up:
                    if not record:
                        record = True
                        record_stat_time = cur_time
                    matching_counter[i, j] = matching_counter[i,j] + 1
                    waiting_time = cur_time - arrival_time
                    waiting_times[i, :] = waiting_times[i, :] + np.array([1, waiting_time, waiting_time**2])

                service_time = np.random.exponential(s[i]/mu[j])
                heappush(event_stream, (cur_time + service_time, np.int16(i), np.int16(j)))
            
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/matching_counter.sum(), matching_counter/(cur_time - record_stat_time), waiting_times, idle_times


@jit(nopython=True, cache=True)
def queueing_sim_loop_lqf(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len, w_only):

    heapify = hq.heapify
    heappop = hq.heappop
    heappush = hq.heappush

    matching_counter = np.zeros((m,n))
    idle_times = np.zeros((n, 3))
    waiting_times = np.zeros((m, 3))
    server_states = np.zeros(n, dtype=np.int8)
    server_idled_at = np.zeros(n, dtype=np.float64)
    interarrival = 1./lamda
    matches = 0
    arrivals = 0
    record = False
    record_stat_time = 0

    heapify(event_stream)
    heappop(event_stream)

    for i in range(m):
        customer_queues[i].pop()
        heappush(event_stream, (np.random.exponential(interarrival[i]), np.int16(i), np.int16(-1)))

    while matches < sim_len:

        event = heappop(event_stream)

        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            if len(customer_queues[i]) > 0:
                customer_queues[i].append(cur_time)

            else:

                j = np.int16(-1)
                longest_idle = 0
                for s_j in c_adj[i]:
                    if server_states[s_j] == 0:
                        if w_only:
                            if w[i, s_j] > longest_idle:
                                j = s_j
                                longest_idle = w[i, s_j]
                        else:
                            if w[i, s_j] * (cur_time - server_idled_at[s_j]) > longest_idle:
                                j = s_j
                                longest_idle = w[i, s_j] * (cur_time - server_idled_at[s_j])

                if j >= 0:

                    matches = matches + 1
                    if matches > warm_up:
                        if not record:
                            record = True
                            record_stat_time = cur_time
                        matching_counter[i, j] = matching_counter[i, j] + 1
                        waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
                        idle_time = cur_time - server_idled_at[s_j]
                        idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
                    server_states[j] = 1
                    service_time = np.random.exponential(s[i]/mu[j])
                    heappush(event_stream, (cur_time + service_time, np.int16(i), np.int16(j)))
                
                else:
                    customer_queues[i].append(cur_time)

            heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), np.int16(i), np.int16(-1)))
        
        else:

            i = np.int16(-1)
            longest_queue = 0

            for c_i in s_adj[j]:
                if len(customer_queues[c_i]) > 0:
                    if w_only:
                        if w[c_i, j] > longest_queue:
                            i = c_i
                            longest_queue = w[c_i, j]                      
                    else:
                        c_i_queue = w[c_i, j] * len(customer_queues[c_i]) 
                        if c_i_queue > longest_queue:
                            i = c_i
                            longest_queue = c_i_queue

            if i >= 0:

                arrival_time = customer_queues[i].pop()
                matches = matches + 1
                if matches > warm_up:
                    if not record:
                        record = True
                        record_stat_time = cur_time
                    matching_counter[i, j] = matching_counter[i,j] + 1
                    waiting_time = cur_time - arrival_time
                    waiting_times[i, :] = waiting_times[i, :] + np.array([1, waiting_time, waiting_time**2])

                service_time = np.random.exponential(s[i]/mu[j])
                heappush(event_stream, (cur_time + service_time, np.int16(i), np.int16(j)))
            
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/matching_counter.sum(), matching_counter/(cur_time - record_stat_time), waiting_times, idle_times


@jit(nopython=True, cache=True)
def assingment_speeds(m, iters, queues):

    a = np.zeros((m,m), dtype=np.float64)

    for k in range(iters):
        i = np.random.randint(m)
        j = np.random.randint(m)
        b = np.random.uniform(0, 1)
        a[i, j] = a[i, j] + b
        queues[i].append(b)


    return a


def get_mean_and_stdev(data_mean, data_stdev, data_k):
    
    data_k_mean = np.divide(data_k[:, 1], data_k[:, 0], out=np.zeros_like(data_k[:, 1]), where=data_k[:, 0]!=0)
    sq_data_k_mean = np.divide(data_k[:, 1]**2, data_k[:, 0], out=np.zeros_like(data_k[:, 1]), where=data_k[:, 0]!=0)
    data_k_n_m_1 = data_k[:, 0] - 1. 
    data_k_stdev = np.divide((data_k[:, 2] - sq_data_k_mean)**0.5, data_k_n_m_1, out=np.zeros_like(data_k_n_m_1), where=data_k_n_m_1!=0)

    data_mean.append(data_k_mean)
    data_stdev.append(data_k_stdev)
    
    return data_mean, data_stdev


def parse_sim_data(compatability_matrix, matching_ratios, matching_rates, waiting_times=None, waiting_times_stdev=None, idle_times=None, idle_times_stdev=None, aux_sim_data=None):

    results = {'mat': dict(), 'col': dict(), 'row': dict(), 'aux': dict()}

    sims = len(matching_rates)

    nnz = compatability_matrix.nonzero()        
    
    results['mat']['sim_matching_rates'] = sum(matching_rates)*(1./sims)
    matching_rates_mean = results['mat']['sim_matching_rates']
    results['mat']['sim_matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean

    results['mat']['sim_matching_ratios'] = sum(matching_ratios)*(1./sims)
    matching_ratios_mean = results['mat']['matching_ratios']
    results['mat']['sim_matching_ratios_stdev'] = (sum((matching_ratios[k] - matching_ratios_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_ratios_mean


    if waiting_times is not None:
        results['row']['sim_waiting_times'] = sum(waiting_times)*(1./sims)
        waiting_times_mean = results['row']['sim_waiting_times']
        results['row']['sim_waiting_times_stdev'] = (sum((waiting_times[k] - waiting_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    
    if waiting_times_stdev is not None:
        results['row']['sig_sim_waiting_times'] = sum(waiting_times_stdev)*(1./sims)
        waiting_times_stdev_mean = results['row']['sig_sim_waiting_times']
        results['row']['sig_sim_waiting_times_stdev'] = (sum((waiting_times_stdev[k] - waiting_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    
    if idle_times is not None:
        results['col']['sim_idle_times'] = sum(idle_times)*(1./sims)
        idle_times_mean = results['col']['sim_idle_times']
        results['col']['sim_idle_times_stdev'] = (sum((idle_times[k] - idle_times_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean

    if idle_times_stdev is not None:
        results['col']['sig_sim_idle_times'] = sum(idle_times_stdev)*(1./sims)
        idle_times_stdev_mean = results['col']['sig_sim_idle_times']
        results['col']['sig_sim_idle_times_stdev'] = (sum((idle_times_stdev[k] - idle_times_stdev_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    
    for key, val in aux_sim_data.items():
        results['aux'][key] = val

    return results



