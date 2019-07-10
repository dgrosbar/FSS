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