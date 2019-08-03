def approximate_sbpss_w_alis(exp, timestamp, rho):


        exp_data = exp[['m', 'n', 'density_level', 'graph_no', 'exp_no', 'beta_dist']].drop_duplicates()
        
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]
        density_level = exp_data['density_level'].iloc[0]

        alpha = np.zeros(m)
        s = np.ones(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m, n))
        sim_matching_rates = np.zeros((m, n))
        sim_matching_rates_stdev = np.zeros((m, n))
        light_approx = np.zeros((m, n))
        rho_approx = np.zeros((m, n))        

        for k, row in exp.iterrows():
            sim_matching_rates[int(row['i']), int(row['j'])] = row['sim_matching_rates']
            sim_matching_rates_stdev[int(row['i']), int(row['j'])] = row['sim_matching_rates_stdev']
            light_approx[int(row['i']), int(row['j'])] = row['light_approx']
            rho_approx[int(row['i']), int(row['j'])] = row['rho_approx']
            alpha[int(row['i'])] = float(row['alpha'])
            beta[int(row['j'])] = float(row['beta'])
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()

        no_of_edges = len(nnz[0])

        sbpss_df = []
                
        st = time()            
        lamda = alpha * rho
        mu = beta

        exp_res = {'mat': dict(), 'row': dict(), 'col': dict(), 'aux': dict()}

        exp_res['mat']['heavy_approx'] = entropy_approximation(compatability_matrix, lamda, mu, pad=True)
        exp_res['mat']['alis_approx'] = alis_approximation(compatability_matrix, alpha, beta, rho)
        exp_res['mat']['rho_approx_alis'] = (1. - rho) * exp_res['mat']['alis_approx'] + (rho) * exp_res['mat']['heavy_approx']
        exp_res['mat']['sim_matching_rates'] = sim_matching_rates
        exp_res['mat']['sim_matching_rates_stdev'] = sim_matching_rates_stdev
        exp_res['mat']['light_approx'] = light_approx
        exp_res['mat']['rho_approx'] = rho_approx

        print('density_level:', density_level, 'graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist, 'rho:', rho)
        print('alis_approx_error:', (np.abs(exp_res['mat']['alis_approx']-sim_matching_rates).sum())/lamda.sum())
        print('heavy_approx_error:', (np.abs(exp_res['mat']['heavy_approx']-sim_matching_rates).sum())/lamda.sum())
        print('rho_approx_alis_error:', (np.abs(exp_res['mat']['rho_approx_alis']-sim_matching_rates).sum())/lamda.sum())
        print('rho_approx_error:', (np.abs(exp_res['mat']['rho_approx']-sim_matching_rates).sum())/lamda.sum())

        exp_res['aux']['graph_no'] =  graph_no
        exp_res['aux']['exp_no'] =  exp_no
        exp_res['aux']['beta_dist'] =  beta_dist
        exp_res['aux']['density_level'] =  density_level
        exp_res['aux']['rho'] = rho


        sbpss_rho_df = log_res_to_df(compatability_matrix, alpha, beta, lamda,  s, mu,  result_dict=exp_res, timestamp=timestamp)

        sbpss_df.append(sbpss_rho_df)

        return sbpss_df


def approximate_sbpss_cd_w_alis(exp, timestamp, rho):


        exp_data = exp[['m', 'n', 'density_level', 'graph_no', 'exp_no', 'beta_dist']].drop_duplicates()
        
        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]
        density_level = exp_data['density_level'].iloc[0]

        alpha = np.zeros(m)
        s = np.zeros(m)
        beta = np.zeros(n)
        mu = np.zeros(n)
        compatability_matrix = np.zeros((m, n))
        sim_matching_rates = np.zeros((m, n))
        sim_matching_rates_stdev = np.zeros((m, n))
        light_approx = np.zeros((m, n))
        rho_approx = np.zeros((m, n))        

        for k, row in exp.iterrows():
            sim_matching_rates[int(row['i']), int(row['j'])] = row['sim_matching_rates']
            sim_matching_rates_stdev[int(row['i']), int(row['j'])] = row['sim_matching_rates_stdev']
            light_approx[int(row['i']), int(row['j'])] = row['light_approx']
            rho_approx[int(row['i']), int(row['j'])] = row['rho_approx']

            beta[int(row['j'])] = float(row['beta'])
            alpha[int(row['i'])] = float(row['alpha'])
            lamda[int(row['i'])] = float(row['lamda'])
            s[int(row['i'])] = float(row['s'])
            compatability_matrix[int(row['i']), int(row['j'])] = 1.

        nnz = compatability_matrix.nonzero()

        no_of_edges = len(nnz[0])

        sbpss_df = []
                
        st = time()            
        lamda = alpha * rho
        mu = beta

        exp_res = {'mat': dict(), 'row': dict(), 'col': dict(), 'aux': dict()}

        heavy_traffic_approx_entropy_eta =  entropy_approximation(compatability_matrix, eta, mu, pad=True)
        heavy_traffic_approx_entropy = np.dot(np.diag(1./s), heavy_traffic_approx_entropy_eta)
        exp_res['mat']['heavy_traffic_approx_entropy'] = heavy_traffic_approx_entropy
        exp_res['mat']['alis_approx'] = alis_approximation(compatability_matrix, alpha, beta, rho)
        exp_res['mat']['rho_approx_alis'] = (1. - rho) * exp_res['mat']['alis_approx'] + (rho) * exp_res['mat']['heavy_approx']
        exp_res['mat']['sim_matching_rates'] = sim_matching_rates
        exp_res['mat']['sim_matching_rates_stdev'] = sim_matching_rates_stdev
        exp_res['mat']['light_approx'] = light_approx
        exp_res['mat']['rho_approx'] = rho_approx

        print('density_level:', density_level, 'graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist, 'rho:', rho)
        print('alis_approx_error:', (np.abs(exp_res['mat']['alis_approx']-sim_matching_rates).sum())/lamda.sum())
        print('heavy_approx_error:', (np.abs(exp_res['mat']['heavy_approx']-sim_matching_rates).sum())/lamda.sum())
        print('rho_approx_alis_error:', (np.abs(exp_res['mat']['rho_approx_alis']-sim_matching_rates).sum())/lamda.sum())
        print('rho_approx_error:', (np.abs(exp_res['mat']['rho_approx']-sim_matching_rates).sum())/lamda.sum())

        exp_res['aux']['graph_no'] =  graph_no
        exp_res['aux']['exp_no'] =  exp_no
        exp_res['aux']['beta_dist'] =  beta_dist
        exp_res['aux']['density_level'] =  density_level
        exp_res['aux']['rho'] = rho


        sbpss_rho_df = log_res_to_df(compatability_matrix, alpha, beta, lamda,  s, mu,  result_dict=exp_res, timestamp=timestamp)

        sbpss_df.append(sbpss_rho_df)

        return sbpss_df

def go_back_and_approximate_sbpss_w_alis(filename='FZ_Kaplan_exp_sbpss_good2'):

    df = pd.read_csv(filename + '.csv')
    p = 3
    pool = mp.Pool(processes=p)

    for density_level in ['high', 'medium', 'low']:
        exps = []
        for (timestamp, rho), exp in df[df['density_level'] == density_level].groupby(by=['timestamp', 'rho'], as_index=False):
            exps.append([exp, timestamp, rho])
            if len(exps) == p:
                print('no_of_exps:', len(exps), 'density_level:', density_level)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_w_alis, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_good_w_alis', sbpss_df)
                exps = []
        else:
            if len(exps) > 0:
                print('no_of_exps:', len(exps), 'density_level:', density_level)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_w_alis, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_good_w_alis', sbpss_df)
                exps = []   


def go_back_and_approximate_sbpss_w_alis(filename='FZ_Kaplan_exp_sbpss_good2'):

    df = pd.read_csv(filename + '.csv')
    p = 3
    pool = mp.Pool(processes=p)

    for density_level in ['high', 'medium', 'low']:
        exps = []
        for (timestamp, rho), exp in df[df['density_level'] == density_level].groupby(by=['timestamp', 'rho'], as_index=False):
            exps.append([exp, timestamp, rho])
            if len(exps) == p:
                print('no_of_exps:', len(exps), 'density_level:', density_level)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_w_alis, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_good_w_alis', sbpss_df)
                exps = []
        else:
            if len(exps) > 0:
                print('no_of_exps:', len(exps), 'density_level:', density_level)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_w_alis, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_good_w_alis', sbpss_df)
                exps = []   

def go_back_and_approximate_sbpss_customer_dependet(filename='FZ_final_w_qp'):

    df = pd.read_csv(filename + '.csv')
    p = 8
    pool = mp.Pool(processes=p)

    for n in range(7,11,1):
        exps = []
        for timestamp, exp in df[df['n'] == n].groupby(by=['timestamp'], as_index=False):
            exps.append([exp, timestamp])
            if len(exps) == p:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_customer_dependent, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_cd4', sbpss_df)
                exps = []
        else:
            if len(exps) > 0:
                print('no_of_exps:', len(exps), 'n:', n)
                print('starting work with {} cpus'.format(p))
                sbpss_dfs = pool.starmap(approximate_sbpss_customer_dependent, exps)
                sbpss_df = pd.concat([df for dfs in sbpss_dfs for df in dfs], axis=0)
                write_df_to_file('FZ_Kaplan_exp_sbpss_cd4', sbpss_df)
                exps = []   

@jit(nopython=True, cache=True)
def queueing_sim_loop_with_w(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len):

    # heapify = hq.heapify
    # heappop = hq.heappop
    # heappush = hq.heappush

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

    for i in range(m):
        customer_queues[i].pop()
        heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    heapify(event_stream)
    heappop(event_stream)
    for i in range(m):
        heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    while arrivals < sim_len:

        event = heappop(event_stream)

        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            arrivals = arrivals + 1

            if len(customer_queues[i]) > 0:
                customer_queues[i].append(cur_time)

            else:

                j = -1
                longest_idle_time = 0

                for s_j in c_adj[i]:
                    if server_states[s_j] == 0:
                        s_j_weighted_idle_time = w[i, s_j] * (cur_time - server_idled_at[s_j])
                        if s_j_weighted_idle_time > longest_idle_time:
                            j = s_j
                            longest_idle_time = s_j_weighted_idle_time

                if j >= 0:

                    matches = matches + 1
                    if matches > warm_up:
                        if not record:
                            record = True
                            record_stat_time = cur_time
                        matching_counter[i, j] = matching_counter[i, j] + 1
                        waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
                        idle_time = cur_time - server_idled_at[j]
                        idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
                    server_states[j] = 1
                    service_time = np.random.exponential(s[i]/mu[j])
                    heappush(event_stream, (cur_time + service_time, i, j))
                
                else:
                    customer_queues[i].append(cur_time)

            heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, -1))
        
        else:

            i = -1
            longest_waiting_time = 0

            for c_i in s_adj[j]:
                if customer_queues[c_i]:
                    c_i_weighted_waiting_time =  w[c_i, j] * (cur_time - customer_queues[c_i][0])
                    if c_i_weighted_waiting_time > longest_waiting_time:
                        i = c_i
                        longest_waiting_time = c_i_weighted_waiting_time

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
                heappush(event_stream, (cur_time + service_time, i, j))
            
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/(cur_time - record_stat_time), waiting_times, idle_times


@jit(nopython=True, cache=True)
def queueing_sim_loop_with_only_w(customer_queues, event_stream, lamda, s, mu, w, s_adj, c_adj, m, n, warm_up, sim_len):

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

    for i in range(m):
        customer_queues[i].pop()
        heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    heapify(event_stream)
    heappop(event_stream)
    for i in range(m):
        heappush(event_stream, (np.random.exponential(interarrival[i]), i, -1))

    while arrivals < sim_len:

        event = heappop(event_stream)

        cur_time = event[0]
        i = event[1]
        j = event[2]

        if j == -1:

            arrivals = arrivals + 1

            if len(customer_queues[i]) > 0:
                customer_queues[i].append(cur_time)

            else:

                j = -1
                longest_idle_time = 0

                for s_j in c_adj[i]:
                    if server_states[s_j] == 0:
                        s_j_weighted_idle_time = w[i, s_j]
                        if s_j_weighted_idle_time > longest_idle_time:
                            j = s_j
                            longest_idle_time =  s_j_weighted_idle_time

                if j >= 0:

                    matches = matches + 1
                    if matches > warm_up:
                        if not record:
                            record = True
                            record_stat_time = cur_time
                        matching_counter[i, j] = matching_counter[i, j] + 1
                        waiting_times[i] = waiting_times[i] + np.array([1, 0, 0])
                        idle_time = cur_time - server_idled_at[j]
                        idle_times[j, :] = idle_times[j, :] + np.array([1, idle_time, idle_time**2])
                    
                    server_states[j] = 1
                    service_time = np.random.exponential(s[i]/mu[j])
                    heappush(event_stream, (cur_time + service_time, i, j))
                
                else:
                    customer_queues[i].append(cur_time)

            heappush(event_stream, (cur_time + np.random.exponential(interarrival[i]), i, -1))
        
        else:

            i = -1
            longest_waiting_time = 0

            for c_i in s_adj[j]:
                if len(customer_queues[c_i]) > 0:
                    c_i_weighted_waiting_time =  w[c_i, j]
                    if c_i_weighted_waiting_time > longest_waiting_time:
                        i = c_i
                        longest_waiting_time = c_i_weighted_waiting_time

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
                heappush(event_stream, (cur_time + service_time, i, j))
            
            else:
                server_states[j] = 0
                server_idled_at[j] = cur_time

    return matching_counter/(cur_time - record_stat_time), waiting_times, idle_times




SIM_LENGTHS = {
    6: 10**5,
    10: 5*10**5,
    100: 2*(10**6),
    300: 7*(10**6),
    1000: 30*(10**6),
}

def run_assignmet(m, iters):


    queues = tuple(list() for i in range(m))
    for i in range(m):
        queues[i].append(-1.)

    type_checker(queues[0])

    # queues = tuple([-1.] for i in range(m))
    # s = time()
    # a = assingment_speeds(m, iters, queues)
    # print(time() - s)
    # s = time()
    # a = assingment_speeds(m, iters, queues)
    # print(time() - s)


def type_checker(obj):

    print(numba.typeof(obj))



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


def simulate_queueing_system2(compatability_matrix, lamda, mu, s=None, w=None, only_w=False, prt=True, sims=30, sim_len=None, warm_up=None, seed=None, sim_name='sim', per_edge=1000,prt_all=True):


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

    s_adj = tuple(np.nonzero(compatability_matrix[:, j])[0] for j in range(n)) 
    c_adj = tuple(np.nonzero(compatability_matrix[i, :])[0] for i in range(m)) 

    if s is None:
        s = np.ones(m)
    
    service_rates = np.dot(np.dot(np.diag(s), compatability_matrix), np.diag(1./mu))

    start_time = time()
    sim_len_mils = ceil(sim_len/10**5)
    warm_up_mils = ceil(warm_up/10**5)

    if prt:
        print('sim length: ', sim_len, ' warm_up_period: ', warm_up)    
    print(range(1, sims + 1, 1))
    for k in range(1, sims + 1, 1):

        
        if prt_all:
            print('starting sim ', k, ' {:.4f} '.format(time() - start_time))

        start_time_k = time()
        
        if seed is not None:
            np.random.seed(seed + k)
        
        customer_queues = tuple([-1.] for i in range(m))
        event_stream = [(-1., -1, -1)]
        matching_counter_k = np.zeros((m, n), dtype=np.int32)
        idle_times_k = np.zeros((n, 3), dtype=np.float64)
        waiting_times_k = np.zeros((m, 3), dtype=np.float64)
        server_states = np.zeros(n, dtype=np.int8)
        server_idled_at = np.zeros(n, dtype=np.float64)
        sim_cur_time  = 0

        print('length in mils: ', )
        for l in range(warm_up_mils + sim_len_mils):
            print('mil: ', l)
            if l == warm_up_mils:
                record_stat_time = sim_cur_time
                matching_counter_k = np.zeros((m, n), dtype=np.int32)
                idle_times_k = np.zeros((n, 3), dtype=np.float64)
                waiting_times_k = np.zeros((m, 3), dtype=np.float64)

            (
                matching_counter_k, waiting_time_k, idle_times_k,
                event_stream, customer_queues, server_states, server_idled_at,
                sim_cur_time
            ) = queueing_sim_loop_warm(
                matching_counter_k, waiting_times_k, idle_times_k,
                event_stream, customer_queues, server_states, server_idled_at,
                lamda, s, mu, s_adj, c_adj, m, n,
                initilize=(l == 0)
            )

        matching_rates_k = matching_counter_k/(sim_cur_time - record_stat_time)
           

        matching_rates.append(matching_rates_k)

        for data_name, data_k, data_sum, data_stdev  in zip(['waiting', 'idle'], [waiting_times_k, idle_times_k], [waiting_times, idle_times], [waiting_times_stdev, idle_times_stdev]):
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

    return parse_sim_data(
        compatability_matrix, matching_rates,
        waiting_times, waiting_times_stdev,
        idle_times, idle_times_stdev,
        sims, total_duration, sim_len, warm_up, seed)


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
    matching_rates_mean = results['mat']['sim_matching_rates']
    results['mat']['sim_matching_rates_stdev'] = (sum((matching_rates[k] - matching_rates_mean)**2 for k in range(sims))*(1./(sims-1)))**0.5 if sims > 1 else 0 * matching_rates_mean
    results['aux']['no_of_sims'] = sims
    results['aux']['sim_duration'] = total_duration
    results['aux']['sim_len'] = sim_len
    results['aux']['warm_up'] = warm_up
    results['aux']['seed'] = seed

    return results


@profile
jit(nopython=True)
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