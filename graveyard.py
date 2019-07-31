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