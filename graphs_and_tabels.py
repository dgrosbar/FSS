import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from itertools import product
from utilities import printarr, printcols


def comparison_graph(filename):

    res_df = pd.read_csv(filename + '.csv')
    res_df = res_df[res_df['exact']]
    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'alph_dist', 'beta_dist']
    data_cols = ['ohm_error_abs','ent_error_abs', 'ohm_error_abs_pct','ent_error_abs_pct']
    cols = base_cols + data_cols

    def f(df):

        d = {}

        d['sum_ohm_error_abs'] = df['ohm_error_abs'].sum()
        d['mean_ohm_error_abs'] = df['ohm_error_abs'].mean()
        d['max_ohm_error_abs'] = df['ohm_error_abs'].max()
        d['sum_ent_error_abs'] = df['ent_error_abs'].sum()
        d['mean_ent_error_abs'] = df['ent_error_abs'].mean()
        d['max_ent_error_abs'] = df['ent_error_abs'].max()

        d['mean_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].mean()
        d['max_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].max()
        d['mean_ent_error_abs_pct'] = df['ent_error_abs_pct'].mean()
        d['max_ent_error_abs_pct'] = df['ent_error_abs_pct'].max()

        index = ['sum_ohm_error_abs','mean_ohm_error_abs','max_ohm_error_abs','sum_ent_error_abs','mean_ent_error_abs','max_ent_error_abs',
        'mean_ohm_error_abs_pct','max_ohm_error_abs_pct','mean_ent_error_abs_pct','max_ent_error_abs_pct']

        return pd.Series(d, index=index)

    agg_res = res_df[cols].groupby(by=base_cols, as_index=False).apply(f).reset_index()

    print(agg_res)

    fig, ax = plt.subplots(2,3)
    col_plt = {'low': 0, 'medium': 1, 'high': 2}
    row_plt = {'unifrom': 0, 'exponential': 1}

    for (density, dist), grp  in agg_res.groupby(by=['density_level', 'alph_dist']):

        print(list(grp.columns.values))

        row = row_plt[dist]
        col = col_plt[density]

        ohm_x = grp['mean_ohm_error_abs']
        ohm_y = grp['max_ohm_error_abs']


        ent_x = grp['mean_ent_error_abs']
        ent_y = grp['max_ent_error_abs']

        max_x = max(ohm_x.max(), ent_x.max())
        max_y = max(ohm_y.max(), ent_y.max())

        ax[row, col].set_title('Density: {}, Distribution: {}'.format(density, dist))
        means_ohm = 'ohm_law - avg_err: mean: {:.4f} max: {:.4f}'.format(ohm_x.mean(), ohm_y.mean())
        means_ent = 'max_ent - avg_err: mean: {:.4f} max: {:.4f}'.format(ent_x.mean(), ent_y.mean())
        #ax[row, col].text(0.02, 0.7, means, ha='left', va='center', transform=-ax[row, col].transAxes)
        ax[row, col].set_xlim(0, max_x*1.1)
        ax[row, col].set_ylim(0, max_y*1.1)
        ax[row, col].legend(loc=1)
        ax[row, col].xaxis.set_label_text('sum_abs_error')
        ax[row, col].yaxis.set_label_text('max_abs_error')

        ax[row, col].scatter(x=ohm_x, y=ohm_y, label=means_ohm, color='red', s=1)
        ax[row, col].scatter(x=ent_x, y=ent_y, label=means_ent, color='blue', s=1)

        ax[row, col].legend(loc=2)

    #plt.show()


def comparison_graph2(filename):

    res_df = pd.read_csv(filename + '.csv')
    # res_df[~res_df['exact']]['ohm_error_abs'] = res_df[~res_df['exact']]['ohm_error_abs_sim']
    # res_df[~res_df['exact']]['ent_error_abs'] = res_df[~res_df['exact']]['ent_error_abs_sim']
    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'alph_dist', 'beta_dist']
    data_cols = ['ohm_error_abs','ent_error_abs', 'ohm_error_abs_pct','ent_error_abs_pct']
    cols = base_cols + data_cols

    def f(df):

        d = {}

        d['sum_ohm_error_abs'] = df['ohm_error_abs'].sum()
        d['mean_ohm_error_abs'] = df['ohm_error_abs'].mean()
        d['max_ohm_error_abs'] = df['ohm_error_abs'].max()
        d['sum_ent_error_abs'] = df['ent_error_abs'].sum()
        d['mean_ent_error_abs'] = df['ent_error_abs'].mean()
        d['max_ent_error_abs'] = df['ent_error_abs'].max()

        d['mean_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].mean()
        d['max_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].max()
        d['mean_ent_error_abs_pct'] = df['ent_error_abs_pct'].mean()
        d['max_ent_error_abs_pct'] = df['ent_error_abs_pct'].max()

        index = ['sum_ohm_error_abs','mean_ohm_error_abs','max_ohm_error_abs','sum_ent_error_abs','mean_ent_error_abs','max_ent_error_abs',
        'mean_ohm_error_abs_pct','max_ohm_error_abs_pct','mean_ent_error_abs_pct','max_ent_error_abs_pct']

        return pd.Series(d, index=index)

    agg_res = res_df[cols].groupby(by=base_cols, as_index=False).apply(f).reset_index()

    #fig, ax = plt.subplots(2,3, sharex=True)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row')
    col_plt = {'low': 0, 'medium': 1, 'high': 2}
    row_plt = {'unifrom': 0, 'exponential': 1}

    max_x = 0
    max_y_sum = 0
    max_y_max = 0

    for density, grp  in agg_res.groupby(by=['density_level']):

        col = col_plt[density]

        cord = grp['graph_no']*120 + 40 + grp['exp_no'] + 40 * (grp['alph_dist'] == 'exponential')
        xticks_labels = grp[['graph_no']]
        xticks_labels.loc[:, 'xticks'] = xticks_labels['graph_no']*120 + 60
        xticks_labels.loc[:, 'xlabels'] = xticks_labels['graph_no'] + 1
        xticks_labels = xticks_labels.drop_duplicates()

        ohm_sum = grp['sum_ohm_error_abs']
        ohm_max = grp['max_ohm_error_abs']

        ent_sum = grp['sum_ent_error_abs']
        ent_max = grp['max_ent_error_abs']

        max_x = max(cord) + 10 if max(cord) + 10 > max_x else max_x
        max_y_sum = max(ohm_sum.max(), ent_sum.max()) if max(ohm_sum.max(), ent_sum.max()) > max_y_sum else max_y_sum 
        max_y_max = max(ohm_max.max(), ent_max.max()) if max(ohm_max.max(), ent_max.max()) > max_y_max else max_y_max

        # means_ohm = 'ohm_law - avg_err: mean: {:.4f} max: {:.4f}'.format(ohm_x.mean(), ohm_y.mean())
        # means_ent = 'max_ent - avg_err: mean: {:.4f} max: {:.4f}'.format(ent_x.mean(), ent_y.mean())
        #ax[row, col].text(0.02, 0.7, means, ha='left', va='center', transform=-ax[row, col].transAxes)
        for row, ohm_data, ent_data in [(0, ohm_sum, ent_sum), (1, ohm_max, ent_max)]:
            if row == 0:
                ax[row, col].set_title('Density: {}'.format(density))
            if col == 0 and row == 0:
                ax[row, col].yaxis.set_label_text('sum_abs_error')
            if col == 0 and row == 1:
                ax[row, col].yaxis.set_label_text('max_abs_error')

            ax[row, col].set_xticks(xticks_labels['xticks'])
            ax[row, col].set_xticklabels(xticks_labels['xlabels'])

            ax[row, col].legend(loc=1)
            ax[row, col].xaxis.set_label_text('graph no.')
            
            ax[row, col].scatter(x=cord, y=ohm_data,  color='red', s=1)
            ax[row, col].scatter(x=cord, y=ent_data,  color='blue', s=1)
            # ax[row, col].xaxis.set_label_text('sum_abs_error')
            # ax[row, col].yaxis.set_label_text('max_abs_error')
            # ax[row, col].scatter(x=cord, y=ohm_y,  color='red', s=1)
            # ax[row, col].scatter(x=cord, y=ent_y,  color='blue', s=1)

    for i in range(3):
        ax[0, i].set_xlim(0, max_x)
        ax[0, i].set_ylim(0, max_y_sum*1.1)

    for i in range(3):
        ax[1, i].set_xlim(0, max_x)
        ax[1, i].set_ylim(0, max_y_max*1.1)

    plt.show()


def comparison_graph3(filename):

    res_df = pd.read_csv(filename + '.csv')
    # res_df[~res_df['exact']]['ohm_error_abs'] = res_df[~res_df['exact']]['ohm_error_abs_sim']
    # res_df[~res_df['exact']]['ent_error_abs'] = res_df[~res_df['exact']]['ent_error_abs_sim']
    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'alph_dist', 'beta_dist']
    data_cols = ['ohm_error_abs','ent_error_abs', 'ohm_error_abs_pct','ent_error_abs_pct']
    cols = base_cols + data_cols

    def f(df):

        d = {}

        d['sum_ohm_error_abs'] = df['ohm_error_abs'].sum()
        d['mean_ohm_error_abs'] = df['ohm_error_abs'].mean()
        d['max_ohm_error_abs'] = df['ohm_error_abs'].max()
        d['sum_ent_error_abs'] = df['ent_error_abs'].sum()
        d['mean_ent_error_abs'] = df['ent_error_abs'].mean()
        d['max_ent_error_abs'] = df['ent_error_abs'].max()

        d['mean_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].mean()
        d['max_ohm_error_abs_pct'] = df['ohm_error_abs_pct'].max()
        d['mean_ent_error_abs_pct'] = df['ent_error_abs_pct'].mean()
        d['max_ent_error_abs_pct'] = df['ent_error_abs_pct'].max()

        index = ['sum_ohm_error_abs','mean_ohm_error_abs','max_ohm_error_abs','sum_ent_error_abs','mean_ent_error_abs','max_ent_error_abs',
        'mean_ohm_error_abs_pct','max_ohm_error_abs_pct','mean_ent_error_abs_pct','max_ent_error_abs_pct']

        return pd.Series(d, index=index)

    agg_res = res_df[cols].groupby(by=base_cols, as_index=False).apply(f).reset_index()

    #fig, ax = plt.subplots(2,3, sharex=True)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row')
    col_plt = {'low': 0, 'medium': 1, 'high': 2}
    row_plt = {'unifrom': 0, 'exponential': 1}

    max_x = 0
    max_y_sum = 0
    max_y_max = 0

    for (density, dist), grp  in agg_res.groupby(by=['density_level', 'alph_dist']):

        col = col_plt[density]

        if dist == 'exponential':
            marker = 'x'
            s = 10
            ohm_col = 'red'
            ent_col = 'blue'
        else:
            marker = '.'
            ohm_col = 'red'
            ent_col = 'blue'
            s =10
        print(marker)
        cord = grp['graph_no']*120 + 40 + grp['exp_no'] + 40 * (dist == 'exponential')
        xticks_labels = grp[['graph_no']]
        xticks_labels.loc[:, 'xticks'] = xticks_labels['graph_no']*120 + 40 + 30
        xticks_labels.loc[:, 'xlabels'] = xticks_labels['graph_no'] + 1
        xticks_labels = xticks_labels.drop_duplicates()

        ohm_sum = grp['sum_ohm_error_abs']
        ohm_max = grp['max_ohm_error_abs']

        ent_sum = grp['sum_ent_error_abs']
        ent_max = grp['max_ent_error_abs']

        max_x = max(cord) + 10 if max(cord) + 10 > max_x else max_x
        max_y_sum = max(ohm_sum.max(), ent_sum.max()) if max(ohm_sum.max(), ent_sum.max()) > max_y_sum else max_y_sum 
        max_y_max = max(ohm_max.max(), ent_max.max()) if max(ohm_max.max(), ent_max.max()) > max_y_max else max_y_max

        # means_ohm = 'ohm_law - avg_err: mean: {:.4f} max: {:.4f}'.format(ohm_x.mean(), ohm_y.mean())
        # means_ent = 'max_ent - avg_err: mean: {:.4f} max: {:.4f}'.format(ent_x.mean(), ent_y.mean())
        #ax[row, col].text(0.02, 0.7, means, ha='left', va='center', transform=-ax[row, col].transAxes)
        for row, ohm_data, ent_data, data_name in [(0, ohm_sum, ent_sum, 'sum'), (1, ohm_max, ent_max, 'max')]:
            if row == 0:
                ax[row, col].set_title('Density: {}'.format(density))
            if col == 0:
                ax[row, col].yaxis.set_label_text(data_name + '_abs_error')

            ax[row, col].set_xticks(xticks_labels['xticks'])
            ax[row, col].set_xticklabels(xticks_labels['xlabels'])

            ax[row, col].xaxis.set_label_text('graph no.')
            
            ax[row, col].scatter(x=cord, y=ohm_data, marker=marker, color=ohm_col, s=s, label='ohm_' + dist)
            ax[row, col].scatter(x=cord, y=ent_data, marker=marker, color=ent_col, s=s, label='ent_' + dist)
            ax[row, col].legend()
            # ax[row, col].xaxis.set_label_text('sum_abs_error')
            # ax[row, col].yaxis.set_label_text('max_abs_error')
            # ax[row, col].scatter(x=cord, y=ohm_y,  color='red', s=1)
            # ax[row, col].scatter(x=cord, y=ent_y,  color='blue', s=1)

    for i in range(3):
        ax[0, i].set_xlim(0, max_x + 40)
        ax[0, i].set_ylim(0, max_y_sum*1.1)

    for i in range(3):
        ax[1, i].set_xlim(0, max_x + 40)
        ax[1, i].set_ylim(0, max_y_max*1.1)

    plt.show()


def comparison_graph4(filename, approx_names):

    res_df = pd.read_csv(filename + '.csv')
    # res_df[~res_df['exact']]['ohm_error_abs'] = res_df[~res_df['exact']]['ohm_error_abs_sim']
    # res_df[~res_df['exact']]['ent_error_abs'] = res_df[~res_df['exact']]['ent_error_abs_sim']
    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'alph_dist', 'beta_dist']
    data_cols = [approx + data_col for approx in approx_names for data_col in ['_error_abs' ,'_error_abs_pct']]


    cols = base_cols + data_cols

    def f(df):

        index = []

        d = {}

        for approx in approx_names:

            d['sum_' + approx + '_error_abs'] = df[approx + '_error_abs'].sum()
            d['mean_' + approx + '_error_abs'] = df[approx + '_error_abs'].mean()
            d['max_' + approx + '_error_abs'] = df[approx + '_error_abs'].max()


            d['mean_' + approx + '_error_abs_pct'] = df[approx + '_error_abs_pct'].mean()
            d['max_' + approx + '_error_abs_pct'] = df[approx + '_error_abs_pct'].max()

            index = index  + [
                'sum_' + approx + '_error_abs',
                'mean_' + approx + '_error_abs',
                'max_' + approx + '_error_abs',
                'mean_' + approx + '_error_abs_pct',
                'max_' + approx + '_error_abs_pct'
            ]

        return pd.Series(d, index=index)

    agg_res = res_df[cols].groupby(by=base_cols, as_index=False).apply(f).reset_index()

    #fig, ax = plt.subplots(2,3, sharex=True)
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row')
    col_plt = {'low': 0, 'medium': 1, 'high': 2}
    row_plt = {'unifrom': 0, 'exponential': 1}

    max_x = 0
    max_y_sum = 0
    max_y_max = 0

    for (density, dist), grp  in agg_res.groupby(by=['density_level', 'alph_dist']):

        col = col_plt[density]

        if dist == 'exponential':
            marker = 'x'
            s = 10
            col_0 = 'red'
            col_1 = 'blue'
        else:
            marker = '.'
            col_0 = 'red'
            col_1 = 'blue'
            s =10
        print(marker)
        cord = grp['graph_no']*120 + 40 + grp['exp_no'] + 40 * (dist == 'exponential')
        xticks_labels = grp[['graph_no']]
        xticks_labels.loc[:, 'xticks'] = xticks_labels['graph_no']*120 + 40 + 30
        xticks_labels.loc[:, 'xlabels'] = xticks_labels['graph_no'] + 1
        xticks_labels = xticks_labels.drop_duplicates()

        approx_sums = [grp['sum_' + approx + '_error_abs'] for approx in approx_names]
        approx_maxs = [grp['max_' + approx + '_error_abs'] for approx in approx_names]

        max_x = max(cord) + 10 if max(cord) + 10 > max_x else max_x
        max_y_sum = max(approx_sums[0].max(), approx_sums[1].max()) if max(approx_sums[0].max(), approx_sums[1].max()) > max_y_sum else max_y_sum 
        max_y_max = max(approx_maxs[0].max(), approx_maxs[1].max()) if max(approx_maxs[0].max(), approx_maxs[1].max()) > max_y_max else max_y_max

        # means_ohm = 'ohm_law - avg_err: mean: {:.4f} max: {:.4f}'.format(ohm_x.mean(), ohm_y.mean())
        # means_ent = 'max_ent - avg_err: mean: {:.4f} max: {:.4f}'.format(ent_x.mean(), ent_y.mean())
        #ax[row, col].text(0.02, 0.7, means, ha='left', va='center', transform=-ax[row, col].transAxes)
        for row, approx_1_data, approx_2_data, data_name in [(0, approx_sums[0], approx_sums[1], 'sum'), (1, approx_maxs[0], approx_maxs[1], 'max')]:

            if row == 0:
                ax[row, col].set_title('Density: {}'.format(density))
            if col == 0:
                ax[row, col].yaxis.set_label_text(data_name + '_abs_error')

            ax[row, col].set_xticks(xticks_labels['xticks'])
            ax[row, col].set_xticklabels(xticks_labels['xlabels'])

            ax[row, col].xaxis.set_label_text('graph no.')
            
            ax[row, col].scatter(x=cord, y=approx_1_data, marker=marker, color=col_0, s=s, label= approx_names[0] + '_' + dist)
            ax[row, col].scatter(x=cord, y=approx_2_data, marker=marker, color=col_1, s=s, label= approx_names[1] + '_' + dist)
            ax[row, col].legend()
            # ax[row, col].xaxis.set_label_text('sum_abs_error')
            # ax[row, col].yaxis.set_label_text('max_abs_error')
            # ax[row, col].scatter(x=cord, y=ohm_y,  color='red', s=1)
            # ax[row, col].scatter(x=cord, y=ent_y,  color='blue', s=1)

    for i in range(3):
        ax[0, i].set_xlim(0, max_x + 40)
        ax[0, i].set_ylim(0, max_y_sum*1.1)

    for i in range(3):
        ax[1, i].set_xlim(0, max_x + 40)
        ax[1, i].set_ylim(0, max_y_max*1.1)

    plt.show()


def comparison_graph5(filename='grid_exp_parallel'):

    df = pd.read_csv(filename + '.csv')
    printcols(df)
    id_vars = [ 'timestamp',
                'structure',
                'm',
                'n',
                'exp_num',
                'max_edges',
                'edge_count',
                'edge_density',
                'utilization',
                'i',
                'j',
                'alpha',
                'beta',
                'exact_matching_rate',
                'sim_matching_rate',
                'sim_matching_rate_CI_95_U',
                'sim_matching_rate_CI_95_L',
                'no_of_sims',
                'sim_matching_rate_stdev',
                '95_CI_len'
            ]

    val_vars = ['entropy_approx','ohm_law_approx', 'quad_approx']

    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rate'])
    df.loc[:, 'abs_error_pct_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rate'])/df['approx_match_rate']


    def f(df):

        x = '_sim'


        d = {}
        
        d['total_rate'] = df['sim_matching_rate'].sum()
        d['sum_abs_error_sim'] = df['abs_error_sim'].sum()
        d['mean_abs_error_sim'] = df['abs_error_sim'].mean()
        d['max_abs_error_sim'] = df['abs_error_sim'].max()
        d['mean_abs_error_pct_sim'] = df['abs_error_pct_sim'].mean()
        d['max_abs_error_pct_sim'] = df['abs_error_pct_sim'].max()

        index = [
            'total_rate',
            'sum_abs_error_sim',
            'mean_abs_error_sim',
            'max_abs_error_sim',
            'mean_abs_error_pct_sim',
            'max_abs_error_pct_sim'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp', 'structure', 'exp_num', 'm', 'n', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res = agg_res.sort_values(by=['timestamp','approximation', 'exp_num'])

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']

    
   
    def g(df):
        
        
        d = {}
        d['approx'] = df['approximation'].max()
        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'approx',
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['approximation', 'm']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate', 'sum_abs_error_sim']]
    print(sum_res)
    sum_res = sum_res.groupby(by=['approximation', 'm'], as_index=False).apply(g).reset_index()
    
    print(sum_res)

    # sum_res.sort_values(by=['approximation', 'density_level', 'rho']).to_csv('FZ_Kaplan_sbpss_sum.csv', index=False)


def comparison_table_grids(filename='grid_final_w_qp'):

    df  = pd.read_csv(filename + '.csv')

    id_vars = ['timestamp','m','n','max_edges','edge_count','edge_density','utilization','exact','no_of_sims','i','j','alpha','beta','exact_matching_rate','sim_matching_rate','sim_matching_rate_stdev','exp_num','size','arc_dist','structure']
    val_vars = ['entropy_approx','ohm_approx', 'quad_approx']

    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rate'])
    df.loc[:, 'abs_error_pct_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rate'])/df['approx_match_rate']

    def f(df):

        x = '_sim'


        d = {}
        
        d['sum_abs_error_sim'] = df['abs_error_sim'].sum()
        d['mean_abs_error_sim'] = df['abs_error_sim'].mean()
        d['max_abs_error_sim'] = df['abs_error_sim'].max()
        d['mean_abs_error_pct_sim'] = df['abs_error_pct_sim'].mean()
        d['max_abs_error_pct_sim'] = df['abs_error_pct_sim'].max()

        index = [
            'sum_abs_error_sim',
            'mean_abs_error_sim',
            'max_abs_error_sim',
            'mean_abs_error_pct_sim',
            'max_abs_error_pct_sim'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp','exp_num', 'm', 'n', 'arc_dist', 'structure', 'edge_density', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    print(agg_res)
    # agg_res.sort_values(by=['timestamp','exp_no', 'm', 'n', 'arc_dist', 'structure', 'edge_density']).to_csv('grid_final_agg.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']

    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['m', 'n', 'arc_dist', 'structure', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['m', 'n', 'arc_dist', 'structure', 'approximation'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['m', 'n', 'structure', 'arc_dist']))

    # sum_res.sort_values(by=['approximation', 'density_level', 'rho', 'split']).to_csv('FZ_Kaplan_sbpss_cd_sum.csv', index=False)



    # print(df)


def growing_chains_graph(filename='growing_chains_new2'):
    
    res_df = pd.read_csv(filename + '.csv')

    res_df.loc[:, 'arc_type'] = np.where(res_df['i']<=res_df['j'], res_df['j'] - res_df['i'], res_df['j'] + res_df['n'] - res_df['i'])
    # for col in res_df.columns.values:
    #     print(col)
    # print(res_df[res_df['n'] == 5])

    # res_df_max = res_df[['timestamp','arc_type']].groupby(by=['timestamp'], as_index=False).max().rename(columns={'arc_type': 'k'})
    # res_df = pd.merge(left=res_df, right=res_df_max, on='timestamp', how='left')
    # for col in res_df.columns.values:
    #     print(col)

    def f(df):

        d = {}

        d['r'] = df['sim_matching_rates'].mean()
        d['sig_r'] = df['sim_matching_rates'].std()

        index = ['r', 'sig_r']

        return pd.Series(d, index=index)

    res_df = res_df[['arc_type','n','k', 'sim_matching_rates']].groupby(by=['n','k','arc_type'], as_index=False).apply(f).reset_index()
    res_df.loc[:, 'scv_r'] = (res_df['sig_r']/res_df['r'])**2
    res_df.loc[:, 'error_pct'] = res_df['r']*(res_df['n'] * res_df['k']) 
    res_df.loc[:, 'abs_error'] = np.abs(res_df['r'] - (1/(res_df['n'] * res_df['k']))) * res_df['k']
    
    res_agg = res_df[['n','k','abs_error']].groupby(by=['n','k'], as_index=False).sum()

    print(res_df)

    print(res_df[(1 == res_df['arc_type'] + 1)].sort_values(by=['k','n'])) 
    print(res_df[(1 == res_df['arc_type'] + 1)].sort_values(by=['n','k'])) 
    print(res_df[(res_df['k'] == res_df['arc_type'] + 1)]) 
    print(res_df[(res_df['k']-1)/2 == res_df['arc_type']])

    print(res_agg)


def ims_table(filename):


    df = pd.read_csv(filename + '.csv')

    id_vars = ['timestamp','density_level','graph_no','m','n','max_edges','edge_count','edge_density','exp_num','alph_dist','beta_dist','utilization','exact','i','j','alpha','beta','exact_matching_rate','sim_matching_rate', 'no_of_sims', 'sim_matching_rate_stdev']
    
    val_vars = ['ohm_approx','entropy_approx','quad_approx']
    
    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error'] = np.abs(df['approx_match_rate'] - df['exact_matching_rate'])
    df.loc[:, 'abs_error_pct'] = np.abs(df['approx_match_rate'] - df['exact_matching_rate'])/df['exact_matching_rate']


    def f(df):


        d = {}
        

        d['sum_abs_error'] = df['abs_error'].sum()
        d['mean_abs_error'] = df['abs_error'].mean()
        d['max_abs_error'] = df['abs_error'].max()
        d['mean_abs_error_pct'] = df['abs_error_pct'].mean()
        d['max_abs_error_pct'] = df['abs_error_pct'].max()

        index = [
            'sum_abs_error',
            'mean_abs_error',
            'max_abs_error',
            'mean_abs_error_pct',
            'max_abs_error_pct'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp', 'graph_no', 'exp_num', 'density_level', 'beta_dist', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    print(agg_res)

    # agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_agg.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error']
   
    def g(df):
        
        d = {}

        d['mean_max_abs_error'] = df['max_abs_error'].mean()


        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l',
            'mean_max_abs_error'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate', 'max_abs_error']].sort_values(by=['approximation', 'density_level'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['approximation', 'density_level']))

    print(sum_res.pivot(index='approximation', columns='density_level', values=['mean_err_pct','max_err_pct' ,'mean_max_abs_error', 'err_pct_95_u', 'err_pct_95_l']))

    # sum_res.sort_values(by=['approximation', 'density_level']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_sum.csv', index=False)


def sbpss_table1(filename='FZ_Kaplan_exp_sbpss_good_w_alis_adj'):

    df = pd.read_csv(filename + '.csv')

    df = df[df['sim_rate_gap'] < 0.03]

    id_vars = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'adj_sim_matching_rates']
    
    val_vars = ['heavy_approx', 'alis_approx', 'rho_approx_alis', 'light_approx', 'rho_approx']

    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error_sim'] = np.abs(df['approx_match_rate'] - df['adj_sim_matching_rates'])
    df.loc[:, 'abs_error_pct_sim'] = np.abs(df['approx_match_rate'] - df['adj_sim_matching_rates'])/df['approx_match_rate']


    def f(df):

        x = '_sim'


        d = {}
        
        d['total_rate'] = df['adj_sim_matching_rates'].sum()
        d['sum_abs_error_sim'] = df['abs_error_sim'].sum()
        d['mean_abs_error_sim'] = df['abs_error_sim'].mean()
        d['max_abs_error_sim'] = df['abs_error_sim'].max()
        d['mean_abs_error_pct_sim'] = df['abs_error_pct_sim'].mean()
        d['max_abs_error_pct_sim'] = df['abs_error_pct_sim'].max()

        index = [
            'total_rate',
            'sum_abs_error_sim',
            'mean_abs_error_sim',
            'max_abs_error_sim',
            'mean_abs_error_pct_sim',
            'max_abs_error_pct_sim'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_agg.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']
   
    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'rho', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['approximation', 'density_level', 'rho'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['approximation', 'density_level', 'rho']))

    sum_res.sort_values(by=['approximation', 'density_level', 'rho']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_sum.csv', index=False)


def sbpss_table2(filename='FZ_Kaplan_exp_sbpss_agg'):

    df = pd.read_csv(filename + '.csv')
    df_full = pd.read_csv('FZ_Kaplan_exp_sbpss_good.csv')
    df_full_base = pd.read_csv('FZ_Kaplan_exp.csv')

    max_cases = []

    for key, grp in df[df['approximation'] == 'rho_approx'].groupby(by=['density_level', 'approximation', 'beta_dist'], as_index=False):

        max_cases.append(grp.ix[grp['err_pct_of_rate'].argmax()])

    for case in max_cases:

        print(case)
        case_df = df_full[(df_full['timestamp'] == case['timestamp']) & (df_full['rho'] == case['rho'])]
        case_df_base = df_full_base[(df_full['timestamp'] == case['timestamp'])]
        for col in case_df.columns.values:
            print(col)

        exp_data = case_df[['m', 'n', 'graph_no', 'exp_no', 'beta_dist', 'rho']].drop_duplicates()

        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        rho = exp_data['rho'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]

        alpha_data = rho * case_df_base[['i', 'alpha']].drop_duplicates()
        beta_data = case_df_base[['j', 'beta']].drop_duplicates()



        print('graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist)

        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m, n))
        matching_rates = np.zeros((m, n))
        rho_approx = np.zeros((m, n))
        heavy_approx = np.zeros((m, n))
        light_approx = np.zeros((m, n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in case_df.iterrows():

            compatability_matrix[int(row['i']), int(row['j'])] = 1
            matching_rates[int(row['i']), int(row['j'])] = float(row['sim_matching_rates'])
            rho_approx[int(row['i']), int(row['j'])] = float(row['rho_approx'])
            heavy_approx[int(row['i']), int(row['j'])] = float(row['heavy_approx'])
            light_approx[int(row['i']), int(row['j'])] = float(row['light_approx'])
            print('{},{},{:.5f},{:.5f},{:.5f}'.format(row['i'], row['j'], row['sim_matching_rates'], row['heavy_approx'], np.abs(row['heavy_approx'] - row['sim_matching_rates'])))

        printarr(matching_rates.sum(axis=0), 'mr_sum_0')
        printarr(heavy_approx.sum(axis=0), 'ha_sum_0')
        printarr(matching_rates.sum(axis=1), 'mr_sum_1')
        printarr(heavy_approx.sum(axis=1), 'ha_sum_1')
        printarr(matching_rates.sum(), 'mr_sum')
        printarr(heavy_approx.sum(), 'ha_sum')
        printarr(matching_rates, 'sim_matching_rates')
        printarr(heavy_approx, 'heavy_approx')
        # printarr(light_approx, 'light_approx')


def sbpss_graph1(filename='FZ_Kaplan_sbpss_sum_w_alis'):

    sum_res = pd.read_csv(filename + '.csv')

    fig, ax = plt.subplots(3,2)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        'rho_approx_bs': 'green',
        'rho_approx': 'black',
        'rho_2_approx': 'black',
        'heavy_approx': 'red',
        'light_entropy': 'blue',
        'alis_approx': 'purple'

    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.315, 0.2955)
    }

    for key, grp in sum_res.groupby(by=['density_level', 'approximation'], as_index=False):
        for col in [0,1]:
            density_level, approximation = key
            color = approx_colors[approximation]
            row = row_plt[density_level]
            x = grp['rho']

            if approximation == 'rho_approx_bs':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=3, label='Mixed_Entropy_Approximation')
                ax[row, col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
                ax[row, col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

            elif approximation == 'heavy_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Heavy Traffic Approximation')

            elif approximation == 'alis_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Light Traffic Approximation')
                ax[row, col].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='--', label='Ent Error for IMS')

    for i in range(3):
        ax[i,0].set_xlim(0, 1)
        ax[i,0].set_ylim(0, 1)
        ax[i,1].set_xlim(0.6, 1)
        ax[i,1].set_ylim(0, .15)


    plt.legend()

    plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sbpss_graph2(filename='FZ_Kaplan_exp_sbpss_good_w_alis_sum'):

    sum_res = pd.read_csv(filename + '.csv')

    fig, ax = plt.subplots(3,2)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        'rho_approx_alis': 'green',
        'rho_approx': 'orange',
        'heavy_approx': 'red',
        'light_approx': 'blue',
        'alis_approx': 'purple'

    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.315, 0.2955)
    }

    for key, grp in sum_res.groupby(by=['density_level', 'approximation'], as_index=False):
        for col in [0,1]:
            density_level, approximation = key
            color = approx_colors[approximation]
            row = row_plt[density_level]
            x = grp['rho']

            if approximation == 'rho_approx_alis':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=3, label='New_Entropy_Approximation')
                ax[row, col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
                ax[row, col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

            elif approximation == 'heavy_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Heavy Traffic Approximation')

            elif approximation == 'rho_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Old Entropy Approximation')

            elif approximation == 'light_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Light Approximation')

            elif approximation == 'alis_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Alis Approximation')
                ax[row, col].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='--', label='Ent Error for IMS')

    for i in range(3):
        ax[i,0].set_xlim(0, 1)
        ax[i,0].set_ylim(0, .5)
        ax[i,1].set_xlim(0, 1)
        ax[i,1].set_ylim(0, .15)


    plt.legend()

    plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sbpss_graph3(filename='FZ_Kaplan_exp_sbpss_good_w_alis_sum', density='low'):

    sum_res = pd.read_csv(filename + '.csv')

    sun_res = sum_res[sum_res['density_level'] == density]

    fig, ax = plt.subplots(3,2)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        'rho_approx_alis': 'green',
        'rho_approx': 'orange',
        'heavy_approx': 'red',
        'light_approx': 'blue',
        'alis_approx': 'purple'
    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.315, 0.2955)
    }

    for key, grp in sum_res.groupby(by=['density_level', 'approximation'], as_index=False):
        for col in [0,1]:
            density_level, approximation = key
            color = approx_colors[approximation]
            row = row_plt[density_level]
            x = grp['rho']

            if approximation == 'rho_approx_alis':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=3, label='New_Entropy_Approximation')
                ax[row, col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
                ax[row, col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

            elif approximation == 'heavy_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Heavy Traffic Approximation')

            elif approximation == 'rho_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Old Entropy Approximation')

            elif approximation == 'light_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Light Approximation')

            elif approximation == 'alis_approx':

                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Alis Approximation')
                ax[row, col].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
                ax[row, col].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='--', label='Ent Error for IMS')

    for i in range(3):
        ax[i,0].set_xlim(0, 1)
        ax[i,0].set_ylim(0, .5)
        ax[i,1].set_xlim(0, 1)
        ax[i,1].set_ylim(0, .15)


    plt.legend()

    plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sbpss_graph4(filename='FZ_Kaplan_exp_sbpss_good_w_alis_sum'):

    sum_res = pd.read_csv(filename + '.csv')

    fig, ax = plt.subplots(1, 3)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        'rho_approx_alis': 'green',
        'rho_approx': 'orange',
        'heavy_approx': 'red',
        'light_approx': 'purple',
        'alis_approx': 'blue'
    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.0315, 0.02955)
    }

    cap_density_level = {
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High'
    }


    for key, grp in sum_res.groupby(by=['density_level', 'approximation'], as_index=False):

        density_level, approximation = key
        color = approx_colors[approximation]
        row = row_plt[density_level]
        ax[row].set_title(cap_density_level[density_level])
        x = grp['rho']

        if approximation == 'rho_approx_alis':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation', marker='x')
            ax[row].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'heavy_approx':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='FCFS Approximation', marker = '.', linestyle='--')

        # elif approximation == 'rho_approx':

        #     ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Old  Approximation')

        # elif approximation == 'light_approx':

        #     ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation')

        elif approximation == 'alis_approx':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation', marker='+', linestyle='-.')
            ax[row].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='-', label='MaxEnt Error for IMS')

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=12)
    fig.suptitle('Graph Density', fontsize=16)
    for i in range(3):
        ax[i].set_xlabel('utilization', fontsize=12)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, .5)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [0, 4, 5, 6, 7, 1, 2, 3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)

    plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sbpss_graph_fixing(filename='FZ_Kaplan_exp_sbpss'):

    df = pd.read_csv(filename + '.csv')
    base_df = pd.read_csv('FZ_Kaplan_exp_dissapative_res.csv')
    base_df = base_df[['timestamp', 'density_level']].drop_duplicates()
    df = pd.merge(left=df, right=base_df, on='timestamp', how='left').rename(
        columns={
        'heavy_traffic_approx_entropy': 'heavy_approx',
        'low_traffic_approx_entropy': 'light_approx'
        })

    print(df[df['light_approx']>1])

    bad_light= [
        {'graph_no': 0, 'exp_no': 13, 'beta_dist': 'exponential', 'density_level': 'low'},
        {'graph_no': 6, 'exp_no':  9, 'beta_dist': 'exponential', 'density_level': 'low'}
    ]

    bad_dfs = [
        df[((df['graph_no'] == 6) & (df['exp_no'] == 9) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] == 0.8))],
        df[((df['graph_no'] == 6) & (df['exp_no'] == 9) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] == 0.9))],
        df[((df['graph_no'] == 0) & (df['exp_no'] == 13) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] < 0.71) & (df['rho'] > 0.69))],
        df[((df['graph_no'] == 0) & (df['exp_no'] == 13) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] == 0.9))],
        df[((df['graph_no'] == 0) & (df['exp_no'] == 13) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] == 0.95))],
        df[((df['graph_no'] == 0) & (df['exp_no'] == 13) & (df['beta_dist'] == 'exponential') & (df['density_level'] =='low') & (df['rho'] == 0.99))]
    ]

    print('len(df): ', len(df))
    bad_df = pd.concat(bad_dfs, axis=0)
    print('len(bad_df): ', len(bad_df))
    df_new = df.merge(bad_df, how='left', indicator=True)
    df_new = df_new[df_new['_merge'] == 'left_only']
    print('len(new_df): ', len(df_new), len(df)-len(bad_df))
    df_new.to_csv('FZ_Kaplan_exp_sbpss_good.csv', index=False)
    print(df_new[df_new['light_approx']>1])
    bad_df.to_csv('FZ_Kaplan_exp_sbpss_bad.csv', index=False)

    # id_vars = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'sim_matching_rates']
    # val_vars = ['rho_approx', 'rho_2_approx', 'heavy_approx', 'light_approx']

    # df = pd.melt(bad_df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    # df.loc[:, 'abs_error_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rates'])
    # df.loc[:, 'abs_error_pct_sim'] = np.abs(df['approx_match_rate'] - df['sim_matching_rates'])/df['approx_match_rate']

    # def f(df):

    #     x = '_sim'


    #     d = {}
        
    #     d['total_rate'] = df['sim_matching_rates'].sum()
    #     d['sum_abs_error_sim'] = df['abs_error_sim'].sum()
    #     d['mean_abs_error_sim'] = df['abs_error_sim'].mean()
    #     d['max_abs_error_sim'] = df['abs_error_sim'].max()
    #     d['mean_abs_error_pct_sim'] = df['abs_error_pct_sim'].mean()
    #     d['max_abs_error_pct_sim'] = df['abs_error_pct_sim'].max()

    #     index = [
    #         'total_rate',
    #         'sum_abs_error_sim',
    #         'mean_abs_error_sim',
    #         'max_abs_error_sim',
    #         'mean_abs_error_pct_sim',
    #         'max_abs_error_pct_sim'
    #     ]

    #     return pd.Series(d, index=index)

    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'approximation']
    # agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    # agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']

    # print(agg_res[base_cols + ['err_pct_of_rate']].sort_values(by=['approximation','rho','density_level','beta_dist','graph_no', 'exp_no']))


    # def g(df):
        
    #     d = {}

    #     d['mean_err_pct'] = df['err_pct_of_rate'].mean()
    #     d['max_err_pct'] = df['err_pct_of_rate'].max()
    #     d['min_err_pct'] = df['err_pct_of_rate'].min()
    #     d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
    #     d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

    #     index = [
    #         'mean_err_pct',
    #         'max_err_pct',
    #         'min_err_pct',
    #         'err_pct_95_u',
    #         'err_pct_95_l'
    #     ]

    #     return pd.Series(d, index=index) 

    # sum_base_cols = ['density_level', 'rho', 'approximation']

    # sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['approximation', 'density_level', 'rho'])
    # sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    # print(sum_res.sort_values(by=['approximation', 'density_level', 'rho']))

    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in sum_res.groupby(by=['density_level', 'approximation'], as_index=False):

    #     density_level, approximation = key
    #     color = approx_colors[approximation]
    #     row = row_plt[density_level]
    #     x = grp['rho']

    #     if approximation == 'rho_approx':

    #         ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=3)
    #         ax[row].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
    #         ax[row].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

    #     elif approximation == 'heavy_approx':

    #         ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1)

    #     elif approximation == 'light_approx':

    #         ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1)

    # for i in range(3):
    #     ax[i].set_xlim(0, 1)
    #     ax[i].set_ylim(0, 0.8)

    # plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sbpss_cd_table1(filename='FZ_Kaplan_exp_sbpss_cd5'):

    df = pd.read_csv(filename + '.csv')

    df.loc[:,'sim_rate_gap'] = np.abs(df['sim_adj'] - 1.)
    df = df[df['sim_rate_gap'] < 0.03]
    df.loc[:,'adj_sim_matching_rates'] = df['sim_adj']*df['sim_matching_rates']

    id_vars = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'adj_sim_matching_rates', 'sim_rate_gap', 'split']
    val_vars = ['rho_approx', 'heavy_traffic_approx_entropy', 'low_traffic_approx_entropy']

    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error_sim'] = np.abs(df['approx_match_rate'] - df['adj_sim_matching_rates'])
    df.loc[:, 'abs_error_pct_sim'] = np.abs(df['approx_match_rate'] - df['adj_sim_matching_rates'])/df['approx_match_rate']


    def f(df):

        x = '_sim'


        d = {}
        
        d['total_rate'] = df['adj_sim_matching_rates'].sum()
        d['sum_abs_error_sim'] = df['abs_error_sim'].sum()
        d['mean_abs_error_sim'] = df['abs_error_sim'].mean()
        d['max_abs_error_sim'] = df['abs_error_sim'].max()
        d['mean_abs_error_pct_sim'] = df['abs_error_pct_sim'].mean()
        d['max_abs_error_pct_sim'] = df['abs_error_pct_sim'].max()

        index = [
            'total_rate',
            'sum_abs_error_sim',
            'mean_abs_error_sim',
            'max_abs_error_sim',
            'mean_abs_error_pct_sim',
            'max_abs_error_pct_sim'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'split' ,'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho', 'split']).to_csv('FZ_Kaplan_sbpss_cd_agg.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']
   
    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'rho', 'approximation', 'split']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['approximation', 'density_level', 'rho', 'split'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['approximation', 'density_level', 'rho', 'split']))

    sum_res.sort_values(by=['approximation', 'density_level', 'rho', 'split']).to_csv('FZ_Kaplan_sbpss_cd_sum.csv', index=False)


def sbpss_cd_table2(filename='FZ_Kaplan_exp_sbpss_cd_agg'):

    df = pd.read_csv(filename + '.csv')
    df_full = pd.read_csv('FZ_Kaplan_exp_sbpss_good.csv')
    df_full_base = pd.read_csv('FZ_Kaplan_exp.csv')

    max_cases = []

    for key, grp in df[df['approximation'] == 'rho_approx'].groupby(by=['density_level', 'approximation', 'beta_dist'], as_index=False):

        max_cases.append(grp.ix[grp['err_pct_of_rate'].argmax()])

    for case in max_cases:

        print(case)
        case_df = df_full[(df_full['timestamp'] == case['timestamp']) & (df_full['rho'] == case['rho'])]
        case_df_base = df_full_base[(df_full['timestamp'] == case['timestamp'])]
        for col in case_df.columns.values:
            print(col)

        exp_data = case_df[['m', 'n', 'graph_no', 'exp_no', 'beta_dist', 'rho']].drop_duplicates()

        m = exp_data['m'].iloc[0]
        n = exp_data['n'].iloc[0]
        rho = exp_data['rho'].iloc[0]
        graph_no = exp_data['graph_no'].iloc[0]
        exp_no = exp_data['exp_no'].iloc[0]
        beta_dist = exp_data['beta_dist'].iloc[0]

        alpha_data = rho * case_df_base[['i', 'alpha']].drop_duplicates()
        beta_data = case_df_base[['j', 'beta']].drop_duplicates()



        print('graph_no:', graph_no, 'exp_no:', exp_no, 'beta_dist:', beta_dist)

        alpha = np.zeros(m)
        beta = np.zeros(n)
        compatability_matrix = np.zeros((m, n))
        matching_rates = np.zeros((m, n))
        rho_approx = np.zeros((m, n))
        heavy_approx = np.zeros((m, n))
        light_approx = np.zeros((m, n))

        for k, row in alpha_data.iterrows():
            alpha[int(row['i'])] = float(row['alpha'])

        for k, row in beta_data.iterrows():
            beta[int(row['j'])] = float(row['beta'])

        for k, row in case_df.iterrows():

            compatability_matrix[int(row['i']), int(row['j'])] = 1
            matching_rates[int(row['i']), int(row['j'])] = float(row['sim_matching_rates'])
            rho_approx[int(row['i']), int(row['j'])] = float(row['rho_approx'])
            heavy_approx[int(row['i']), int(row['j'])] = float(row['heavy_approx'])
            light_approx[int(row['i']), int(row['j'])] = float(row['light_approx'])
            print('{},{},{:.5f},{:.5f},{:.5f}'.format(row['i'], row['j'], row['sim_matching_rates'], row['heavy_approx'], np.abs(row['heavy_approx'] - row['sim_matching_rates'])))

        printarr(matching_rates.sum(axis=0), 'mr_sum_0')
        printarr(heavy_approx.sum(axis=0), 'ha_sum_0')
        printarr(matching_rates.sum(axis=1), 'mr_sum_1')
        printarr(heavy_approx.sum(axis=1), 'ha_sum_1')
        printarr(matching_rates.sum(), 'mr_sum')
        printarr(heavy_approx.sum(), 'ha_sum')
        printarr(matching_rates, 'sim_matching_rates')
        printarr(heavy_approx, 'heavy_approx')
        # printarr(light_approx, 'light_approx')


def sbpss_cd_graph1(filename='FZ_Kaplan_sbpss_cd_sum'):

    sum_res = pd.read_csv(filename + '.csv')

    fig, ax = plt.subplots(3,3)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    col_plt = {'one': 0, 'half': 1, 'rand': 2}


    approx_colors = {
        'rho_approx': 'green',
        'heavy_traffic_approx_entropy': 'red',
        'low_traffic_approx_entropy': 'blue'
    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.0315, 0.2955)
    }

    lines = []
    labels = []


    for key, grp in sum_res.groupby(by=['split', 'density_level' ,'approximation'], as_index=False):

        split, density_level, approximation = key
        color = approx_colors[approximation]
        row = row_plt[density_level]
        col = col_plt[split]

        x = grp['rho']

        if approximation == 'rho_approx':

            if row == 0 and col == 0:
                lines.append(ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=3))
                lines.append(ax[row, col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':'))
                lines.append(ax[row, col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':'))
                labels.append('Mixed_Entropy_Approximation')
            else:
                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=3, label='Mixed_Entropy_Approximation')
                ax[row, col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
                ax[row, col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'heavy_traffic_approx_entropy':

            if row == 0 and col == 0:
                lines.append(ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1))
                labels.append('Heavy Traffic Approximation')
            else:
                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1)

        elif approximation == 'low_traffic_approx_entropy':

            if row == 0 and col == 0:
                lines.append(ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1))
                lines.append(ax[row, col].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--'))
                lines.append(ax[row, col].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.'))
                lines.append(ax[row, col].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='--'))
                labels.append('low_traffic_approx_entropy')
                labels.append('Ohm Error for IMS')
                labels.append('QP Error for IMS')
                labels.append('Ent Error for IMS')

            else:
                ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Light Traffic Approximation')
                ax[row, col].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--')
                ax[row, col].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.')
                ax[row, col].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='--')

    for i,j in product(range(3), range(3)):
        ax[i,j].set_xlim(0, 1)
        ax[i,j].set_ylim(0, 0.6)


    print(lines)
    plt.figlegend(lines, labels, loc = 'lower center', ncol=1, labelspacing=0.1 )

    plt.show()


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'approximation']
    
    # fig, ax = plt.subplots(3,1)

    # row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    # approx_colors = {
    #     'rho_approx': 'green',
    #     'rho_2_approx': 'black',
    #     'heavy_approx': 'red',
    #     'light_approx': 'blue'
    # }

    # for key, grp in agg_res.groupby(by=base_cols, as_index=False):

    #     timestamp, graph_no, exp_no, m, n, density_level, beta_dist, approximation = key
    #     if beta_dist != 'unifrom' and approximation != 'rho_2_approx':
            

    #         color = approx_colors[approximation]
    #         row = row_plt[density_level]

    #         x = grp['rho']
    #         y = grp['err_pct_of_rate']

    #         ax[row].plot(x, y, color=color)

    # plt.show()


def sim_rates_vs_lamda(filename='FZ_Kaplan_exp_sbpss_cd4'):


    df = pd.read_csv(filename + '.csv')

    total_rates = df[['timestamp', 'rho', 'split', 'i', 'lamda']].drop_duplicates()[['timestamp','rho','split', 'lamda']].groupby(by=['timestamp','rho','split'], as_index=False).sum().rename(columns={'lamda':'total_lamda'})
    total_sim_rates = df[['timestamp','rho','split','sim_matching_rates']].groupby(by=['timestamp','rho','split'], as_index=False).sum().rename(columns={'sim_matching_rates':'total_sim_rates'})
    df = pd.merge(
            left=df, 
            right = pd.merge(left=total_rates, right=total_sim_rates, on=['timestamp', 'rho', 'split'], how='left'),
            on = ['timestamp', 'rho', 'split'],
            how='left'
        )

    df.to_csv('FZ_Kaplan_exp_sbpss_cd5.csv', index=False)


def increasing_n_res(filename='increasing_n_system'):

    df = pd.read_csv(filename + '.csv')
    df.loc[:, 'i=j'] = df['i'] == df['j']
    df_sum = df[['timestamp','m','n','i=j','matching_rates']].groupby(by=['timestamp', 'm','n','i=j'], as_index=False).sum()
    df_norm_sum = df[['timestamp','m','n','matching_rates']].groupby(by=['timestamp', 'm','n'], as_index=False).sum().rename(columns = {'matching_rates':'total_rate'})
    df_sum = pd.merge(left=df_sum, right=df_norm_sum, on =['timestamp','m','n'], how='left')
    df_sum.loc[:, 'matching_rates'] = df_sum['matching_rates']/df_sum['total_rate']
    df_sum = df_sum[:12]
    df_sum_eq = df_sum[df_sum['i=j']].rename(columns={'matching_rates':'i_i_rates'}).drop(columns=['i=j', 'total_rate'])
    df_sum_neq = df_sum[~df_sum['i=j']].rename(columns={'matching_rates':'i_j_rates'}).drop(columns=['i=j'])
    df_sum = pd.merge(left=df_sum_eq, right=df_sum_neq, on=['timestamp','m','n'], how='left')
    df_sum['i_j_rates'] = df_sum['i_i_rates'] + df_sum['i_j_rates']
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{xfrac}']
    mpl.rcParams['hatch.linewidth'] = 0.05

    fig = plt.figure()
    x_lim_l = 5
    x_lim_r = 100
    y_lim = 1
    ax1 = fig.add_subplot(111)
    ax1.plot(df_sum['m'], df_sum['i_i_rates'], marker='x', color='black')
    ax1.plot(df_sum['m'], df_sum['i_j_rates'], color='black')

    # ax1.plot([x_lim_l, x_lim_l], [0, 1], color='black')
    ax1.plot([x_lim_r, x_lim_r], [0, 1], color='black')
    
    ax1.fill_between(df_sum['m'], 0, df_sum['i_i_rates'], facecolor="grey", alpha=.2, hatch='....')
    ax1.fill_between(df_sum['m'], df_sum['i_i_rates'], df_sum['i_j_rates'], facecolor="grey", alpha=.5)
    ax1.set_xlim(5, 100.5)
    ax1.set_ylim(0,1.01)
    ax1.set_xticks([5,10,25,50, 75, 100])
    ax1.set_xticklabels(['5','10','25','50','75','100'])
    y_vals = [.25,.5,.75,1.]
    ax1.set_yticks(y_vals)
    ax1.set_yticklabels(['.25', '.5', '.75', '1'])
    ax1.set_xlabel('No. of Customer/Server Classes',fontsize=16)
    ax1.set_ylabel("Match Rate", fontsize=16)
    ax1.text(50, 0.4, '$E_{=}$', fontsize=35,
            horizontalalignment='center',
            verticalalignment='center',
            multialignment='center')

    ax1.text(85, .875, "$E_{\\neq}$", fontsize=35,
            horizontalalignment='center',
            verticalalignment='center',
            multialignment='center')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.show()



    
    df = pd.read_csv(filename + '.csv')

    

if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize, precision=5)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    # increasing_n_res()
    # ims_table('FZ_final_w_qp')
    sbpss_graph4()

    # comparison_table_grids()
    # growing_chains_graph()


