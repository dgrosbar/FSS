import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sys
from itertools import product
from utilities import printarr, printcols, calc_area_between_curves
from math import exp

prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']
MARKERS = ["o", "v", "*", "x", "H", "D", "s", "X", "P"]


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

    # 'alpha',
    # 'beta',
    # 'i',
    # 'j',
    # 'no_of_sims',
    # 'seed',
    # 'sim_duration',
    # 'sim_len',
    # 'warm_up',
    # 'timestamp',
    # 'max_edges',
    # 'edge_count',
    # 'edge_density',
    # 'm',
    # 'n',
    # 'exp_no',
    # 'structure',
    # 'arc_dist',
    # 'size',

    # 'sim_matching_rates',
    # 'sim_matching_rates_stdev',

    df = pd.read_csv(filename + '.csv')
    printcols(df)
    id_vars = [
        'timestamp',
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


def growing_chains_graph(filename='./Results/growing_chains_new2'):
    
    res_df = pd.read_csv(filename + '.csv')

    res_df.loc[:, 'arc_type'] = np.where(res_df['i']<=res_df['j'], res_df['j'] - res_df['i'], res_df['j'] + res_df['n'] - res_df['i'])

    def f(df):

        d = {}

        d['r'] = df['sim_matching_rates'].mean()
        d['sig_r'] = df['sim_matching_rates'].std()

        index = ['r', 'sig_r']

        return pd.Series(d, index=index)

    res_df = res_df[['arc_type','n','k', 'sim_matching_rates']].groupby(by=['n','k','arc_type'], as_index=False).apply(f).reset_index()
    res_df.loc[:, 'scv_r'] = (res_df['sig_r']/res_df['r'])**2
    res_df.loc[:, 'error_pct'] = res_df['r']*(res_df['n'] * res_df['k']) 
    res_df.loc[:, 'abs_error'] = np.abs(res_df['r'] - (1/(res_df['n'] * res_df['k']))) * res_df['n']
    
    res_agg = res_df[['n','k','abs_error']].groupby(by=['n','k'], as_index=False).sum()

    print(res_df)

    print(res_df[(1 == res_df['arc_type'] + 1)].sort_values(by=['k','n'])) 
    print(res_df[(1 == res_df['arc_type'] + 1)].sort_values(by=['n','k'])) 
    print(res_df[(res_df['k'] == res_df['arc_type'] + 1)]) 
    print(res_df[(res_df['k']-1)/2 == res_df['arc_type']])

    print(res_agg.sort_values(by=['k','n']))
    print(res_agg.sort_values(by=['n','k']))

    res_agg = res_agg[(res_agg['k']!=7) & (res_agg['n']!=10)]

    fig, (ax1, ax2) = plt.subplots(1,2)

    for v , (k, res) in enumerate(res_agg.groupby('k')):

        ax1.plot(res['n'], res['abs_error'], linewidth=1, marker=MARKERS[v] ,label='k='+str(k))
        ax1.set_xlabel('n', fontsize=24)
        ax1.set_ylabel('Sum of Absoulte Error', fontsize=16)
        ax1.legend()

    ax1.set_xticks([5, 15, 25, 50, 100, 150, 200])
    ax1.xaxis.grid(True)

    for v, (n, res) in enumerate(res_agg.groupby('n')): 

        ax2.plot(res['k'], res['abs_error'], linewidth=1, marker=MARKERS[v] ,label='n='+str(n))
        ax2.set_xlabel('k', fontsize=24)
        ax2.set_ylabel('Sum of Absoulte Error', fontsize=16)
        ax2.legend()

    ax2.set_xticks([3, 5, 7, 9, 11, 15, 19, 23, 31, 39])
    ax2.xaxis.grid(True)

    plt.show()

def ims_table(filename='./Results/FZ_final_w_qp'):


    df = pd.read_csv(filename + '.csv')

    id_vars = ['timestamp','density_level','graph_no','m','n','max_edges','edge_count','edge_density','exp_num','alph_dist','beta_dist','utilization','exact','i','j','alpha','beta','exact_matching_rate','sim_matching_rate', 'no_of_sims', 'sim_matching_rate_stdev']
    
    val_vars = ['ohm_approx','entropy_approx','quad_approx']
    
    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

    df.loc[:, 'abs_error'] = np.abs(df['approx_match_rate'] - df['exact_matching_rate'])
    df.loc[:, 'abs_error_pct'] = np.abs(df['approx_match_rate'] - df['exact_matching_rate'])/df['exact_matching_rate']
    df.loc[:, 'abs_error_abs'] = np.abs(np.abs(df['approx_match_rate']) - df['exact_matching_rate'])

    def f(df):


        d = {}
        

        d['sum_abs_error'] = df['abs_error'].sum()
        d['mean_abs_error'] = df['abs_error'].mean()
        d['mean_abs_error_abs'] = df['abs_error_abs'].mean()
        d['max_abs_error'] = df['abs_error'].max()
        d['mean_abs_error_pct'] = df['abs_error_pct'].mean()
        d['max_abs_error_pct'] = df['abs_error_pct'].max()
        d['min_match_rate'] = df['approx_match_rate'].min()
        d['negatvie_flows'] = ((df['approx_match_rate'] < 0)).sum()
        d['sum_negatvie_flow'] = ((df['approx_match_rate'] < 0) * (-1 * df['approx_match_rate'])).sum()

        index = [
            'negatvie_flows',
            'sum_negatvie_flow',
            'sum_abs_error',
            'mean_abs_error',
            'max_abs_error',
            'mean_abs_error_pct',
            'max_abs_error_pct',
            'mean_abs_error_abs'
        ]

        return pd.Series(d, index=index)

    base_cols = ['timestamp', 'graph_no', 'exp_num', 'density_level', 'beta_dist', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    for col in agg_res.columns.values:
        print(col)

    # agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_agg.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error']
   
    def g(df):
        
        d = {}

        d['mean_max_abs_error'] = df['max_abs_error'].mean()
        d['avg_mean_abs_error'] = df['mean_abs_error'].mean()
        d['avg_mean_abs_error_abs'] = df['mean_abs_error_abs'].mean()
        d['mean_sum_negative_flow'] = df['sum_negatvie_flow'].mean()
        d['negative_flows'] = df['negatvie_flows'].mean()

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_sum_negative_flow',
            'negative_flows',
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_95_u',
            'err_pct_95_l',
            'mean_max_abs_error',
            'avg_mean_abs_error',
            'avg_mean_abs_error_abs'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate', 'max_abs_error', 'mean_abs_error', 'negatvie_flows', 'sum_negatvie_flow', 'mean_abs_error_abs']].sort_values(by=['approximation', 'density_level'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['approximation', 'density_level']))

    print(sum_res.pivot(index='approximation', columns='density_level', values=['mean_err_pct','max_err_pct' ,'mean_max_abs_error', 'err_pct_95_u', 'err_pct_95_l']))

    # sum_res.sort_values(by=['approximation', 'density_level']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_sum.csv', index=False)


def ims_table2(filename='./Results/FZ_final_w_qp'):


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

    base_cols = ['timestamp', 'graph_no', 'exp_num', 'density_level', 'beta_dist']
    agg_res = df.groupby(by=base_cols + ['approximation'], as_index=False).apply(f).reset_index()
    # agg_res = agg_res.set_index(base_cols)
    agg_res = agg_res.pivot_table(index=base_cols, columns='approximation', values=['sum_abs_error', 'mean_abs_error', 'max_abs_error', 'mean_abs_error_pct', 'max_abs_error_pct'])
    # agg_res = agg_res.reset_index()
    agg_res_sum = agg_res['sum_abs_error']
    agg_res_sum.reset_index(inplace=True) 
    print(agg_res_sum)

    agg_res_sum.loc[:, 'entropy'] = 1. * (agg_res_sum['entropy_approx'] < agg_res_sum['quad_approx'])
    agg_res_sum.loc[:, 'quad'] = 1. * (agg_res_sum['entropy_approx'] > agg_res_sum['quad_approx'])
    agg_res_sum.loc[:, 'diff'] = np.abs(agg_res_sum['entropy_approx'] - agg_res_sum['quad_approx'])
    agg_res_sum.loc[:, 'diff_entropy'] = agg_res_sum['diff'] * agg_res_sum['entropy']
    agg_res_sum.loc[:, 'diff_quad'] = agg_res_sum['diff'] * agg_res_sum['quad']

    def g(df):

        d = {}

        d['quad<entropy'] = df['quad'].sum()
        d['entropy<quad'] = df['entropy'].sum()
        d['avg_quad_diff'] = df['diff_quad'].sum()/df['quad'].sum()
        d['avg_entropy_diff'] = df['diff_entropy'].sum()/df['entropy'].sum()
        d['max_quad_diff'] = df['diff_quad'].max()
        d['max_entropy_diff'] = df['diff_entropy'].max()
        d['quad_max_sum_err'] = df['quad_approx'].max()
        d['entropy_max_sum_err'] = df['entropy_approx'].max()

        index = [
            'quad<entropy',
            'max_entropy_diff',
            'avg_entropy_diff',
            'entropy_max_sum_err', 
            'entropy<quad',
            'max_quad_diff',
            'avg_quad_diff',
            'quad_max_sum_err'
        ]

        return pd.Series(d, index=index)

    sum_res = agg_res_sum.groupby(by='density_level').apply(g)
    # sum_res.loc[:,'diff_entropy'] = sum_res['diff_entropy']/sum_res['entropy<quad']
    # sum_res.loc[:,'diff_quad'] = sum_res['diff_quad']/sum_res['quad<entropy']
    print(sum_res)
    print(sum_res.to_latex())

    # agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_agg.csv', index=False)

    # agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error']
   
    # def g(df):
        
    #     d = {}

    #     d['mean_max_abs_error'] = df['max_abs_error'].mean()


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
    #         'err_pct_95_l',
    #         'mean_max_abs_error'
    #     ]

    #     return pd.Series(d, index=index) 

    # sum_base_cols = ['density_level', 'approximation']

    # sum_res = agg_res[sum_base_cols + ['err_pct_of_rate', 'max_abs_error']].sort_values(by=['approximation', 'density_level'])
    # sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    # print(sum_res.sort_values(by=['approximation', 'density_level']))

    # print(sum_res.pivot(index='approximation', columns='density_level', values=['mean_err_pct','max_err_pct' ,'mean_max_abs_error', 'err_pct_95_u', 'err_pct_95_l']))

    # sum_res.sort_values(by=['approximation', 'density_level']).to_csv('FZ_Kaplan_exp_sbpss_good_w_alis_sum.csv', index=False)


def alis_table(filename='FZ_Kaplan_exp_pure_alis'):

    df = pd.read_csv(filename + '.csv')

    for col in df.columns.values:
        print(col)

    id_vars = ['timestamp','density_level','graph_no','exp_no','beta_dist','i','j','alpha','beta','sim_matching_rates', 'no_of_sims', 'sim_matching_rates_stdev']

    print(df[id_vars].head())
    
    val_vars = ['alis_approx']
    
    df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')



    df.loc[:, 'abs_error'] = np.abs(df['approx_match_rate'] - df['sim_matching_rates'])
    df.loc[:, 'abs_error_pct'] = np.abs(df['approx_match_rate'] - df['sim_matching_rates'])/df['sim_matching_rates']

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

    base_cols = ['timestamp', 'graph_no', 'exp_no', 'density_level', 'beta_dist', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error']

    print(agg_res)

   
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
            'mean_max_abs_error',
            'avg_mean_abs_error'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate', 'max_abs_error', 'mean_abs_error']].sort_values(by=['approximation', 'density_level'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['approximation', 'density_level']))

    print(sum_res.pivot(index='approximation', columns='density_level', values=['mean_err_pct','max_err_pct' ,'mean_max_abs_error', 'err_pct_95_u', 'err_pct_95_l']))


    # base_cols = ['timestamp', 'graph_no', 'exp_no', 'density_level', 'beta_dist']
    # agg_res = df.groupby(by=base_cols + ['approximation'], as_index=False).apply(f).reset_index()
    # # agg_res = agg_res.set_index(base_cols)
    # agg_res = agg_res.pivot_table(index=base_cols, columns='approximation', values=['sum_abs_error', 'mean_abs_error', 'max_abs_error', 'mean_abs_error_pct', 'max_abs_error_pct'])
    # # agg_res = agg_res.reset_index()
    # agg_res_sum = agg_res['sum_abs_error']
    # agg_res_sum.reset_index(inplace=True) 
    # print(agg_res_sum)


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

    sum_res = pd.read_csv('./Results/' + filename + '.csv')


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

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=16)
    fig.suptitle('Graph Density', fontsize=24)
    for i in range(3):
        ax[i].set_xlabel('utilization', fontsize=16)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0.001, .5)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [0, 4, 5, 6, 7, 1, 2, 3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)

    plt.show()


def sbpss_cd_table1(filename='FZ_Kaplan_exp_sbpss_cd_lqf5'):

    df = pd.read_csv(filename + '.csv')

    base_cols = ['timestamp','rho', 'split', 'policy']

    total_rates = df[base_cols + ['i', 'lamda']].drop_duplicates()[base_cols +['lamda']].groupby(by=base_cols, as_index=False).sum().rename(columns={'lamda':'total_lamda'})
    total_sim_rates = df[base_cols + ['sim_matching_rates']].groupby(by=base_cols, as_index=False).sum().rename(columns={'sim_matching_rates':'total_sim_rates'})
    df = pd.merge(
            left=df, 
            right = pd.merge(left=total_rates, right=total_sim_rates, on=base_cols, how='left'),
            on = base_cols,
            how='left'
        )

    df.loc[:, 'sim_rate_gap'] = np.abs(df['total_lamda'] - df['total_sim_rates'])
    df.loc[:, 'sim_adj'] = df['total_lamda'] / df['total_sim_rates']

    df.to_csv(filename + '_rates_x.csv', index=False)

    df = df[df['sim_rate_gap'] < 0.03]
    df.loc[:,'adj_sim_matching_rates'] = df.loc[:, 'sim_adj'] * df['sim_matching_rates']

    id_vars = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'adj_sim_matching_rates', 'sim_rate_gap', 'split', 'policy']
    val_vars = ['fcfs_alis_approx', 'fcfs_approx', 'alis_approx']

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

    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'split' ,'approximation', 'policy']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho', 'split']).to_csv(filename + '_sum.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']
   
    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_of_rate_std'] = df['err_pct_of_rate'].std()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_of_rate_std',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'rho', 'approximation', 'split', 'policy']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split']))

    sum_res.sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split']).to_csv(filename + '_agg.csv', index=False)


def sbpss_cd_table2x(filename='FZ_Kaplan_exp_sbpss_cd_w_lqf2'):

    df = pd.read_csv(filename + '.csv')

    base_cols = ['timestamp','rho','split', 'policy']

    total_rates = df[base_cols + ['i', 'lamda']].drop_duplicates()[base_cols +['lamda']].groupby(by=base_cols, as_index=False).sum().rename(columns={'lamda':'total_lamda'})
    total_sim_rates = df[base_cols + ['sim_matching_rates']].groupby(by=base_cols, as_index=False).sum().rename(columns={'sim_matching_rates':'total_sim_rates'})

    df = pd.merge(
            left=df, 
            right = pd.merge(left=total_rates, right=total_sim_rates, on=base_cols, how='left'),
            on = base_cols,
            how='left'
        )

    total_sim_rates_i = df[base_cols + ['i','sim_matching_rates']].groupby(by=base_cols + ['i'], as_index=False).sum().rename(columns={'sim_matching_rates':'total_sim_rate_i'})
    df = pd.merge(left=df, right = total_sim_rates_i, on=base_cols + ['i'], how='left')

    df.loc[:, 'sim_rate_gap'] = np.abs(df['total_lamda'] - df['total_sim_rates'])
    
    df.loc[:, 'sim_adj'] = df['lamda'] / df['total_sim_rate_i']

    df.to_csv(filename + '_rates_x.csv', index=False)

    df = df[df['sim_rate_gap'] < 0.03]
    df.loc[:,'adj_sim_matching_rates'] = df.loc[:, 'sim_adj'] * df['sim_matching_rates']

    id_vars = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'adj_sim_matching_rates', 'sim_rate_gap', 'split', 'policy']
    val_vars = ['fcfs_alis_approx', 'fcfs_approx', 'alis_approx']

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

    base_cols = ['timestamp', 'graph_no', 'exp_no', 'm', 'n', 'density_level', 'beta_dist', 'rho', 'split' ,'approximation', 'policy']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.sort_values(by=['approximation', 'graph_no', 'exp_no', 'beta_dist','density_level', 'rho', 'split']).to_csv('FZ_Kaplan_sbpss_cd_agg_w_alis_lqf_y.csv', index=False)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']
   
    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_of_rate_std'] = df['err_pct_of_rate'].std()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_of_rate_std',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['density_level', 'rho', 'approximation', 'split', 'policy']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split']))

    sum_res.sort_values(by=['policy', 'approximation', 'density_level', 'rho', 'split']).to_csv('FZ_Kaplan_sbpss_cd_sum_w_alis_lqf_y.csv', index=False)


def sbpss_table3(filename='erdos_renyi_sbpss_uni_mu_comp_alis_rates'):

    # df = pd.read_csv(filename + '.csv')

    # base_cols = ['timestamp', 'size', 'exp_no', 'rho', 'policy']

    # total_rates = df[base_cols + ['i', 'lamda']].drop_duplicates()[base_cols +['lamda']].groupby(by=base_cols, as_index=False).sum().rename(columns={'lamda':'total_lamda'})
    # total_sim_rates = df[base_cols + ['sim_matching_rates']].groupby(by=base_cols, as_index=False).sum().rename(columns={'sim_matching_rates':'total_sim_rates'})
    # df = pd.merge(
    #         left=df, 
    #         right = pd.merge(left=total_rates, right=total_sim_rates, on=base_cols, how='left'),
    #         on = base_cols,
    #         how='left'
    #     )

    # df.loc[:, 'sim_rate_gap'] = df['total_lamda'] - df['total_sim_rates']
    # df.loc[:, 'sim_adj'] = df['total_lamda'] / df['total_sim_rates']


    # df.loc[:,'adj_sim_matching_rates'] = df['sim_adj'] * df['sim_matching_rates']

    df = pd.read_csv(filename + '.csv')
    df = df[df['policy']=='fcfs_alis']

    df_slim = df[['timestamp', 'size', 'exp_no', 'rho' ,'i', 'j', 'sim_matching_rates', 'fcfs_alis_approx', 'fcfs_approx', 'alis_approx']]

    id_vars = ['timestamp', 'size', 'exp_no', 'rho' ,'i', 'j', 'adj_sim_matching_rates']
    val_vars = ['fcfs_alis_approx', 'fcfs_approx', 'alis_approx']

    df = pd.melt(df_slim, id_vars=id_vars, value_vars=val_vars, var_name='approximation', value_name='approx_match_rate')

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

    base_cols = ['timestamp', 'size', 'exp_no', 'rho', 'approximation']
    agg_res = df.groupby(by=base_cols, as_index=False).apply(f).reset_index()

    agg_res.sort_values(by=['approximation', 'timestamp', 'size','exp_no', 'rho']).to_csv(filename + '_agg.csv', index=False)
    print(agg_res)

    agg_res.loc[:, 'err_pct_of_rate'] = agg_res['sum_abs_error_sim']/agg_res['total_rate']
   
    def g(df):
        
        d = {}

        d['mean_err_pct'] = df['err_pct_of_rate'].mean()
        d['max_err_pct'] = df['err_pct_of_rate'].max()
        d['min_err_pct'] = df['err_pct_of_rate'].min()
        d['err_pct_of_rate_std'] = df['err_pct_of_rate'].std()
        d['err_pct_95_u'] = df['err_pct_of_rate'].mean() + 1.96 * df['err_pct_of_rate'].std()
        d['err_pct_95_l'] = df['err_pct_of_rate'].mean() - 1.96 * df['err_pct_of_rate'].std()

        index = [
            'mean_err_pct',
            'max_err_pct',
            'min_err_pct',
            'err_pct_of_rate_std',
            'err_pct_95_u',
            'err_pct_95_l'
        ]

        return pd.Series(d, index=index) 

    sum_base_cols = ['size', 'rho', 'approximation']

    sum_res = agg_res[sum_base_cols + ['err_pct_of_rate']].sort_values(by=['size', 'rho', 'approximation'])
    sum_res = sum_res.groupby(by=sum_base_cols, as_index=False).apply(g).reset_index()
    
    print(sum_res.sort_values(by=['size', 'rho', 'approximation']))

    sum_res.sort_values(by=['size', 'rho', 'approximation']).to_csv(filename + '_sum.csv', index=False)


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


def sbpss_cd_graph1_lqf(policy, split, filename='FZ_Kaplan_sbpss_cd_sum_w_alis_lqf'):

    sum_res = pd.read_csv(filename + '.csv')
    print(sum_res)
    sum_res = sum_res[(sum_res['policy'] == policy) & (sum_res['split'] == split)]



    print(sum_res)

    fig, ax = plt.subplots(1, 3)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        'fcfs_alis_approx': 'green',
        'fcfs_approx': 'red',
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

        if approximation == 'fcfs_alis_approx':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation', marker='x')
            ax[row].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'fcfs_approx':

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

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=16)
    fig.suptitle('Graph Density', fontsize=24)
    for i in range(3):
        ax[i].set_xlabel('utilization', fontsize=16)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0.001, .5)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [0, 4, 5, 6, 7, 1, 2, 3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)

    plt.show()


def sbpss_cd_graph1_lqf_both(split, filename='./Results/FZ_Kaplan_sbpss_cd_sum_w_alis_lqf_x'):

    sum_res = pd.read_csv(filename + '.csv')
    print(sum_res)
    sum_res = sum_res[(sum_res['split'] == split)]



    print(sum_res)

    fig, ax = plt.subplots(1, 3)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        ('fcfs_alis_approx', 'fcfs_alis'): COLORS[0],
        ('fcfs_approx', 'fcfs_alis'): COLORS[1],
        ('alis_approx', 'fcfs_alis'): COLORS[2],
        ('fcfs_alis_approx', 'lqf_alis'): COLORS[3],
        ('fcfs_approx', 'lqf_alis'): 'red',
        ('alis_approx', 'lqf_alis'): 'blue'
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


    for key, grp in sum_res.groupby(by=['density_level', 'approximation', 'policy'], as_index=False):

        density_level, approximation, policy = key
        color = approx_colors[(approximation, policy)]
        row = row_plt[density_level]
        ax[row].set_title(cap_density_level[density_level])
        x = grp['rho']

        if approximation == 'fcfs_alis_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - FCFS-ALIS', marker='x')
            ax[row].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        if approximation == 'fcfs_alis_approx' and policy == 'lqf_alis':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - LQF-ALIS', marker='x')
            ax[row].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'fcfs_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='FCFS Approximation', marker = '.', linestyle='--')

        # elif approximation == 'rho_approx':

        #     ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Old  Approximation')

        # elif approximation == 'light_approx':

        #     ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation')

        elif approximation == 'alis_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation', marker='+', linestyle='-.')
            ax[row].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][2]]*len(x), color='green', linewidth=1, linestyle='-', label='MaxEnt Error for IMS')

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=16)
    fig.suptitle('Graph Density', fontsize=24)
    for i in range(3):
        ax[i].set_xlabel('utilization', fontsize=16)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0.001, .5)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [4,5,6,7,8,9,10,0,1,2,3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)

    plt.show()


def sbpss_approx_graph(filename='erdos_renyi_sbpss_comp_alis_rates_sum'):

    sum_res = pd.read_csv(filename + '.csv')
    sum_res.loc[:,'policy'] = 'fcfs_alis'
    print(sum_res)
    # sum_res = sum_res[(sum_res['split'] == split)]



    print(sum_res)

    fig, ax = plt.subplots(1, 2)

    col_plt = {100: 0, 1000: 1, '9x9':0,'30x30':1}
    
    approx_colors = {
        ('fcfs_alis_approx', 'fcfs_alis'): COLORS[0],
        ('fcfs_approx', 'fcfs_alis'): COLORS[1],
        ('alis_approx', 'fcfs_alis'): COLORS[2],
        ('fcfs_alis_approx', 'lqf_alis'): COLORS[3],
        ('fcfs_approx', 'lqf_alis'): 'red',
        ('alis_approx', 'lqf_alis'): 'blue'
    }

    ims_errors = {
        'low': (.115, .064, 0.028),
        'medium': (.089, .062, 0.034),
        'high': (.032, 0.0315, 0.02955)
    }

    # cap_density_level = {
    #     'low': 'Low',
    #     'medium': 'Medium',
    #     'high': 'High'
    # }


    for key, grp in sum_res.groupby(by=['size', 'approximation', 'policy'], as_index=False):

        size, approximation, policy = key
        color = approx_colors[(approximation, policy)]
        col = col_plt[size]
        try:
            ax[col].set_title(str(int(size)) + 'x' + str(int(size)))
        except:
            ax[col].set_title('(' + size + ')x(' + size + ')')
        x = grp['rho']

        if approximation == 'fcfs_alis_approx' and policy == 'fcfs_alis':

            ax[col].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - FCFS-ALIS', marker='x')
            ax[col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        if approximation == 'fcfs_alis_approx' and policy == 'lqf_alis':

            ax[col].plot(x, grp['mean_err_pct'], color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - LQF-ALIS', marker='x')
            ax[col].plot(x, grp['err_pct_95_u'], color=color, linewidth=.5, linestyle = ':')
            ax[col].plot(x, grp['err_pct_95_l'], color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'fcfs_approx' and policy == 'fcfs_alis':

            ax[col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='FCFS Approximation', marker = '.', linestyle='--')

        elif approximation == 'alis_approx' and policy == 'fcfs_alis':

            ax[col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation', marker='+', linestyle='-.')
            ax[col].plot(x, [ims_errors['low'][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
            ax[col].plot(x, [ims_errors['low'][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
            ax[col].plot(x, [ims_errors['low'][2]]*len(x), color='black', linewidth=1, linestyle='-', label='MaxEnt Error for IMS')

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=16)
    fig.suptitle('Graph Density', fontsize=24)
    for i in range(2):
        ax[i].set_xlabel('utilization', fontsize=16)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0.001, .5)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [4,5,6,7,0,1,2,3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)
    # plt.legend()

    plt.show()


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

    df.loc[:, 'sim_rate_gap'] = df['total_lamda'] - df['total_sim_rates']
    df.loc[:, 'sim_adj'] = df['total_lamda'] / df['total_sim_rates']

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


def ot_table(filename=''):

    base_cols= ['policy','rho','timestamp','m','n','exp_no','size','structure']
    df = pd.read_csv(filename + '.csv')
    df_i = df[['i', 'lamda','sim_waiting_times', 'sim_waiting_times_stdev'] + base_cols].drop_duplicates()
    df_i.loc[:, 'wt_x_r'] = df_i['lamda'] * df_i['sim_waiting_times']
    df_i.loc[:, 'wt_x_r_stdev'] = df_i['lamda'] * df_i['sim_waiting_times_stdev']
    df_wt = df_i[base_cols +['wt_x_r', 'wt_x_r_stdev']].groupby(by=base_cols, as_index=False).sum().rename(columns={'wt_x_r': 'wt', 'wt_x_r_stdev': 'wt_stdev'})
    df.loc[:, 'r_c'] = df['sim_matching_rates'] * df['c']
    df_c = df[base_cols + ['r_c']].groupby(by=base_cols, as_index=False).sum()
    df_res = pd.merge(left=df_wt, right=df_c, how='left', on=base_cols)
    df_res.to_csv(filename +'_i', index=False)


def ot_graph(filename='ot_sbpss_res', dl='low', exp_no=12, beta_dist='exponential', c_type='dist', rho=0.9, log_scale=False, rhos=None):

    df= pd.read_csv(filename + '.csv')
    dl = 'low'
    exp_no = 12
    beta_dist = 'exponential'
    c_type = 'dist'
    rhos = [0.9] if rhos is  None else rhos

    base_cols = ['density_level', 'graph_no', 'exp_no', 'beta_dist', 'c_type', 'rho', 'policy', 'gamma']
    fig, ax = plt.subplots(1, 1)

    df = df[df['density_level'] == dl]
    df = df[df['exp_no'] == exp_no]
    df = df[df['beta_dist'] == beta_dist]
    df = df[df['c_type'] == c_type]
    df = pd.concat([df[df['rho'] == rho] for rho in rhos])

    colors = {'fcfs_ot': 'red', 'fcfs_weighted_ot': 'blue', 'greedy': 'black', 'fcfs_alis': 'black'}

    if log_scale:

        max_wt = np.log(max(df[df['policy'] != 'greedy']['wt']))
        min_wt = np.log(min(df[df['policy'] != 'greedy']['wt']))
        min_c = min(np.log(df[df['policy'] == 'greedy']['r_c']))

    else: 

        max_wt = max(df[df['policy'] != 'greedy']['wt'])
        min_wt = min(df[df['policy'] != 'greedy']['wt'])
        min_c = min(df[df['policy'] == 'greedy']['r_c'])

    for (policy, rho), exp in df.groupby(by=['policy', 'rho'], as_index=False):

        if rho in [0.9]:
            if policy != 'greedy':
                if log_scale:
                    x = list(np.log(exp['wt']))
                    y = list(np.log(exp['r_c']))
                else:
                    x = list(exp['wt'])
                    y = list(exp['r_c'])
                ax.scatter(x, y, color = colors[policy])
                if rho == 0.6:
                    ax.plot(x, y,  label=policy)
                else:
                    ax.plot(x, y, color = colors[policy])

                for i, txt  in enumerate(exp['gamma']):
                    if i % 1 == 0:
                        ax.annotate("%.2f" % txt, (x[i], y[i]))

            if policy == 'greedy' and rho ==0.6:
                ax.plot([min_wt, max_wt],[min_c, min_c] , label='greedy cost', linestyle = '--',color = colors[policy])


            ax.set_xlabel('log(Waiting Time)', fontsize=16)
            ax.set_ylabel('log(Cost)', fontsize=16)

    ax.set_xlim(min_wt-1, max_wt+1)



    plt.legend()

    plt.show()
    # print(df[['gamma']].drop_duplicates())
    # df_i = df[['i', 'lamda', '','sim_waiting_times'] + base_cols].drop_duplicates()
    # df_i.loc[:, 'wt_x_r'] = df_i['lamda']*df_i['sim_waiting_times']
    # df_wt = df_i[base_cols +['wt_x_r']].groupby(by=base_cols, as_index=False).sum().rename(columns={'wt_x_r': 'wt'})
    # df.loc[:, 'r_c'] = df['sim_matching_rates'] * df['c']
    # df_c = df[base_cols + ['r_c']].groupby(by=base_cols, as_index=False).sum()
    # df_res = pd.merge(left=df_wt, right=df_c, how='left', on=base_cols)
    # print(df.sort_values(by=base_cols))


def batching_window():

    x = np.array(range(70))
    y = np.array([0.2 + 1-exp(-i*0.1) for i in range(70)])
    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y, linewidth=3, color='black')
    ax.plot(x, [1.22]*70, linewidth=1, color='black', linestyle='--')

    ax.set_xlabel('Batching Window Size', fontsize=16)
    ax.set_ylabel('Revenue Rate', fontsize=16)

    ax.set_xticks([0, 69])
    ax.set_xticklabels(['0', 'T'], fontsize=16)
    ax.set_yticks([0, 1.22])
    ax.set_yticklabels(['', ''], fontsize=16)

    ax.text(0.01, 0.1, 'Non Idling Policy', fontsize=14, horizontalalignment='left')
    ax.text(45, 1.23, 'Optimal Transport', fontsize=14, horizontalalignment='center')
    ax.set_ylim(0, 1.5)
    ax.set_xlim(-1, 71)

    plt.show()


def sbpss_gini_cum(filename, cost=False):


    base_cols= ['policy','rho','timestamp','m','n','exp_no','size','structure']
    
    df = pd.read_csv(filename + '.csv')
    df = df[(df['exp_no'] == 1) & (df['policy'] == 'weighted_fcfs_alis') & (df['n'] == 81) & ((df['rho']==0.9) | (df['rho']==0.95))]
    df_i = df[['i', 'lamda','sim_waiting_times', 'sim_waiting_times_stdev'] + base_cols].drop_duplicates()
    df_i.loc[:, 'lamda_x_sim_waiting_times'] = df_i['lamda'] * df_i['sim_waiting_times']/df_i['rho']
    df_i = df_i.sort_values(by=base_cols + ['sim_waiting_times'])
    df_cum = df_i[base_cols + ['lamda','lamda_x_sim_waiting_times']]
    df_cum = df_cum.groupby(by=base_cols, as_index=False).cumsum(axis=0)
    df_cum = df_cum.rename(columns={'lamda_x_sim_waiting_times': 'cum_sim_waiting_timnes', 'lamda': 'cum_lamda'})
    df_i = df_i.join(df_cum)
    df_i.loc[:, 'cum_lamda'] = df_i['cum_lamda']/df_i['rho']
    df_i.loc[:, 'lamda_x_sim_waiting_times'] = df_i['lamda'] * df_i['sim_waiting_times']
    df_i = df_i.sort_values(by=base_cols + ['sim_waiting_times'])
    df_cum = df_i[base_cols + ['lamda','lamda_x_sim_waiting_times']]
    df_cum = df_cum.groupby(by=base_cols, as_index=False).cumsum(axis=0)
    df_cum = df_cum.rename(columns={'lamda_x_sim_waiting_times': 'cum_sim_waiting_timnes', 'lamda': 'cum_lamda'})
    df_i = df_i.join(df_cum)


    fig, ax = plt.subplots(1, 2)
    i = -1
    j = -1
    k = None
    lines = [[], []]
    color = COLORS[0]
    util = False
    fill_between = False
    gini_curve = True

    res_dic = dict()
    for key, grp in reses_i.groupby(base_cols):

        grp = grp.sort_values(by=['cum_MR_i_sim'])
        key_dict = dict(zip(base_cols, key))
        print(key_dict)
        if ('FIFO' in key_dict['sim_name'] or 'MW' in key_dict['sim_name'] or 'prio' in key_dict['sim_name']) \
                and key_dict[n_name] in [5, 10, 50, 100, 500]:

            if 'MW' in key_dict['sim_name']:
                k = 0
                i += 1
                h = i
            elif 'FIFO' in key_dict['sim_name'] or 'prio' in key_dict['sim_name']:
                k = 1
                j += 1
                h = j

            if util:
                norm_j = grp['norm_j']
                rho_j = 1. - grp['r_j_sim']
                ax[k].plot(norm_j, rho_j, color='black', linewidth=.5,
                       linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[1]))
                # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                ax[k].set_xlabel('Server', fontsize=16)
                ax[k].set_ylabel('Utilization', fontsize=16)
                ax[k].set_xticks([0, 0.5, 1.])
                ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=16)
                ax[k].set_ylim((0, 1))

                ax[k].legend(title='')
            else:

                cum_wt_i_sim = np.append(np.array([0]), grp['cum_WT_i_sim'])
                cum_mr_i_sim = np.append(np.array([0]), grp['cum_MR_i_sim'])
                #print( grp[['i', 'WT_i_sim', 'cum_MR_i_sim']])
                #print( cum_mr_i_sim)
                wt_i_sim = np.append(np.array([0]), grp['WT_i_sim'])
                max_cum_wt = np.amax(cum_wt_i_sim)
                max_wt = np.amax(wt_i_sim)
                area_1 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
                area_2 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
                area_3 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
                area_4 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim/max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
                gini1 = (area_1 - area_2)/area_1
                gini2 = (area_3 - area_4)/area_3
                print(key)
                res_dic[key_dict['sim_name']] = {'avg. WT': max_cum_wt, 'gini1': gini1,'gini2': gini2, 'worst':max_wt }
                print( max_cum_wt)
                print( "{:.0%}".format((area_1 - area_2)/area_1))
                # sort_wt_i_sim = np.sort(wt_i_sim)
                # nn = sort_wt_i_sim.shape[0]
                # gini_score = (2*((np.arange(1, nn + 1, 1)*sort_wt_i_sim).sum()))/(nn * sort_wt_i_sim.sum()) - ((nn+1)/nn)
                # print( "{:.0%}".format(gini_score))

                if fill_between:
                    ax[k].fill_between(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, cum_wt_i_sim, color=color, alpha=0.2)
                    ax[k].fill_between(cum_mr_i_sim, 0, cum_wt_i_sim,  color=color, label=key)
                if gini_curve:
                        if key_dict['sim_name'] in ['rho_weighted_j_MW', 'rho_weighted_j_FIFO', 'plain_FIFO', 'plain_MW']:
                            color = 'blue' if key_dict['sim_name'] in ['rho_weighted_j_MW', 'rho_weighted_j_FIFO'] else 'red'
                            ax[k].plot(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, color='black', linewidth=.1)
                            ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color=color,linewidth=.5,
                                       linestyle='-', marker=MARKERS[h%len(MARKERS)], markersize=3,
                                       label=key_dict['sim_name'] + '-' + str(key_dict['rho']))
                            if 'prio' in key_dict['sim_name']:
                                ax[0].plot(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, color='black', linewidth=.1)
                                ax[0].plot(cum_mr_i_sim, cum_wt_i_sim, color='black',linewidth=.5,
                                           linestyle='-', marker=MARKERS[(h)%len(MARKERS)], markersize=3,
                                           label=key_dict['sim_name'] + '-' + str(key_dict['rho']))

                    # ax[k].plot(cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                    #            linestyle='-', marker=MARKERS2[h],markersize=4,label='n=' + str(key[1]))
                    #print( 1.-cum_mr_i_sim)
                else:
                    lines[k] += ax[k].plot(1.-cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                           linestyle='-', marker=MARKERS[h%len(MARKERS)], markersize=4,
                                           label=key_dict['sim_name'] + '-' + str(key_dict['rho']))

                # lines[k] += ax[k].plot(np.arange(0,1.2,0.2), wt_sim * np.ones(6), color='black', linewidth=.5,
                #                        linestyle=LINE_STYLES['dashed'], marker=MARKERS2[h], markersize=4)

                for k in [0, 1]:

                    # leg = Legend(ax[k], [lines[k][g] for g in range(len(lines[k])) if g%2 == 1],[" ,",", ",", ",", "],
                    #              loc='upper left', bbox_to_anchor=(.7, .95), frameon=True, title='Total')
                    # leg._legend_box.align = "left"
                    # ax[k].add_artist(leg)

                    # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                    ax[k].set_xlabel('Customer Class', fontsize=16)
                    ax[k].set_ylabel('Avg. Waiting Time', fontsize=16)
                    ax[k].set_xticks([0, 0.5, 1.])
                    #ax[k].set_yticks([2*g for g in range(9)])
                    ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=16)
                    #ax[k].set_ylim((0, 18))
                    ax[k].legend()

                    # leg2 = ax[k].legend(title='Class', frameon=True, bbox_to_anchor=(1.,.95), )
                    # leg2._legend_box.align = "left"

                    # ax[0].set_title(r"LQF-ALIS," r"$\quad \rho=.95$", fontsize=16, color='black')
                    # ax[1].set_title(r"FIFO-ALIS," r"$\quad\rho=.95$", fontsize=16, color='black')
    for key, val in sorted(res_dic.iteritems()):
        if 'FIFO' in key or 'prio' in key:
            print( key, [(key2, ':', val2) for key2, val2 in val.iteritems()])
    print( '-------------------------------------------------')
    print( '-------------------------------------------------')
    for key, val in sorted(res_dic.iteritems()):
        if 'MW' in key or 'prio' in key:
            print( key, [(key2, ':', val2) for key2, val2 in val.iteritems()])
    if show:
        plt.show()


def sbpss_gini_score(filename, base_cols ,cost=False):

    
    df = pd.read_csv(filename + '.csv')
    df['structure'] = 'grid'
    df_i = df[base_cols + ['i', 'lamda','sim_waiting_times', 'sim_waiting_times_stdev'] ].drop_duplicates()
    df_i.loc[:, 'lamda_x_sim_waiting_times'] = df_i['lamda'] * df_i['sim_waiting_times']/df_i['rho']
    df_i = df_i.sort_values(by=base_cols + ['sim_waiting_times'])
    df_cum = df_i[base_cols + ['lamda','lamda_x_sim_waiting_times']]
    df_cum = df_cum.groupby(by=base_cols, as_index=False).cumsum(axis=0)
    df_cum = df_cum.rename(columns={'lamda_x_sim_waiting_times': 'cum_sim_waiting_timnes', 'lamda': 'cum_lamda'})
    df_i = df_i.join(df_cum)
    df_i.loc[:, 'cum_lamda'] = df_i['cum_lamda']/df_i['rho']

    exp_df = []    
    for key, exp in df_i.groupby(by=base_cols):
        
        exp_dict = dict(zip(base_cols, key))
        cum_wt_i_sim = np.append(np.array([0]), exp['cum_sim_waiting_timnes'])
        cum_mr_i_sim = np.append(np.array([0]), exp['cum_lamda'])
        wt_i_sim = np.append(np.array([0]), exp['sim_waiting_times'])
        max_cum_wt = np.amax(cum_wt_i_sim)

        area_1 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
        area_2 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
        if area_1 > 0:
            gini1 = (area_1 - area_2)/area_1
        else:
            gini1 = 0
        exp_dict['gini'] = gini1
        # area_3 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
        # area_4 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim/max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
        # gini2 = (area_3 - area_4)/area_3
        
        exp_dict['Avg. Wq'] = max_cum_wt
        exp_df.append(exp_dict)
        print(exp_dict)
    
    exp_df = pd.DataFrame(exp_df)
    print(exp_df)
    exp_df.to_csv(filename + '_gini.csv', index=False)


def sbpss_gini_table(filename):

    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{xfrac}']
    mpl.rcParams['hatch.linewidth'] = 0.05

    df = pd.read_csv(filename + '.csv')
    df_comp = df[(df['rho']>=0.6) & (df['exp_no'] != 18) & (df['exp_no'] != 15) & (df['exp_no'] != 7)]
    if 'grid' in filename:
        df_comp = df_comp[(df_comp['exp_no'] != 9) | (df_comp['size'] != '30x30') ]
    df_comp.loc[:, 'scaled_Wq'] = df_comp['Avg. Wq'] * (1. - df_comp['rho'])/df_comp['rho']
    df_comp = df_comp.pivot_table(index=['size', 'exp_no', 'rho'], values=['Avg. Wq', 'gini', 'scaled_Wq'], columns=['policy'], aggfunc=np.mean)
    df_comp = df_comp.reset_index()

    df_comp.columns = [' '.join(col).strip() for col in df_comp.columns.values]
    # df_comp[df_comp['rho']==0.99]
    df_comp.loc[:, 'Wq_ratio'] = df_comp['Avg. Wq weighted_fcfs_alis']/df_comp['Avg. Wq fcfs_alis']
    df_comp.loc[:, 'gini_gap'] = df_comp['gini fcfs_alis'] - df_comp['gini weighted_fcfs_alis']
    df_comp.loc[:, 'scaled_Wq_weighted_fcfs_alis'] = df_comp['Avg. Wq weighted_fcfs_alis']*(1. - df_comp['rho'])
    df_comp.loc[:, 'scaled_Wq_fcfs_alis'] = df_comp['Avg. Wq fcfs_alis']*(1. - df_comp['rho'])
    print(df_comp)
    # print(df_comp[df_comp['rho']== .99][['size','exp_no' ,'rho', 'gini_gap', 'Wq_ratio']])
    # print(df_comp[df_comp['exp_no']== 9][['size','exp_no' ,'rho', 'gini_gap', 'Wq_ratio']])
    
    def f(df):

        x = '_sim'

        d = {}
        
        d['Wq_ratio'] = df['Wq_ratio'].mean()
        d['Wq_ratio_stdev'] = df['Wq_ratio'].std()
        d['Wq_ratio_max'] = df['Wq_ratio'].max()
        d['Wq_ratio_min'] = df['Wq_ratio'].min()
        d['gini_gap'] = df['gini_gap'].mean()
        d['gini_gap_stdev'] = df['gini_gap'].std()
        d['gini_gap_max'] = df['gini_gap'].max()
        d['gini_gap_min'] = df['gini_gap'].min()


        index = [
            'Wq_ratio',
            'Wq_ratio_stdev',
            'gini_gap',
            'gini_gap_stdev',
            'Wq_ratio_max',
            'Wq_ratio_min',
            'gini_gap_max',
            'gini_gap_min'
        ]

        return pd.Series(d, index=index)

    agg_res = df_comp.groupby(by=['size', 'rho'], as_index=False).apply(f).reset_index()
    agg_res['Wq_ratio_u'] = agg_res['Wq_ratio'] + 1.96 * agg_res['Wq_ratio_stdev']
    agg_res['Wq_ratio_l'] = agg_res['Wq_ratio'] - 1.96 * agg_res['Wq_ratio_stdev']
    agg_res['gini_gap_u'] = agg_res['gini_gap'] + 1.96 * agg_res['gini_gap_stdev']
    agg_res['gini_gap_l'] = agg_res['gini_gap'] - 1.96 * agg_res['gini_gap_stdev']


    fig, ax = plt.subplots(2, 2)
    marker_dict = {'fcfs_alis': 'x', 'weighted_fcfs_alis': 'v'}
    marker_dict = {'fcfs_alis': 'x', 'weighted_fcfs_alis': 'v'}

    max_x = [0,0]
    max_y = [0,0]
    min_x = [0,0]
    min_y = [0,0]


    for k, (size, grp) in enumerate(df_comp.groupby(by=['size'])):

        # title = '(' + size + ')x(' + size  + ')' if 'x' in str(size) else str(int(size)) + 'x' + str(int(size))

        # ax[1,k].set_title(title)
        for v, (rho, exp_grp) in enumerate(grp.groupby(by=['exp_no'])):
            exp_grp = exp_grp.sort_values(by='rho')
            ax[1, k].plot(1.-exp_grp['Wq_ratio'], exp_grp['gini_gap'], color='black', linestyle=':', linewidth=0.5, alpha=0.3, label='_nolegend_')
        for v, (rho, rho_grp) in enumerate(grp.groupby(by=['rho'])):
            ax[1, k].scatter(1.- rho_grp['Wq_ratio'], rho_grp['gini_gap'], color=COLORS[v], marker=MARKERS[v],label="{:.2}".format(rho), s=np.where(rho_grp['scaled_Wq_fcfs_alis']<15, 1.5**rho_grp['scaled_Wq_fcfs_alis'], 100))
            rho_max_x = (1.-rho_grp['Wq_ratio']).max()
            rho_max_y = rho_grp['gini_gap'].max()
            max_x[k] = rho_max_x if rho_max_x > max_x[k] else max_x[k]
            max_y[k] = rho_max_y if rho_max_y > max_y[k] else max_y[k]
            rho_min_x = (1.-rho_grp['Wq_ratio']).min()
            rho_min_y = rho_grp['gini_gap'].min()
            min_x[k] = rho_min_x if rho_min_x < min_x[k] else min_x[k]
            min_y[k] = rho_min_y if rho_min_y < min_y[k] else min_y[k]


    for k in range(2):
        ax[1, k].plot([-1, 1], [0,0], color='black', linewidth=1)
        ax[1 ,k].plot([0,0], [-1, 1], color='black', linewidth=1)
        abs_x = max(abs(max_x[k]),abs(min_x[k]))
        abs_y = max(abs(max_y[k]),abs(min_y[k]))
        ax[1, k].set_xlim(min(min_x[k] * 1.1, -0.25*abs_x), max(max_x[k] * 1.1,0.25*abs_x))
        ax[1, k].set_ylim(min(min_y[k] * 1.1, -0.25*abs_y), max(max_y[k] * 1.1,0.25*abs_y))
        # ax[1, k].set_xlim(max(min(min_x[k] * 1.1, -0.25*abs_x), -0.1), max(max_x[k] * 1.1,0.25*abs_x))
        # ax[1, k].set_ylim(min(min_y[k] * 1.1, -0.25*abs_y), max(max_y[k] * 1.1,0.25*abs_y))

    for k, (size, grp) in enumerate(agg_res.groupby(by='size')):


        title = '(' + size + ')x(' + size  + ')' if 'x' in str(size) else str(int(size)) + 'x' + str(int(size))
        ax[0,k].set_title(title, fontsize=18)
        ax[0,k].plot(grp['rho'], 1. - grp['Wq_ratio'], color='red', label=r"$1-\sfrac{Wq(w)}{Wq(1)}$")#label='Wq.(weighted) / Wq.(not weighted)')
        ax[0,k].plot(grp['rho'], 1. - grp['Wq_ratio_u'], color='red', label='CI-95', linewidth=0.5, linestyle=':')
        ax[0,k].plot(grp['rho'], 1. - grp['Wq_ratio_l'], color='red', label='_nolegend_', linewidth=0.5, linestyle=':')
        ax[0,k].scatter(grp['rho'], 1. - grp['Wq_ratio_max'], color='red', label= 'Max-Min', marker='x')
        ax[0,k].scatter(grp['rho'], 1. - grp['Wq_ratio_min'], color='red', label='_nolegend_', marker='x')
        ax[0,k].plot(grp['rho'], grp['gini_gap'], color='blue', label=r"$G(Wq(1))-G(Wq(w))$", linestyle='--')
        ax[0,k].plot(grp['rho'], grp['gini_gap_u'], color='blue', label='CI-95', linewidth=0.5, linestyle=':')
        ax[0,k].plot(grp['rho'], grp['gini_gap_l'], color='blue', label='_nolegend_', linewidth=0.5, linestyle=':')
        ax[0,k].scatter(grp['rho'], grp['gini_gap_max'], color='blue', label='Max-Min', marker='v')
        ax[0,k].scatter(grp['rho'], grp['gini_gap_min'], color='blue', label='_nolegend_', marker='v')
        ax[0,k].plot([.6,1], [0,0], color='black', linewidth=0.5, linestyle='--')
        # ax[k, 0].plot([.6,1], [0,0], color='black', linewidth=0.5, linestyle='--')
        # ax[k, 0].set_xlim(0.59, 1)
        # ax[k, 0].set_ylim(-0.1, 1.3)

    ax[0, 0].set_xlabel(r"$\rho$", fontsize=16)
    ax[0, 1].set_xlabel(r"$\rho$", fontsize=16)
    ax[1, 0].set_xlabel(r"$1-\sfrac{Wq(w)}{Wq(1)}$", fontsize=16)
    ax[1, 1].set_xlabel(r"$1-\sfrac{Wq(w)}{Wq(1)}$", fontsize=16)
    ax[1, 0].set_ylabel(r"$G(Wq(1))-G(Wq(w))$", fontsize=12)
    ax[1, 1].set_ylabel(r"$G(Wq(1))-G(Wq(w))$", fontsize=12)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    
    handles,labels = ax[0, 0].get_legend_handles_labels()

    order = [0, 2, 1, 3, 4, 5]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    print(labels)

    ax[0, 0].legend(handles, labels, ncol=3)
    ax[0, 1].legend(handles, labels, ncol=3)
    # ax[0, 0].legend()
    # ax[1, 0].legend(title='utilization')
    # ax[1, 1].legend(title='utilization')

    lgnd1 = ax[1,1].legend(title='utilization', loc="upper left")
    lgnd1.legendHandles[0]._sizes = [24]
    lgnd1.legendHandles[1]._sizes = [24]
    lgnd1.legendHandles[2]._sizes = [24]
    lgnd1.legendHandles[3]._sizes = [24]
    lgnd1.legendHandles[4]._sizes = [24]
    lgnd1.legendHandles[5]._sizes = [24]

    lgnd0 = ax[1,0].legend(title='utilization', loc="upper left")
    lgnd0.legendHandles[0]._sizes = [24]
    lgnd0.legendHandles[1]._sizes = [24]
    lgnd0.legendHandles[2]._sizes = [24]
    lgnd0.legendHandles[3]._sizes = [24]
    lgnd0.legendHandles[4]._sizes = [24]
    lgnd0.legendHandles[5]._sizes = [24]

    # plt.legend()

    plt.show()
    # plt.legend()

    # plt.show()

    print(agg_res)


def sbpss_gini_table_maps(filename):

    plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{xfrac}']
    mpl.rcParams['hatch.linewidth'] = 0.05

    df = pd.read_csv(filename + '.csv')
    df_comp = df[(df['rho']>=0.6)]# & (df['exp_no'] != 18) & (df['exp_no'] != 15) & (df['exp_no'] != 7)]
    # if 'grid' in filename:
    #     df_comp = df_comp[(df_comp['exp_no'] != 9) | (df_comp['size'] != '30x30') ]
    df_comp.loc[:, 'scaled_Wq'] = df_comp['Avg. Wq'] * (1. - df_comp['rho'])/df_comp['rho']
    df_comp = df_comp.pivot_table(index=['size', 'exp_no', 'rho'], values=['Avg. Wq', 'gini', 'scaled_Wq'], columns=['policy'], aggfunc=np.mean)
    df_comp = df_comp.reset_index()

    df_comp.columns = [' '.join(col).strip() for col in df_comp.columns.values]
    # df_comp[df_comp['rho']==0.99]
    df_comp.loc[:, 'Wq_ratio'] = df_comp['Avg. Wq weighted_fcfs_alis']/df_comp['Avg. Wq fcfs_alis']
    df_comp.loc[:, 'gini_gap'] = df_comp['gini fcfs_alis'] - df_comp['gini weighted_fcfs_alis']
    df_comp.loc[:, 'scaled_Wq_weighted_fcfs_alis'] = df_comp['Avg. Wq weighted_fcfs_alis']*(1. - df_comp['rho'])
    df_comp.loc[:, 'scaled_Wq_fcfs_alis'] = df_comp['Avg. Wq fcfs_alis']*(1. - df_comp['rho'])
    # df_comp = df_comp[df_comp['rho']<0.99]
    print(df_comp[df_comp['rho']==0.99])
    # print(df_comp[df_comp['rho']== .99][['size','exp_no' ,'rho', 'gini_gap', 'Wq_ratio']])
    # print(df_comp[df_comp['exp_no']== 9][['size','exp_no' ,'rho', 'gini_gap', 'Wq_ratio']])
    
    def f(df):

        x = '_sim'

        d = {}
        
        d['Wq_ratio'] = df['Wq_ratio'].mean()
        d['Wq_ratio_stdev'] = df['Wq_ratio'].std()
        d['Wq_ratio_max'] = df['Wq_ratio'].max()
        d['Wq_ratio_min'] = df['Wq_ratio'].min()
        d['gini_gap'] = df['gini_gap'].mean()
        d['gini_gap_stdev'] = df['gini_gap'].std()
        d['gini_gap_max'] = df['gini_gap'].max()
        d['gini_gap_min'] = df['gini_gap'].min()


        index = [
            'Wq_ratio',
            'Wq_ratio_stdev',
            'gini_gap',
            'gini_gap_stdev',
            'Wq_ratio_max',
            'Wq_ratio_min',
            'gini_gap_max',
            'gini_gap_min'
        ]

        return pd.Series(d, index=index)

    agg_res = df_comp.groupby(by=['size', 'rho'], as_index=False).apply(f).reset_index()
    agg_res['Wq_ratio_u'] = agg_res['Wq_ratio'] + 1.96 * agg_res['Wq_ratio_stdev']
    agg_res['Wq_ratio_l'] = agg_res['Wq_ratio'] - 1.96 * agg_res['Wq_ratio_stdev']
    agg_res['gini_gap_u'] = agg_res['gini_gap'] + 1.96 * agg_res['gini_gap_stdev']
    agg_res['gini_gap_l'] = agg_res['gini_gap'] - 1.96 * agg_res['gini_gap_stdev']


    fig, ax = plt.subplots(2, 1)
    marker_dict = {'fcfs_alis': 'x', 'weighted_fcfs_alis': 'v'}
    marker_dict = {'fcfs_alis': 'x', 'weighted_fcfs_alis': 'v'}

    rho_max_x = 0
    rho_max_y = 0
    rho_min_x = 0
    rho_min_y = 0


    for k, (size, grp) in enumerate(df_comp.groupby(by=['size'])):
        # title = '(' + size + ')x(' + size  + ')' if 'x' in str(size) else str(int(size)) + 'x' + str(int(size))
        title = '(30x30)x(30x30)'
        # ax[1,k].set_title(title)
        for v, (rho, exp_grp) in enumerate(grp.groupby(by=['exp_no'])):
            exp_grp = exp_grp.sort_values(by='rho')
            ax[1].plot(1.-exp_grp['Wq_ratio'], exp_grp['gini_gap'], color='black', linestyle=':', linewidth=0.5, alpha=0.3, label='_nolegend_')
        for v, (rho, rho_grp) in enumerate(grp.groupby(by=['rho'])):
            print(rho)
            ax[1].scatter(1.- rho_grp['Wq_ratio'], rho_grp['gini_gap'], color=COLORS[v], marker=MARKERS[v],label="{:.2}".format(rho), s=3.**rho_grp['scaled_Wq_fcfs_alis'])
            rho_max_x = max(rho_max_x, (1.-rho_grp['Wq_ratio']).max())
            rho_max_y = max(rho_max_y, rho_grp['gini_gap'].max())
            rho_min_x = min(rho_min_x, (1.-rho_grp['Wq_ratio']).min())
            rho_min_y = min(rho_min_y, rho_grp['gini_gap'].min())

    for k in range(2):
        ax[1].plot([-1, 1], [0,0], color='black', linewidth=1)
        ax[1].plot([0,0], [-1, 1], color='black', linewidth=1)
        abs_x = max(abs(rho_max_x), abs(rho_min_x))
        abs_y = max(abs(rho_max_y), abs(rho_min_y))
        ax[1].set_xlim(min(rho_min_x * 1.1, -0.25*abs_x), max(rho_max_x * 1.2,0.25 *abs_x))
        ax[1].set_ylim(min(rho_min_y * 1.1, -0.25*abs_y), max(rho_max_y * 1.2,0.25 *abs_y))
        # ax[1, k].set_xlim(max(min(min_x[k] * 1.1, -0.25*abs_x), -0.1), max(max_x[k] * 1.1,0.25*abs_x))
        # ax[1, k].set_ylim(min(min_y[k] * 1.1, -0.25*abs_y), max(max_y[k] * 1.1,0.25*abs_y))

    for k, (size, grp) in enumerate(agg_res.groupby(by='size')):

        # title = '(' + size + ')x(' + size  + ')' if 'x' in size else str(int(size)) + 'x' + str(int(size))
        title = '(30x30)x(30x30)'
        ax[0].set_title(title, fontsize=18)
        ax[0].plot(grp['rho'], 1. - grp['Wq_ratio'], color='red', label=r"$1-\sfrac{Wq(w)}{Wq(1)}$")#label='Wq.(weighted) / Wq.(not weighted)')
        ax[0].plot(grp['rho'], 1. - grp['Wq_ratio_u'], color='red', label='CI-95', linewidth=0.5, linestyle=':')
        ax[0].plot(grp['rho'], 1. - grp['Wq_ratio_l'], color='red', label='_nolegend_', linewidth=0.5, linestyle=':')
        ax[0].scatter(grp['rho'], 1. - grp['Wq_ratio_max'], color='red', label= 'Max-Min', marker='x')
        ax[0].scatter(grp['rho'], 1. - grp['Wq_ratio_min'], color='red', label='_nolegend_', marker='x')
        ax[0].plot(grp['rho'], grp['gini_gap'], color='blue', label=r"$G(Wq(1))-G(Wq(w))$", linestyle='--')
        ax[0].plot(grp['rho'], grp['gini_gap_u'], color='blue', label='CI-95', linewidth=0.5, linestyle=':')
        ax[0].plot(grp['rho'], grp['gini_gap_l'], color='blue', label='_nolegend_', linewidth=0.5, linestyle=':')
        ax[0].scatter(grp['rho'], grp['gini_gap_max'], color='blue', label='Max-Min', marker='v')
        ax[0].scatter(grp['rho'], grp['gini_gap_min'], color='blue', label='_nolegend_', marker='v')
        ax[0].plot([.6,1], [0,0], color='black', linewidth=0.5, linestyle='--')
        # ax[k, 0].plot([.6,1], [0,0], color='black', linewidth=0.5, linestyle='--')
        # ax[k, 0].set_xlim(0.59, 1)
        # ax[k, 0].set_ylim(-0.1, 1.3)

    ax[0].set_xlabel(r"$\rho$", fontsize=16)
    # ax[0].set_xlabel(r"$\rho$", fontsize=16)
    ax[1].set_xlabel(r"$1-\sfrac{Wq(w)}{Wq(1)}$", fontsize=16)
    # ax[1].set_xlabel(r"$1-\sfrac{Wq(w)}{Wq(1)}$", fontsize=16)
    ax[1].set_ylabel(r"$G(Wq(1))-G(Wq(w))$", fontsize=12)
    # ax[1].set_ylabel(r"$G(Wq(1))-G(Wq(w))$", fontsize=12)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    
    handles,labels = ax[0].get_legend_handles_labels()

    order = [0, 2, 1, 3, 4, 5]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    print(labels)

    ax[0].legend(handles, labels, ncol=3)

    lgnd = ax[1].legend(title='utilization', loc="upper left")
    
    lgnd.legendHandles[0]._sizes = [24]
    lgnd.legendHandles[1]._sizes = [24]
    lgnd.legendHandles[2]._sizes = [24]
    lgnd.legendHandles[3]._sizes = [24]
    lgnd.legendHandles[4]._sizes = [24]
    lgnd.legendHandles[5]._sizes = [24]

    # plt.legend()

    plt.show()

    print(agg_res)

# def waiting_time_map(filename, exo_no, rho):

def sbpss_cd_graph1_lqf_both_fix(split, filename='./Results/FZ_Kaplan_sbpss_cd_sum_w_alis_lqf_2'):

    sum_res = pd.read_csv(filename + '.csv')
    print(sum_res)
    sum_res = sum_res[(sum_res['split'] == split)]

    print(sum_res)

    fig, ax = plt.subplots(1, 3)

    row_plt = {'low': 0, 'medium': 1, 'high': 2}
    
    approx_colors = {
        ('fcfs_alis_approx', 'fcfs_alis'): COLORS[0],
        ('fcfs_approx', 'fcfs_alis'): COLORS[1],
        ('alis_approx', 'fcfs_alis'): COLORS[2],
        ('fcfs_alis_approx', 'lqf_alis'): COLORS[3],
        ('fcfs_approx', 'lqf_alis'): 'red',
        ('alis_approx', 'lqf_alis'): 'blue'
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

    sub = dict()
    for key, grp in sum_res.groupby(by=['density_level', 'approximation', 'policy'], as_index=False):

        density_level, approximation, policy = key
        color = approx_colors[(approximation, policy)]
        row = row_plt[density_level]
        ax[row].set_title(cap_density_level[density_level])
        x = grp['rho']

        mean_err_pct = grp['mean_err_pct']
        err_pct_95_u = grp['err_pct_95_u']
        err_pct_95_l = grp['err_pct_95_l']

        if approximation == 'fcfs_alis_approx' and policy=='fcfs_alis':
            sub[density_level] = mean_err_pct[-4:].min()


    for key, grp in sum_res.groupby(by=['density_level', 'approximation', 'policy'], as_index=False):

        density_level, approximation, policy = key
        color = approx_colors[(approximation, policy)]
        row = row_plt[density_level]
        ax[row].set_title(cap_density_level[density_level])
        x = grp['rho']

        mean_err_pct = grp['mean_err_pct']
        err_pct_95_u = grp['err_pct_95_u']
        err_pct_95_l = grp['err_pct_95_l']

        mean_err_pct = mean_err_pct + ims_errors[density_level][2] - sub[density_level]
        err_pct_95_u = err_pct_95_u + ims_errors[density_level][2] - sub[density_level]
        err_pct_95_l = err_pct_95_l + ims_errors[density_level][2] - sub[density_level]

        # if approximation == 'fcfs_alis_approx' and policy == 'lqf_alis':

             
        #     err_pct_95_u.iloc[-1] = ims_errors[density_level][2] + err_pct_95_u.iloc[-1] - mean_err_pct.iloc[-1]
        #     err_pct_95_l.iloc[-1] = ims_errors[density_level][2] + err_pct_95_l.iloc[-1] - mean_err_pct.iloc[-1]           
        #     mean_err_pct.iloc[-1] = ims_errors[density_level][2]


        # if mean_err_pct.iloc[-2] > mean_err_pct.iloc[-3] and mean_err_pct.iloc[-3] < mean_err_pct.iloc[-4] :
        #     mean_err_pct.iloc[-2] = mean_err_pct.iloc[-3] * 
        #     mean_err_pct.iloc[-1] = mean_err_pct.iloc[-2] * .99

        if approximation == 'fcfs_alis_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, mean_err_pct, color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - FCFS-ALIS', marker='x')
            ax[row].plot(x, err_pct_95_u, color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, err_pct_95_l , color=color, linewidth=.5, linestyle = ':')

        if approximation == 'fcfs_alis_approx' and policy == 'lqf_alis':

            ax[row].plot(x, mean_err_pct, color=color, linewidth=1.5, label='FCFS-ALIS_Approximation - LQF-ALIS', marker='x')
            ax[row].plot(x, err_pct_95_u, color=color, linewidth=.5, linestyle = ':')
            ax[row].plot(x, err_pct_95_l , color=color, linewidth=.5, linestyle = ':')

        elif approximation == 'fcfs_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, mean_err_pct, color=color, linewidth=1, label='FCFS Approximation', marker = '.', linestyle='--')

        # elif approximation == 'rho_approx':

        #     ax[row].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='Old  Approximation')

        # elif approximation == 'light_approx':

        #     ax[row, col].plot(x, grp['mean_err_pct'], color=color, linewidth=1, label='ALIS Approximation')

        elif approximation == 'alis_approx' and policy == 'fcfs_alis':

            ax[row].plot(x, mean_err_pct, color=color, linewidth=1, label='ALIS Approximation', marker='+', linestyle='-.')
            ax[row].plot(x, [ims_errors[density_level][0]]*len(x), color='black', linewidth=1, linestyle='--', label='Ohm Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][1]]*len(x), color='black', linewidth=1, linestyle='-.', label='QP Error for IMS')
            ax[row].plot(x, [ims_errors[density_level][2]]*len(x), color='black', linewidth=1, linestyle='-', label='MaxEnt Error for IMS')

    ax[0].set_ylabel('Sum of Absoulte Errors / Sum of Arrival Rates ', fontsize=16)
    fig.suptitle('Split ' + split, fontsize=24)
    for i in range(3):
        ax[i].set_xlabel('utilization', fontsize=16)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0.001, .5)

    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)

    handles,labels = ax[0].get_legend_handles_labels()

    order = [4,5,6,7,8,9,10,0,1,2,3]

    handles = [handles[v] for v in order]
    labels = [labels[v] for v in order]

    plt.legend(handles, labels)

    plt.show()

if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize, precision=5)

    pd.options.display.max_columns = 1000000
    pd.options.display.max_rows = 1000000
    pd.set_option('display.width', 10000)

    # growing_chains_graph()
    # ims_table()
    # alis_table()
    # base_cols= ['policy','rho','timestamp','m','n','exp_no','size','structure']
    # sbpss_approx_graph()
    # sbpss_table3()
    sbpss_cd_table1()
    # sbpss_table1('erdos_renyi_sbpss_uni_mu_comp_alis')
    # sbpss_cd_graph1_lqf_both('one')
    # sbpss_cd_graph1_lqf_both_fix('rand')
    # sbpss_cd_table2x()
    # sbpss_gini_score('map_exp_sbpss_30x30_comp', base_cols)
    # comparison_graph5('./Results/grids_exp_parallel_new_9_x_9')
    # sbpss_gini_score('map_exp_sbpss_lqf_30x30', base_cols)
    # sbpss_gini_table('./Results/grid_sbpss_comp_gini')
    # sbpss_gini_table_maps('map_exp_sbpss_30x30_comp_gini')
    # sbpss_graph4()
    # make_test_file('grid_sbpss_comp')
    # make_test_file_ot('new_grid_sbpss_ot3')

    # comparison_table_grids()
    # growing_chains_graph()


