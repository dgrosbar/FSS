import numpy as np
import pandas as pd
from math import floor
from scipy import sparse as sps
import os
import datetime as dt

def printcols(df):
	
    for col in df.columns.values:
    	print(col)

def printarr(arr, arr_name=None):

    if arr_name is None:
        print('-'*75)
    else:
        x = floor((75-len(arr_name)-2)/2)
        print('-'*x + ' ' + arr_name + ' ' + '_'*(x+1))

    print(np.array2string(arr, max_line_width=np.inf, formatter={'float_kind':lambda x: "%.6f" % x}))
    
    print('-'*75)

def fast_choice(arr, size, shuffle=False):

    idx = set()
    while len(idx) < size:
        idx.add(np.random.randint(0, len(arr)))
    idx = np.array(list(idx))
    if shuffle:
        np.random.shuffle(idx)
    return [arr[x] for x in idx]

def sp_unique(sp_matrix, axis=0):
    ''' Returns a sparse matrix with the unique rows (axis=0)
    or columns (axis=1) of an input sparse matrix sp_matrix'''
    if axis == 1:
        sp_matrix = sp_matrix.T

    old_format = sp_matrix.getformat()
    dt = np.dtype(sp_matrix)
    ncols = sp_matrix.shape[1]

    if old_format != 'lil':
        sp_matrix = sp_matrix.tolil()

    _, ind = np.unique(sp_matrix.data + sp_matrix.rows, return_index=True)
    rows = sp_matrix.rows[ind]
    data = sp_matrix.data[ind]
    nrows_uniq = data.shape[0]

    sp_matrix = sps.lil_matrix((nrows_uniq, ncols), dtype=dt)  #  or sp_matrix.resize(nrows_uniq, ncols)
    sp_matrix.data = data
    sp_matrix.rows = rows

    ret = sp_matrix.asformat(old_format)
    if axis == 1:
        ret = ret.T        
    return ret

def lexsort_row(A):
    ''' numpy lexsort of the rows, not used in sp_unique'''
    return A[np.lexsort(A.T[::-1])]

def write_df_to_file(filename, df):

    if os.path.exists(filename + '.csv'):
        with open(filename + '.csv', 'a') as file:
            df.to_csv(file, header=False, index=False)
    else:
        with open(filename + '.csv', 'w') as file:
            df.to_csv(file, index=False)

def log_res_to_df(compatability_matrix, row_sums, col_sums, result_dict, alpha_beta=True, aux_data=None, timestamp=None):

    if timestamp is None:
        timestamp = dt.datetime.now() 
    m, n = compatability_matrix.shape
    nnz = compatability_matrix.nonzero()
    nnz_dict = {'row': nnz[0], 'col': nnz[1], 'mat': nnz}
    edge_count = len(nnz[0])

    if alpha_beta:
        row_sum_name = 'alpha'
        col_sum_name = 'beta'
    else:
        row_sum_name = 'lamda'
        col_sum_name = 'mu'

    cols = ['timestamp', 'm', 'n', 'max_edges', 'edge_count', 'edge_density', 'utilization','i', 'j'] + [row_sum_name, col_sum_name] + [key for key in result_dict]

    input_dict = {
        'i': nnz[0],
        'j': nnz[1],
        row_sum_name: row_sums[nnz[0]],
        col_sum_name: col_sums[nnz[1]],
    }

    result_dict = dict((key, data[nnz_dict[nnz_mask]]) for key, (data, nnz_mask) in result_dict.items())
    
    res_df = pd.DataFrame.from_dict({**input_dict, **result_dict})


    res_df.loc[:, 'timestamp'] = dt.datetime.now() 
    res_df.loc[:, 'max_edges'] = m * n
    res_df.loc[:, 'edge_count'] = edge_count
    res_df.loc[:, 'edge_density'] = len(nnz[0])/ (m*n)
    res_df.loc[:, 'm'] = m
    res_df.loc[:, 'n'] = n
    res_df.loc[:, 'utilization'] = row_sums.sum()/col_sums.sum()
    if aux_data is not None:
        for key, val in aux_data.items():
            res_df.loc[:, key] = val

    return res_df


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

