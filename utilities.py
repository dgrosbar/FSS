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

def log_res_to_df(compatability_matrix, alpha=None, beta=None, lamda=None, s = None, mu=None, result_dict=None, timestamp=None, aux_data=None):

    if timestamp is None:
        timestamp = dt.datetime.now() 
    m, n = compatability_matrix.shape
    nnz = compatability_matrix.nonzero()

    def prep_data_for_df(data, data_struc):

        if data_struc == 'mat':
            return data[nnz]
        if data_struc == 'row':
            return data[nnz[0]]
        if data_struc == 'col':
            return data[nnz[1]]
        if data_struc == 'aux':
            return [data] * len(nnz[0])

    edge_count = len(nnz[0])

    input_dict = {
        'i': nnz[0],
        'j': nnz[1],
    }
    for data_name , data in zip(['alpha', 'lamda', 's'], [alpha, lamda, s]):
        if data is not None:
            try:
                input_dict[data_name] = data[nnz[0]]
            except:
                print(data_name, data)
                raise Exception

    for data_name , data in zip(['beta', 'mu'], [beta, mu]):
        if data is not None:
            input_dict[data_name] = data[nnz[1]]

    result_dict = dict((col, prep_data_for_df(data, data_struc)) for data_struc, data_struc_dict in result_dict.items() for col, data in data_struc_dict.items())
    
    res_df = pd.DataFrame.from_dict({**input_dict, **result_dict})

    res_df.loc[:, 'timestamp'] = dt.datetime.now() 
    res_df.loc[:, 'max_edges'] = m * n
    res_df.loc[:, 'edge_count'] = edge_count
    res_df.loc[:, 'edge_density'] = edge_count/ (m*n)
    res_df.loc[:, 'm'] = m
    res_df.loc[:, 'n'] = n
    
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

