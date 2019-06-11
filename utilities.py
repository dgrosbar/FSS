import numpy as np
from math import floor
from scipy import sparse as sps
import os

def printcols(df):
	
    for col in df.columns.values:
    	print(col)


def printarr(arr, arr_name=None):

    if arr_name is None:
        print('-'*75)
    else:
        x = floor((75-len(arr_name)-2)/2)
        print('-'*x + ' ' + arr_name + ' ' + '_'*(x+1))

    print(np.array2string(arr, max_line_width=np.inf))
    
    print('-'*75)

def fast_choice(arr, size, shuffle=False):

    idx = set()
    while len(idx) < size:
        idx.add(np.random.randint(0, len(arr)))
    idx = np.array(list(idx))
    if shuffle:
        np.random.shuffle(idx)
    return [arr[x] for x in idx]

def write_df_to_file(filename, df):

    if os.path.exists(filename + '.csv'):
        with open(filename + '.csv', 'a') as file:
            df.to_csv(file, header=False, index=False)
    else:
        with open(filename + '.csv', 'w') as file:
            df.to_csv(file, index=False)
            
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
