import numpy as np
import pandas as pd
from typing import Callable, Iterable
from scipy.stats import pearsonr, linregress
from sklearn.metrics import r2_score


# Define color highlight criteria for correlation spreadsheet cells
def null_hl(x):
    color = (-0.6 < x) & (x < 0.6)
    return ['background-color: darkgrey' if y else '' for y in color]

def six_to_sev_hl(x):
    color = ((-0.7 < x) & (x <= -0.6)) | ((0.6 <= x) & (x < 0.7))
    return ['background-color: gold' if y else '' for y in color]

def sev_to_eig_hl(x):
    color = ((-0.8 < x) & (x <= -0.7)) | ((0.7 <= x) & (x < 0.8))
    return ['background-color: peru' if y else '' for y in color]

def eig_to_nin_hl(x):
    color = ((-0.9 < x) & (x <= -0.8)) | ((0.8 <= x) & (x < 0.9))
    return ['background-color: deepskyblue' if y else '' for y in color]

def best_hl(x):
    color = ((-1 <= x) & (x <= -0.9)) | ((0.9 <= x) & (x <= 1))
    return ['background-color: dodgerblue' if y else '' for y in color]

# Encompassing function that performs all the colorings at once
def corr_highlight(correlations: pd.DataFrame) -> object:
    corr_hl = correlations.style.apply(null_hl)
    corr_hl = corr_hl.apply(six_to_sev_hl)
    corr_hl = corr_hl.apply(sev_to_eig_hl)
    corr_hl = corr_hl.apply(eig_to_nin_hl)
    corr_hl = corr_hl.apply(best_hl)
    
    return corr_hl

# Define color highlight criteria for hypothesis tests
def sig_hl(x):
    color = (x == 1)
    return ['background-color: dodgerblue' if y else '' for y in color]

def notsig_hl(x):
    color = (x == 0)
    return ['background-color: darkgrey' if y else '' for y in color]

# Colors all at once
def pval_highlight(pvals: pd.DataFrame) -> object:
    pval_hl = pvals.style.apply(sig_hl)
    pval_hl = pval_hl.apply(notsig_hl)
    
    return pval_hl

'''TODO
change to account for no intermediate values
'''
def grab_column_split() -> tuple[int]:
    '''
    For each prompt, enter a lower/uppercase single letter representing the column.
    Assumes columns only go from A-Z. Will refactor if column needs are greater.
    '''
    
    inp = ord(input('End of input columns? (letter): ').upper()) - 64
    inter = ord(input('End of intermediate columns? (letter): ').upper()) - 64
    
    return inp, inter

# Removes invalid values in preparation for computing correlations
def clean(data):
    data.replace([np.inf, -np.inf], np.nan, inplace = True)
    data.dropna(inplace = True)
    
# Computes transformation matrix
def populate(data, func, format_func, idx, args, comm):

    if args == 1:
        matrix = data[data.columns[idx[0] : idx[1]]].copy()
        matrix = func(matrix)

        new_cols = []
        new_idx = idx[1] - idx[0]

        for i in range(new_idx):
            col_name = matrix.columns[i]
            new_cols.append(format_func(col_name))

        matrix.columns = new_cols

    elif args == 2:
        matrix = pd.DataFrame()

        for i in range(idx[0], idx[1]):
            col1_name = data.columns[i]
            col1 = data[col1_name].to_numpy()

            for j in range(i, idx[1]):
                col2_name = data.columns[j]
                col2 = data[col2_name].to_numpy()
                cond = (comm == 0) and (col1_name != col2_name)

                key = format_func(col1_name, col2_name)
                compute = pd.Series(func(col1, col2), name = key)

                if cond:
                    comm_key = format_func(col2_name, col1_name)
                    comm_comp = pd.Series(func(col2, col1), name = comm_key)

                if matrix.empty:
                    matrix = pd.DataFrame(compute)

                    if cond:
                        matrix = matrix.join(comm_comp).copy()
                    continue

                matrix = matrix.join(compute).copy()
                if cond:
                    matrix = matrix.join(comm_comp).copy()


    return matrix

def R2(X, Y):
    slope, intercept, _, _, _ = linregress(X, Y)
    pred = X * slope + intercept
    coeff_of_det = r2_score(Y, pred)
    return coeff_of_det

def functional_corr(data: pd.DataFrame, column_split: Iterable, func: Callable, format_func: Callable, args: int, comm: int = 1) -> dict:
    '''
    Given data with associated features, transform the data using func and format the new labels usng format_func.
    args provides the number of arguments for func (1 for inplace operations and 2 for double operand operations).
    comm states whether an operation is commutative (1 for yes, 0 for no). maybe switch to bool later.
    '''
    
    pvalue = lambda x,y: 1 if (pearsonr(x, y)[1] < 0.05) else 0
    
    inp_idx, inter_idx = column_split

    # Compute input transformations
    input_w_inter = populate(data, func, format_func, (0, inp_idx), args, comm)
    clean(input_w_inter)
    saved_1 = input_w_inter.copy()

    # Input-with-Intermediate Correlations
    columns = data.columns[inp_idx:inter_idx]
    input_w_inter[columns] = data[columns].copy()
    input_w_inter_f = input_w_inter.corr()
    sig_1 = input_w_inter.corr(method = pvalue)
    cod_1 = input_w_inter_f ** 2

    input_w_inter_f = input_w_inter_f[columns].iloc[:-len(columns)].copy()
    sig_1 = sig_1[columns].iloc[:-len(columns)].copy()
    cod_1 = cod_1[columns].iloc[:-len(columns)].copy()

    # Compute intermediate transformations
    inter_w_final = populate(data, func, format_func, (inp_idx, inter_idx), args, comm)
    clean(inter_w_final)
    saved_2 = inter_w_final.copy()

    # Intermediate-with-Final Correlations
    columns = data.columns[inter_idx:]
    inter_w_final[columns] = data[columns].copy()
    inter_w_final_f = inter_w_final.corr()
    sig_2 = inter_w_final.corr(method = pvalue)
    cod_2 = inter_w_final_f ** 2

    inter_w_final_f = inter_w_final_f[columns].iloc[:-len(columns)].copy()
    sig_2 = sig_2[columns].iloc[:-len(columns)].copy()
    cod_2 = cod_2[columns].iloc[:-len(columns)].copy()

    # Compute all transformations
    final = populate(data, func, format_func, (0, inter_idx), args, comm)
    clean(final)
    saved_3 = final.copy()

    # (Input + Intermediate)-with-Final Correlations
    final[columns] = data[columns].copy()
    final_f = final.corr()
    sig_3 = final.corr(method = pvalue)
    cod_3 = final_f ** 2

    final_f = final_f[columns].iloc[:-len(columns)].copy()
    sig_3 = sig_3[columns].iloc[:-len(columns)].copy()
    cod_3 = cod_3[columns].iloc[:-len(columns)].copy()
    
    # Store computations
    corrs = [input_w_inter_f, inter_w_final_f, final_f]
    pvals = [sig_1, sig_2, sig_3]
    cods = [cod_1, cod_2, cod_3]
    intermediate_steps = [saved_1, saved_2, saved_3]
    
    # Round some computations to 3 sigfigs
    for i in range(3):
        corrs[i] = np.round(corrs[i], 3)
        cods[i] = np.round(cods[i], 3)

    # Drops rows with NaN (divide by zero errors and similar issues)
    for i, df in enumerate(corrs):
        if df.isnull().values.any():
            corrs[i] = df.dropna(axis = 0)

    return {'correlations' : corrs, 'computations' : intermediate_steps, 'significance': pvals, 'coeff of determination': cods}

def t_corr_helper(corr_columns, data, func, format_func, idxs, args, comm):

    pvalue = lambda x,y: 1 if (pearsonr(x, y)[1] < 0.05) else 0

    t_data = populate(data, func, format_func, idxs, args, comm)
    clean(t_data)
    
    t_data[corr_columns] = data[corr_columns].copy()
    t_w_ut_corr = np.round(t_data.corr(), 3)
    pvals = t_data.corr(method = pvalue)
    cods = np.round(t_w_ut_corr ** 2, 3)

    t_w_ut_corr = t_w_ut_corr[corr_columns].iloc[:-len(corr_columns)].copy()
    pvals = pvals[corr_columns].iloc[:-len(corr_columns)].copy()
    cods = cods[corr_columns].iloc[:-len(corr_columns)].copy()

    if t_w_ut_corr.isnull().values.any():
        t_w_ut_corr.dropna(axis = 0, inplace = True)

    return [t_w_ut_corr, pvals, cods]

def t_corr(data: pd.DataFrame, column_split: Iterable, func: Callable, format_func: Callable, args: int, comm: int = 1) -> dict:
    inp_idx, inter_idx = column_split

    corr_columns = data.columns[inp_idx:inter_idx]
    input_w_inter = t_corr_helper(corr_columns, data, func, format_func, (0, inp_idx), args, comm)

    corr_columns = data.columns[inter_idx:]
    inter_w_final = t_corr_helper(corr_columns, data, func, format_func, (inp_idx, inter_idx), args, comm)

    final = t_corr_helper(corr_columns, data, func, format_func, (0, inter_idx), args, comm)

    return input_w_inter, inter_w_final, final
