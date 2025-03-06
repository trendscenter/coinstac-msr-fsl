#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:00:16 2018

@author: Harshvardhan
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm


def gather_local_stats(X, y):

    y_labels = list(y.columns)
    biased_X = sm.add_constant(X)
    meanY_vector, lenY_vector = [], []
    X_labels = list(biased_X.columns)

    local_params = []
    local_sse = []
    local_pvalues = []
    local_tvalues = []
    local_rsquared = []

    for column in y.columns:
        curr_y = list(y[column])
        meanY_vector.append(np.mean(curr_y))
        lenY_vector.append(len(y))

        # Printing local stats as well
        model = sm.OLS(curr_y, biased_X.astype(float)).fit()
        local_params.append(model.params)
        local_sse.append(model.ssr)
        local_pvalues.append(model.pvalues)
        local_tvalues.append(model.tvalues)
        local_rsquared.append(model.rsquared_adj)

    keys = ["beta", "sse", "pval", "tval", "rsquared", "covariate_labels"]
    local_stats_list = []

    for index, _ in enumerate(y_labels):
        values = [
            local_params[index].tolist(), local_sse[index],
            local_pvalues[index].tolist(), local_tvalues[index].tolist(),
            local_rsquared[index], list(X_labels)
        ]
        local_stats_dict = {key: value for key, value in zip(keys, values)}
        local_stats_list.append(local_stats_dict)
    
    beta_vector = [l.tolist() for l in local_params]

    return meanY_vector, lenY_vector, local_stats_list, beta_vector


def add_site_covariates(args, X):
    biased_X = sm.add_constant(X)
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros(
        (np.array(X).shape[0], len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns if args["state"]["clientId"] in col
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)

    return augmented_X

def get_cost(y_actual, y_predicted):
    return np.average((y_actual-y_predicted)**2)

def check_cols_to_normalize(X):
    columns_to_normalize=[]
    max_vals = X.max(axis=0).to_numpy()
    minval = np.min(max_vals[np.nonzero(max_vals)])
    X_headers = list(X.columns)
    ranges = max_vals / minval
    temp_cols_indxs = np.where(ranges > 10000)[0]
    for col_indx in temp_cols_indxs:
        columns_to_normalize.append(X_headers[col_indx])
    return columns_to_normalize

def normalize_columns(data_df, cols):
    for col in cols:
        data_df[col] = (data_df[col] - data_df[col].mean())/(data_df[col].std())
    return data_df