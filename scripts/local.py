#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for multi-shot ridge
regression with decentralized statistic calculation
"""
import os
import warnings

warnings.simplefilter("ignore")

import ujson as json
import numpy as np
import sys
import regression as reg
from coinstacparsers import parsers
import pandas as pd
from local_ancillary import gather_local_stats, add_site_covariates
import utils as ut

from regression import sum_squared_error, y_estimate
import local_ancillary as lc


def local_0(args):
    input_list = args["input"]
    lamb = input_list["lambda"]

    (X, y) = parsers.fsl_parser(args)
    columns_to_normalize = lc.check_cols_to_normalize(X)

    tol = input_list["tol"]
    eta = input_list["eta"]
    max_iter = input_list["max_iter"]

    output_dict = {
        "computation_phase": "local_0",
        "tol": tol,
        "eta": eta,
        "columns_to_normalize": columns_to_normalize,
    }

    cache_dict = {
        "covariates": X.to_json(orient="records"),
        "dependents": y.to_json(orient="records"),
        "lambda": lamb,
        "max_iter": max_iter,
    }

    computation_output_dict = {
        "output": output_dict,
        "cache": cache_dict,
    }
    # raise Exception(computation_output_dict)
    return json.dumps(computation_output_dict)


def local_1(args):
    """Read data from the local sites, perform local regressions and send
    local statistics to the remote site"""

    input_list = args["input"]
    ut.log(input_list, args["state"])

    X = pd.read_json(args["cache"]["covariates"], orient="records")

    X = lc.normalize_columns(X, input_list["columns_to_normalize"])
    ut.log(
        f'\n\nNormalizing the following column values to their z-scores: {input_list["columns_to_normalize"]} \n ',
        args["state"],
    )

    y = pd.read_json(args["cache"]["dependents"], orient="records")
    y_labels = list(y.columns)

    meanY_vector, lenY_vector, local_stats_list, beta_vector = (
        lc.gather_local_stats(X, y)
    )
    ut.log(f"\nlocal stats list: {str(local_stats_list)} ", args["state"])
    augmented_X = lc.add_site_covariates(args, X)

    beta_vec_size = augmented_X.shape[1]
    X_labels = list(X.columns)

    output_dict = {
        "beta_vec_size": beta_vec_size,
        "beta_vector_local": beta_vector,
        "X_labels": X_labels,
        "augmented_X_labels": list(augmented_X.columns),
        "number_of_regressions": len(y_labels),
        "computation_phase": "local_1",
    }

    cache_dict = {
        "beta_vec_size": beta_vec_size,
        "number_of_regressions": len(y_labels),
        "covariates": augmented_X.to_json(orient="records"),
        "y_labels": y_labels,
        "mean_y_local": meanY_vector,
        "count_local": lenY_vector,
        "local_stats_list": local_stats_list,
        "max_iter": args["cache"]["max_iter"],
    }

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_output)


def local_2(args):
    X = pd.read_json(args["cache"]["covariates"], orient="records")
    y = pd.read_json(args["cache"]["dependents"], orient="records")

    X.to_csv(
        os.path.join(
            args["state"]["outputDirectory"], f'{args["state"]["clientId"]}_X.csv'
        )
    )
    y.to_csv(
        os.path.join(
            args["state"]["outputDirectory"], f'{args["state"]["clientId"]}_y.csv'
        )
    )

    beta_vec_size = args["cache"]["beta_vec_size"]
    number_of_regressions = args["cache"]["number_of_regressions"]

    mask_flag = args["input"].get(
        "mask_flag", np.zeros(number_of_regressions, dtype=bool)
    )

    biased_X = np.array(X)
    y = pd.DataFrame(y.values)

    w = args["input"]["remote_beta"]

    gradient = np.zeros((number_of_regressions, beta_vec_size))
    cost = np.zeros(number_of_regressions)

    for i in range(number_of_regressions):
        y_ = y[i]
        w_ = w[i]
        if not mask_flag[i]:
            gradient[i, :] = (1 / len(X)) * np.dot(
                biased_X.T, np.dot(biased_X, w_) - y_
            )
        cost[i] = lc.get_cost(y_actual=y[i], y_predicted=np.dot(biased_X, w_))

    output_dict = {
        "local_grad": gradient.tolist(),
        "local_cost": cost.tolist(),
        "computation_phase": "local_2",
    }

    cache_dict = {"max_iter": args["cache"]["max_iter"]}

    computation_phase = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_phase)


def local_3(args):

    output_dict = {
        "mean_y_local": args["cache"]["mean_y_local"],
        "count_local": args["cache"]["count_local"],
        "local_stats_list": args["cache"]["local_stats_list"],
        "y_labels": args["cache"]["y_labels"],
        "computation_phase": "local_3",
    }

    cache_dict = {}

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_output)


def local_4(args):
    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = pd.read_json(cache_list["covariates"], orient="records")
    y = pd.read_json(cache_list["dependents"], orient="records")
    biased_X = np.array(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local, SST_local = [], []
    for index, column in enumerate(y.columns):
        curr_y = y[column].values
        SSE_local.append(
            sum_squared_error(curr_y, y_estimate(biased_X, avg_beta_vector)[index])
        )
        SST_local.append(
            np.sum(np.square(np.subtract(curr_y, mean_y_global[index])), dtype=float)
        )

    varX_matrix_local = np.dot(biased_X.T, biased_X)

    output_dict = {
        "SSE_local": SSE_local,
        "SST_local": SST_local,
        "varX_matrix_local": varX_matrix_local.tolist(),
        "computation_phase": "local_4",
    }

    cache_dict = {}

    computation_output = {
        "output": output_dict,
        "cache": cache_dict,
    }

    return json.dumps(computation_output)


if __name__ == "__main__":

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(reg.listRecursive(parsed_args, "computation_phase"))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_0" in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_2a" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_2b" in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_3" in phase_key:
        computation_output = local_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
