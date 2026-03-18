import pandas as pd
import numpy as np
import pickle as pkl
import sys
import os
from os.path import join as oj
import time
import argparse
from joblib import Parallel, delayed

sys.path.append(".")
from python.load import load_brca_data, load_subtype_data
from python.clean import apply_log_transform
from python.fit import fit_rf_loo, fit_knn_loo

if __name__ == '__main__':

    # read in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--array_id", type=int, default=None, help="Array ID")
    parser.add_argument("--model", type=str, default=None, help="Name of model to fit")
    args = parser.parse_args()

    # translate option into selected method name
    method_names = ["rf", "knn"]
    if args.array_id is None and args.model is None:
        raise ValueError("Please provide either --array_id or --model")
    elif args.model is not None:
        assert args.model in method_names
        method_name = args.model
    else:
        assert (args.array_id >= 1) and (args.array_id <= len(method_names))
        method_name = method_names[args.array_id-1]
    print(f"Fitting {method_name}...")

    # for parallel processing
    n_cores = int(os.getenv("NSLOTS"))
    print("Number of cores available:", n_cores)

    # Helper variables
    n = n_cores * 10  # Do leave-one-out for the first n samples
    DATA_PATH = oj("data")
    RESULTS_PATH = oj("results")

    # Load data
    X = load_brca_data(DATA_PATH).values  # convert to numpy array
    X = apply_log_transform(X)
    y = load_subtype_data(DATA_PATH)['subtype'].values  # convert to numpy array

    if method_name == "rf":
        fit_loo = fit_rf_loo
    elif method_name == "knn":
        fit_loo = fit_knn_loo
    else:
        raise ValueError("Unknown method name")

    start_time = time.time()
    preds = Parallel(n_jobs=n_cores)(delayed(fit_loo)(i, X, y) for i in range(n))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time with joblib parallelization:", execution_time)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(oj(RESULTS_PATH, f"preds_{method_name}.pkl"), "wb") as f:
        pkl.dump(preds, f)
