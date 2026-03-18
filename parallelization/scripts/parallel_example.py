import pandas as pd
import numpy as np
import pickle as pkl
import sys
import os
from os.path import join as oj
import time
from joblib import Parallel, delayed

sys.path.append(".")
from python.load import load_brca_data, load_subtype_data
from python.clean import apply_log_transform
from python.fit import fit_rf_loo

if __name__ == '__main__':

    # Helper variables
    DATA_PATH = oj("data")
    RESULTS_PATH = oj("results")
    n_cores = int(os.getenv("NSLOTS"))
    print("Number of cores available:", n_cores)
    n = n_cores * 10  # Do leave-one-out for the first n samples

    # Load data
    X = load_brca_data(DATA_PATH).values  # convert to numpy array
    X = apply_log_transform(X)
    y = load_subtype_data(DATA_PATH)['subtype'].values  # convert to numpy array

    start_time = time.time()
    preds = Parallel(n_jobs=n_cores)(delayed(fit_rf_loo)(i, X, y) for i in range(n))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time with joblib parallelization:", execution_time)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(oj(RESULTS_PATH, "preds_joblib.pkl"), "wb") as f:
        pkl.dump(preds, f)
