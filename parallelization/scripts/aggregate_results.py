from os.path import join as oj
import pandas as pd
import pickle as pkl


if __name__ == "__main__":
    print("Aggregating results...")
    RESULTS_PATH = oj("results")
    method_names = ["rf", "knn"]

    # Load the results
    results_dict = {}
    for method_name in method_names:
        results_dict[method_name] = pkl.load(
            open(oj(RESULTS_PATH, f"preds_{method_name}.pkl"), "rb")
        )
    results = pd.DataFrame(results_dict)

    # Save the aggregated results
    results.to_csv(oj(RESULTS_PATH, "preds_python.csv"), index=False)
    print("Done!")
    