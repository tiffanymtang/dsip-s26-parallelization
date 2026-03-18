# to submit job: qsub -N parallel_example submit_r_job.sh scripts/parallel_example
library(future)
library(furrr)

# load in helper functions
for (fname in list.files(here::here("R"), pattern = "*.R")) {
  source(here::here(file.path("R", fname)))
}

# set seed
set.seed(242)

# set up parallel processing
n_cores <- as.integer(Sys.getenv("NSLOTS"))
print(paste("Using the following number of cores:", n_cores))
plan(multicore, workers = n_cores)

# helper variables
n <- n_cores * 10  # do leave-one-out for this many number of samples
DATA_PATH <- here::here("data")
RESULTS_PATH <- here::here("results")

# load in TCGA breast cancer data
X <- load_brca_data(DATA_PATH) |> 
  apply_log_transform()
y <- as.factor(load_subtype_data(DATA_PATH)$subtype)

start_time <- Sys.time()
preds_furrr <- furrr::future_map(
  1:n,
  function(i) {
    fit_rf_loo(i, X, y)
  },
  .options = furrr::furrr_options(seed = TRUE)
)
end_time <- Sys.time()
execution_time <- end_time - start_time
print(execution_time)

if (!dir.exists(RESULTS_PATH)) {
  dir.create(RESULTS_PATH)
}
saveRDS(
  preds_furrr, 
  file = here::here(RESULTS_PATH, "preds_furrr.rds")
)