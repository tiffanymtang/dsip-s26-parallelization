# to submit job: qsub -N parallel_example_with_args submit_r_array_job.sh scripts/parallel_example_with_args
library(future)
library(furrr)
library(optparse)

# load in helper functions
for (fname in list.files(here::here("R"), pattern = "*.R")) {
  source(here::here(file.path("R", fname)))
}

# set seed
set.seed(242)

# define the list of command line options
option_list <- list(
  make_option(
    "--array_id", type = "integer", default = NULL, 
    help = "array id"
  ),
  make_option(
    "--model", type = "character", default = NULL,
    help = "name of model to fit"
  )
)
# parse the command line options
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# translate option into selected method name
method_names <- c("rf", "knn")
if (is.null(opt$array_id) && is.null(opt$model)) {
  stop("Please provide either --array_id or --model")
} else if (!is.null(opt$model)) {
  method_name <- match.arg(opt$model, method_names)
} else {
  if (opt$array_id > length(method_names) || opt$array_id < 0) {
    stop("Invalid array id. Must be between 0 and ", length(method_names))
  }
  method_name <- method_names[opt$array_id]
}
cat(sprintf("Fitting %s...\n", method_name))

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

if (method_name == "rf") {
  fit_loo <- fit_rf_loo
} else if (method_name == "knn") {
  fit_loo <- fit_knn_loo
} else {
  stop("Invalid method name")
}

start_time <- Sys.time()
preds_furrr <- furrr::future_map(
  1:n,
  function(i) {
    fit_loo(i, X, y)
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
  file = here::here(RESULTS_PATH, sprintf("preds_%s.rds", method_name))
)