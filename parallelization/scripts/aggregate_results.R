cat("Aggregating results...\n")
RESULTS_PATH <- here::here("results")
method_names <- c("rf", "knn")

# load in results
preds <- purrr::map(
    method_names,
    ~ unlist(readRDS(here::here(RESULTS_PATH, sprintf("preds_%s.rds", .x))))
) |>
  setNames(method_names) |>
  dplyr::bind_cols()

# save aggregated results
write.csv(
  preds, 
  file = here::here(RESULTS_PATH, "preds_r.csv"),
  row.names = FALSE
)
cat("Done!\n")