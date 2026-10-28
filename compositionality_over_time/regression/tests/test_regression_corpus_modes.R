#!/usr/bin/env Rscript
# Test the current regression interface for both COHA context modes.
# Fabricated ratings and temporary outputs verify fitting and completion receipts.
arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
test_dir <- dirname(normalizePath(sub("^--file=", "", arg[[1L]])))
regression_dir <- normalizePath(file.path(test_dir, "..", "modified_regressions"))
source(file.path(regression_dir, "coha_v3_regression_data.R"))
source(file.path(regression_dir, "batch_status.R"))
source(file.path(regression_dir, "report_grid_filter.R"))

main <- function() {
  root <- tempfile("coha-regression-handoff-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  ratings_root <- file.path(test_dir, "..", "..", "tests", "test_data", "ratings")
  rows <- do.call(rbind, lapply(c("reddy_90.txt", "cordeiro_90.txt", "cordeiro_100.txt"),
                              function(name) read.delim(file.path(ratings_root, name))))
  rows[["sim_cpf_beta:0"]] <- sin(seq_len(nrow(rows)) / 7) + cos(seq_len(nrow(rows)) / 13)
  fitter <- file.path(regression_dir, "regression_exps.r")
  rscript <- file.path(R.home("bin"), "Rscript")
  for (mode in c("local5", "sentence")) {
    input_root <- file.path(root, mode, "inputs")
    out <- file.path(root, mode, "results")
    dir.create(file.path(input_root, mode), recursive = TRUE)
    dir.create(out)
    write.csv(rows, coha_v3_feature_path(input_root, mode, 10L, "features",
              "Agnostic", "RAW", "UnTagged", 10000L, 0L), row.names = FALSE)
    arguments <- c(shQuote(fitter), "--corpus", "coha", "--context-mode", mode,
      "--input-root", shQuote(input_root), "--output-dir", shQuote(out),
      "--context-size", "10", "--setting", "Agnostic", "--ppmi", "RAW",
      "--tagging", "UnTagged", "--timespans", "10000", "--cutoffs", "0",
      "--targets", "compound", "--feature-sets", "sim_cpf_beta", "--pilot")
    log <- file.path(root, paste0(mode, ".log"))
    status <- system2(rscript, arguments, stdout = log, stderr = log)
    if (status != 0L) stop(paste(tail(readLines(log), 50L), collapse = "\n"))
    predictions <- read.csv(list.files(out, pattern = "^predictions_coha_v3_", full.names = TRUE))
    stopifnot(nrow(predictions) == 100L, all(predictions$corpus == "coha"),
              all(predictions$context_mode == mode), all(is.finite(predictions$predicted_spearman)))
    task <- list(task_id = paste0("coha-", mode), corpus = "coha", context_mode = mode,
      tagging = "UnTagged", vocab = "10", weight = "RAW", setting = "Agnostic",
      timespan = "10000", cutoff = "0", input_root = input_root, out = out,
      feature_sets = "sim_cpf_beta", targets = "compound", frequency_family = "freq",
      productivity_family = "prod", folds = "3", repeats = "1", alpha_count = "3",
      lambda_count = "12", permutations = "1", train_variant = "modified", test_variant = "modified")
    inspected <- google_v3_status_run(as.data.frame(task, stringsAsFactors = FALSE), regression_dir)
    stopifnot(inspected$status == "completed")
    Sys.setenv(GRID_CORPUS = "coha", GRID_MODES = mode, GRID_VOCABS = "10",
      GRID_WEIGHTINGS = "RAW", GRID_SETTINGS = "Agnostic", GRID_TAGGINGS = "UnTagged",
      GRID_TIMESPANS = "10000", GRID_CUTOFFS = "0", GRID_TARGETS = "compound",
      GRID_TRAIN_VARIANT = "modified", GRID_TEST_VARIANT = "modified",
      GRID_FOLDS = "3", GRID_REPEATS = "1", GRID_ALPHA_COUNT = "3",
      GRID_LAMBDA_COUNT = "12", GRID_PERMUTATIONS = "1")
    coverage <- google_v3_validate_results(out, input_root)
    stopifnot(nrow(coverage) == 1L, coverage$corpus == "coha", coverage$context_mode == mode)
    receipt_path <- list.files(out, pattern = "^complete_coha_v3_", full.names = TRUE)
    receipt <- readRDS(receipt_path)
    wrong <- receipt
    wrong$corpus <- "google"
    saveRDS(wrong, receipt_path)
    stopifnot(inherits(try(google_v3_validate_results(out, input_root), silent = TRUE), "try-error"))
    inspected <- google_v3_status_run(as.data.frame(task, stringsAsFactors = FALSE), regression_dir)
    stopifnot(inspected$status == "incompatible", grepl("corpus", inspected$detail))
    saveRDS(receipt, receipt_path)
  }
  stopifnot(inherits(try(coha_v3_validate_study("google", "sentence"), silent = TRUE), "try-error"),
            inherits(try(coha_v3_validate_study("other", "local5"), silent = TRUE), "try-error"))
  cat("COHA regression handoff passed: both modes fit, retain corpus identity and verify completion\n")
}

main()
