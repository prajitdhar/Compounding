#!/usr/bin/env Rscript
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "modified_regressions", "report_grid_filter.R"))

main <- function() {
  task_root <- tempfile("google-report-completeness-")
  dir.create(file.path(task_root, "inputs", "local5"), recursive = TRUE)
  dir.create(file.path(task_root, "results"))
  on.exit(unlink(task_root, recursive = TRUE), add = TRUE)
  input_root <- file.path(task_root, "inputs")
  output_root <- file.path(task_root, "results")
  Sys.setenv(GRID_CORPUS = "google", GRID_MODES = "local5", GRID_VOCABS = "5000", GRID_WEIGHTINGS = "RAW",
    GRID_SETTINGS = "Agnostic", GRID_TAGGINGS = "UnTagged Tagged", GRID_TIMESPANS = "10000,50",
    GRID_CUTOFFS = "0 10", GRID_TARGETS = "compound", GRID_TRAIN_VARIANT = "modified", GRID_TEST_VARIANT = "modified")
  Sys.unsetenv(c("GRID_FOLDS", "GRID_REPEATS", "GRID_ALPHA_COUNT", "GRID_LAMBDA_COUNT", "GRID_PERMUTATIONS"))
  code_md5 <- unname(tools::md5sum(file.path(regression_root, "modified_regressions",
    c("regression_exps.r", "coha_v3_regression_data.R"))))
  output_types <- c("rsquared", "coefimp", "permimp", "familyimp", "predictions", "cvgrid", "cvfolds")
  for (tag in c("UnTagged", "Tagged")) for (width in c(10000L, 50L)) for (cutoff in c(0L, 10L)) {
    inputs <- file.path(input_root, "local5", sprintf("features_CompoundAgnostic_withSetting_RAW_%s_ctx5000_%s_%s_na.csv", tag, width, cutoff))
    if (width != 10000L) inputs <- c(inputs, file.path(input_root, "local5", sprintf("temporal_CompoundAgnostic_withSetting_RAW_%s_ctx5000_%s_%s_na.csv", tag, width, cutoff)))
    for (input in inputs) write.csv(data.frame(synthetic = 1L), input, row.names = FALSE)
    for (pattern in if (tag == "Tagged") c("nn", "an") else "all") {
      suffix <- sprintf("google_v3_local5_ctx5000_%s_RAW_Agnostic_%s_%s_%s_freq_prod_trainmodified_testmodified", tag, width, cutoff, pattern)
      paths <- file.path(output_root, paste0(output_types, "_", suffix, ".csv"))
      scores <- data.frame(corpus = "google", features = "sim_cpf_beta", y = "compound", train_dim = 180L, test_dim = 100L,
        context_mode = "local5", context_size = 5000L, tag = tag, ppmi = "RAW", setting = "Agnostic",
        timespan = width, cutoff = cutoff, pattern = pattern, train_variant = "modified", test_variant = "modified",
        frequency_family = "freq", productivity_family = "prod", impute = "fold_median")
      for (path in paths) write.csv(scores, path, row.names = FALSE)
      saveRDS(list(input_md5 = unname(tools::md5sum(inputs)), code_md5 = code_md5,
        corpus = "google", context_mode = "local5",
        targets = "compound", feature_sets = c("all", "sim_cpf_beta"), train_variant = "modified", test_variant = "modified",
        folds = 5L, repeats = 10L, alpha_count = 10L, lambda_count = 200L, permutations = 5L,
        frequency_family = "freq", productivity_family = "prod"),
        file.path(output_root, paste0("complete_", suffix, ".rds")))
    }
  }
  coverage <- suppressMessages(google_v3_validate_results(output_root, input_root))
  stopifnot(nrow(coverage) == 12L, all(coverage$n_fitted_models == 1L), all(coverage$n_skipped_recipes == 1L))
  suffix <- "google_v3_local5_ctx5000_Tagged_RAW_Agnostic_50_10_an_freq_prod_trainmodified_testmodified"
  receipt <- file.path(output_root, paste0("complete_", suffix, ".rds"))
  saved <- readRDS(receipt)
  file.rename(receipt, paste0(receipt, ".saved"))
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  file.rename(paste0(receipt, ".saved"), receipt)
  missing <- file.path(output_root, paste0("permimp_", suffix, ".csv"))
  file.rename(missing, paste0(missing, ".saved"))
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  file.rename(paste0(missing, ".saved"), missing)
  wrong <- saved
  wrong$code_md5 <- c("modified", "modified")
  saveRDS(wrong, receipt)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  saveRDS(saved, receipt)
  wrong <- saved
  wrong$folds <- 3L
  saveRDS(wrong, receipt)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  saveRDS(saved, receipt)
  Sys.setenv(GRID_FOLDS = "3")
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  Sys.unsetenv("GRID_FOLDS")
  scores_path <- file.path(output_root, paste0("rsquared_", suffix, ".csv"))
  saved_scores <- read.csv(scores_path)
  wrong_scores <- saved_scores
  wrong_scores$timespan <- 999L
  write.csv(wrong_scores, scores_path, row.names = FALSE)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  write.csv(rbind(saved_scores, saved_scores), scores_path, row.names = FALSE)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  wrong_scores <- saved_scores
  wrong_scores$features <- "unrequested_recipe"
  write.csv(wrong_scores, scores_path, row.names = FALSE)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  write.csv(saved_scores, scores_path, row.names = FALSE)
  wrong <- saved
  wrong$input_md5 <- rep("modified", length(wrong$input_md5))
  saveRDS(wrong, receipt)
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  saveRDS(saved, receipt)
  Sys.setenv(GRID_TARGETS = "compound head")
  stopifnot(inherits(try(google_v3_validate_results(output_root, input_root), silent = TRUE), "try-error"))
  Sys.setenv(GRID_TARGETS = "compound", GRID_CUTOFFS = "0")
  file.rename(receipt, paste0(receipt, ".saved"))
  # Removed cells must not invalidate an explicitly narrowed requested grid.
  coverage <- suppressMessages(google_v3_validate_results(output_root, input_root))
  stopifnot(nrow(coverage) == 6L, all(coverage$cutoff == 0L))
  cat("Google report completeness passed: missing cells/CSV, input/code/target/tuning changes, wrong metadata and duplicate models are rejected; legitimate skipped recipes remain documented\n")
}

main()
