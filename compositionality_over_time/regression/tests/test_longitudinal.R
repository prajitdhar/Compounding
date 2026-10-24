#!/usr/bin/env Rscript
# Synthetic historical features and a synthetic rating roster. All
# model, input and output artifacts are isolated in a temporary directory.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
test_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
driver_path <- file.path(test_dir, "..", "longitudinal", "longitudinal.R")
source(driver_path)

expect_error <- function(expression, pattern = NULL) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  if (is.null(error) || (!is.null(pattern) && !grepl(pattern, conditionMessage(error), fixed = TRUE))) {
    stop("Expected error containing: ", pattern, "; got: ", if (is.null(error)) "no error" else conditionMessage(error))
  }
}

json_write <- function(value, path) jsonlite::write_json(value, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
identity_json <- function(value) as.character(jsonlite::toJSON(value, auto_unbox = TRUE, digits = NA))
record <- function(path) list(path = normalizePath(path), sha256 = google_longitudinal_sha(path), bytes = file.info(path)$size)

test_current_checkpoint_identity <- function(root) {
  directory <- file.path(root, "checkpoint-identity")
  dir.create(directory)
  paths <- c(google_longitudinal_script, google_longitudinal_helper)
  config <- list(options = list(context_size = 5000L, folds = 3L),
    input_identity = strrep("b", 64L), input_sha256 = c(input = strrep("c", 64L)),
    software = list(R = R.version.string),
    source_sha256 = setNames(vapply(paths, google_longitudinal_sha, character(1)), paths))
  frozen <- list(predictors = "test_predictor", prediction = 2.5)
  checkpoint <- list(config = config, frozen = frozen,
    frozen_identity = digest::digest(frozen, algo = "sha256"))
  model_path <- file.path(directory, "frozen_modern_model.rds")
  saveRDS(checkpoint, model_path)
  before <- google_longitudinal_sha(model_path)
  stopifnot(is.null(google_longitudinal_checkpoint_source_info(checkpoint, config, model_path)))
  altered <- list(config, config, config, config)
  altered[[1L]]$options$context_size <- 3000L
  altered[[2L]]$input_identity <- strrep("d", 64L)
  altered[[3L]]$input_sha256[["input"]] <- strrep("d", 64L)
  altered[[4L]]$source_sha256[[google_longitudinal_script]] <- strrep("d", 64L)
  for (changed in altered) expect_error(
    google_longitudinal_checkpoint_source_info(checkpoint, changed, model_path), "identity changed")
  changed <- checkpoint; changed$frozen$prediction <- 99
  expect_error(google_longitudinal_checkpoint_source_info(changed, config, model_path), "content identity")
  changed <- checkpoint; changed$frozen_identity <- NULL
  expect_error(google_longitudinal_checkpoint_source_info(changed, config, model_path), "content identity")
  stopifnot(before == google_longitudinal_sha(model_path))
}

make_test_data <- function(root) {
  args <- google_longitudinal_defaults()
  args$feature_root <- file.path(root, "modern")
  args$cumulative_root <- file.path(root, "cumulative")
  args$output_dir <- file.path(root, "results")
  args$pilot <- TRUE
  args <- google_longitudinal_options(args)
  dir.create(file.path(args$feature_root, "local5"), recursive = TRUE)
  ratings_root <- Sys.getenv("TEST_RATINGS_DIR", unset = file.path(test_dir, "..", "..", "tests", "test_data", "ratings"))
  ratings <- do.call(rbind, lapply(c("reddy90", "cordeiro90", "cordeiro100"), function(name) {
    filename <- c(reddy90 = "reddy_90.txt", cordeiro90 = "cordeiro_90.txt", cordeiro100 = "cordeiro_100.txt")[[name]]
    rows <- read.delim(file.path(ratings_root, filename), check.names = FALSE)
    rows$source <- name; rows$is_adj <- as.logical(rows$is_adj); rows$is_original <- as.logical(rows$is_original)
    rows
  }))
  stopifnot(nrow(ratings) == 287L)
  # One real predictor exercises glmnet zero padding and fold-local imputation.
  ratings[["sim_cpf_beta:0"]] <- ratings$compositionality + sin(seq_len(nrow(ratings)) / 7)
  ratings[["sim_cpf_beta:0"]][seq_len(nrow(ratings)) %% 23L == 0L] <- NA_real_
  ratings[["eligible:0"]] <- 1L
  modern_path <- coha_v3_feature_path(args$feature_root, "local5", 5000L, "features", "Pooled", "PPMI", "UnTagged", 10000L, 0L)
  write.csv(ratings, modern_path, row.names = FALSE, na = "NA")
  modern_source <- record(modern_path)
  input_material <- list(context_mode = "local5", context_size = 5000L, setting = "Pooled",
    condition = "CompoundPooled", ppmi = TRUE, tagged = FALSE, cutoff = 0L, time_step = 10L,
    feature_definition_version = 3L, beta_policy = "clipped_unit_interval", modern_source = modern_source)
  input_text <- identity_json(input_material)
  input_id <- google_longitudinal_hash_text(input_text)
  manifests <- vector("list", 2L)
  for (i in seq_along(manifests)) {
    endpoint <- c(1900L, 1910L)[[i]]
    directory <- file.path(args$cumulative_root, "local5", paste0("through_", endpoint))
    dir.create(directory, recursive = TRUE)
    snapshot <- ratings
    if (i == 1L) {
      snapshot[["sim_cpf_beta:0"]] <- snapshot[["sim_cpf_beta:0"]] - 0.5
      snapshot[["eligible:0"]][snapshot$source == "cordeiro100"] <- 0L
      # The shared wide writer uses NA for ineligible items at early endpoints.
      missing_eligibility <- which(snapshot$source == "cordeiro100" & snapshot$is_original)[c(2L, 3L)]
      snapshot[["eligible:0"]][missing_eligibility] <- NA_integer_
      # The manually added contact_lens must map to canonical contact_lenses.
      selected <- snapshot$source == "cordeiro100" & !snapshot$is_original
      snapshot[["eligible:0"]][selected] <- 1L
      snapshot[["sim_cpf_beta:0"]][snapshot$compound == "contact_lens"] <- -0.5
      # An attested row with no predictors must remain absent from predictions.
      # Select a retained test item: original lexical variants can be replaced
      # by their modified counterparts before historical prediction.
      unavailable <- which(snapshot$source == "cordeiro100" & snapshot$is_original &
        !snapshot$compound %in% coha_v3_variant_pairs$original)[[1L]]
      snapshot[["eligible:0"]][unavailable] <- 1L
      snapshot[["sim_cpf_beta:0"]][unavailable] <- NA_real_
    }
    csv <- file.path(directory, basename(modern_path))
    write.csv(snapshot, csv, row.names = FALSE, na = "NA")
    files <- list(record(csv))
    snapshot_material <- list(input_identity = input_id, endpoint_decade = endpoint, end_year = endpoint + 9L, files = files)
    snapshot_text <- identity_json(snapshot_material)
    info <- c(list(schema_version = 1L, stage = "google_cumulative_features"), input_material,
      list(input_identity = input_id, input_identity_material = input_text,
        snapshot_identity = google_longitudinal_hash_text(snapshot_text), snapshot_identity_material = snapshot_text,
        endpoint_decade = endpoint, end_year = endpoint + 9L, full_input_last_decade = 1910L, files = files))
    path <- file.path(directory, "cumulative_manifest.json")
    json_write(info, path); manifests[[i]] <- record(path)
  }
  json_write(list(status = "complete", input_identity = input_id, endpoints = c(1900L, 1910L), manifests = manifests),
             file.path(args$cumulative_root, "features_complete.json"))
  list(args = args, ratings = ratings, modern_path = modern_path, manifests = manifests)
}

main <- function() {
  if (!all(vapply(c("jsonlite", "digest", "caret", "glmnet", "ggplot2"), requireNamespace, logical(1), quietly = TRUE))) {
    stop("The longitudinal smoke test requires the existing R regression environment")
  }
  root <- tempfile("google-longitudinal-test-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  test_current_checkpoint_identity(root)
  data <- make_test_data(root); args <- data$args
  stopifnot(!dir.exists(args$output_dir)) # Sourcing driver must never auto-run.
  input <- google_longitudinal_input(args)
  stopifnot(nrow(input$split$train) == 180L, nrow(input$split$test) == 100L,
    all(coha_v3_variant_pairs$modified %in% c(input$split$train$compound, input$split$test$compound)))
  # Residual R2 distinguishes bad calibrated predictions from perfect ranking.
  stopifnot(google_longitudinal_r2(c(101, 102, 103), c(1, 2, 3)) < 0,
    google_longitudinal_spearman(c(101, 102, 103), c(1, 2, 3)) == 1,
    is.na(google_longitudinal_r2(c(1, 1), c(2, 2))))
  reordered <- data$ratings[nrow(data$ratings):1L, , drop = FALSE]
  stopifnot(google_longitudinal_check_parity(data$ratings, reordered))
  changed <- reordered; changed[["sim_cpf_beta:0"]][[1L]] <- 900
  expect_error(google_longitudinal_check_parity(data$ratings, changed), "Final cumulative feature parity failed")
  changed <- reordered; changed$avgHead[[1L]] <- -10
  expect_error(google_longitudinal_check_parity(data$ratings, changed), "Rating payload differs")
  expect_error(google_longitudinal_identity(list(input_identity = strrep("a", 64L), input_identity_material = "{}"), "input_identity"),
               "Invalid or unverified input_identity")
  receipt <- google_longitudinal_run(args)
  stopifnot(receipt$status == "complete", receipt$growth_status == "too_few_eligible_compounds_or_endpoints")
  frozen <- readRDS(file.path(args$output_dir, "frozen_modern_model.rds"))$frozen
  stopifnot(identical(frozen$predictors, "sim_cpf_beta:0"), !is.null(frozen$preprocessing),
    all(is.finite(frozen$cv_predictions$pred)), nrow(frozen$cv_predictions) == 180L,
    all(frozen$cv_predictions$cv_objective == "Spearman"))
  trajectories <- read.csv(file.path(args$output_dir, "trajectories.csv"))
  coverage <- read.csv(file.path(args$output_dir, "coverage.csv"))
  compound_coverage <- read.csv(file.path(args$output_dir, "coverage_by_compound.csv"))
  stopifnot(nrow(compound_coverage) == 200L, all(coverage$total_test == 100L),
    coverage$unattested_test[[1L]] > 0L, coverage$all_predictors_missing_test[[1L]] >= 1L,
    coverage$predicted_test[[2L]] == sum(!is.na(google_longitudinal_x(input$split$test, args, frozen$predictors)[[frozen$predictors]])),
    !any(trajectories$compound %in% compound_coverage$compound[compound_coverage$status == "unattested"] & trajectories$endpoint_decade == 1900L),
    any(trajectories$compound == "contact_lens" & trajectories$canonical_compound == "contact_lenses"),
    all(is.finite(trajectories$predicted_compositionality)), file.exists(file.path(args$output_dir, "growth_trajectories.png")))
  predictions <- read.csv(file.path(args$output_dir, "modern_predictions.csv"))
  metrics <- read.csv(file.path(args$output_dir, "modern_performance.csv"))
  stopifnot(isTRUE(all.equal(metrics$test_r2, google_longitudinal_r2(predictions$predicted, predictions$observed))),
    isTRUE(all.equal(metrics$test_mae, mean(abs(predictions$predicted - predictions$observed)))))
  outputs <- list.files(args$output_dir, full.names = TRUE)
  hashes <- tools::md5sum(outputs); times <- file.info(outputs)$mtime
  google_longitudinal_run(args)
  stopifnot(identical(hashes, tools::md5sum(outputs)), identical(times, file.info(outputs)$mtime))
  # An interrupted run can reuse the immutable model checkpoint and preprocessing.
  receipt_path <- file.path(args$output_dir, "complete_longitudinal.rds")
  stopifnot(file.rename(receipt_path, file.path(root, "saved-completion.rds")))
  model_path <- file.path(args$output_dir, "frozen_modern_model.rds")
  model_hash <- google_longitudinal_sha(model_path); model_time <- file.info(model_path)$mtime
  # A separate R process exercises command-line parsing and RDS checkpoint
  # identities across sessions, rather than only within this interpreter.
  log_path <- file.path(root, "cli-resume.log")
  cli_status <- system2(file.path(R.home("bin"), "Rscript"),
    c(shQuote(driver_path), "--feature-root", shQuote(args$feature_root),
      "--cumulative-root", shQuote(args$cumulative_root), "--output-dir", shQuote(args$output_dir), "--pilot"),
    stdout = log_path, stderr = log_path)
  if (cli_status != 0L) stop(paste(readLines(log_path), collapse = "\n"))
  stopifnot(model_hash == google_longitudinal_sha(model_path), identical(model_time, file.info(model_path)$mtime))
  if (requireNamespace("lme4", quietly = TRUE)) {
    # Keep Matrix namespace loaded but unattached: this is exactly the dispatch
    # situation needed to exercise explicit covariance-matrix conversion.
    stopifnot(!"package:Matrix" %in% search())
    growth_data <- expand.grid(compound_id = paste0("synthetic_", 1:12), endpoint_decade = seq(1900L, 1950L, 10L))
    index <- as.integer(factor(growth_data$compound_id))
    time <- (growth_data$endpoint_decade - 1900L) / 10
    growth_data$predicted_compositionality <- 2 + index / 30 + (0.05 + index / 100) * time + sin(index + time) / 20
    growth_data$end_year <- growth_data$endpoint_decade + 9L
    growth <- google_longitudinal_growth(growth_data, seq(1900L, 1950L, 10L))
    stopifnot(growth$status == "fitted", nrow(growth$fitted) == 72L,
      all(is.finite(growth$slope$estimate)), all(is.finite(growth$slope$std_error)),
      is.logical(growth$converged), is.logical(growth$singular))
    growth_plot <- file.path(root, "fitted-growth.png")
    google_longitudinal_plot(growth_data, growth, growth_plot)
    stopifnot(file.exists(growth_plot), file.info(growth_plot)$size > 0L,
      !"package:Matrix" %in% search())
  } else cat("lme4 unavailable: random-slope fit branch was not exercised\n")
  # Completion receipts are not enough: edited input artifacts must be rejected.
  writeLines(c(readLines(data$modern_path), ""), data$modern_path)
  expect_error(google_longitudinal_input(args), "Artifact hash/size mismatch")
  cat("Google longitudinal tests passed: identities/parity, variants, pilot/CV, growth fit/plot and strict checkpoint reuse\n")
}

if (sys.nframe() == 0L) main()
