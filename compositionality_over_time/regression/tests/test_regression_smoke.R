#!/usr/bin/env Rscript
# End-to-end regression check using synthetic predictors and fabricated rating test data.
# All feature/result files live in a temporary directory; no production inputs change.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "modified_regressions", "coha_v3_regression_data.R"))

main <- function() {
  corpus <- Sys.getenv("TEST_CORPUS", unset = "google")
  mode <- Sys.getenv("TEST_CONTEXT_MODE", unset = "local5")
  coha_v3_validate_study(corpus, mode)
  task_root <- tempfile("google-regression-smoke-")
  dir.create(file.path(task_root, "inputs", mode), recursive = TRUE)
  dir.create(file.path(task_root, "results"))
  on.exit(unlink(task_root, recursive = TRUE), add = TRUE)
  ratings_root <- Sys.getenv("TEST_RATINGS_DIR", unset = file.path(regression_root, "..", "tests", "test_data", "ratings"))
  ratings <- do.call(rbind, lapply(c("reddy90", "cordeiro90", "cordeiro100"), function(name) {
    filename <- c(reddy90 = "reddy_90.txt", cordeiro90 = "cordeiro_90.txt", cordeiro100 = "cordeiro_100.txt")[[name]]
    rows <- read.delim(file.path(ratings_root, filename))
    rows$source <- name
    rows$is_adj <- as.logical(rows$is_adj)
    rows$is_original <- as.logical(rows$is_original)
    rows
  }))
  input_root <- file.path(task_root, "inputs")
  output_root <- file.path(task_root, "results")
  for (tagging in c("UnTagged", "Tagged")) {
    rows <- ratings
    if (tagging == "Tagged") {
      noun <- rows
      noun$modifier <- paste0(noun$modifier, "_NOUN")
      noun$head <- paste0(noun$head, "_NOUN")
      mixed <- rows[rows$is_adj, , drop = FALSE]
      mixed$modifier <- paste0(mixed$modifier, "_ADJ")
      mixed$head <- paste0(mixed$head, "_NOUN")
      rows <- rbind(noun, mixed)
    }
    for (width in c(10000L, 50L)) {
      static <- temporal <- rows
      periods <- if (width == 10000L) 0L else c(1900L, 1950L)
      bases <- unique(unlist(coha_v3_feature_groups[c("cordeiro", "cosine_sim", "info_theory", "association_counts", "freq", "prod")], use.names = FALSE))
      for (period in periods) for (j in seq_along(bases)) {
        values <- sin(seq_len(nrow(rows)) * (j + 1L) / 7) + cos(seq_len(nrow(rows)) / 13) + period / 10000
        # Missing values are not globally imputed before the R fitting process.
        values[seq_len(nrow(rows)) %% 23L == 0L] <- NA_real_
        static[[paste0(bases[[j]], ":", period)]] <- values
      }
      for (j in seq_along(coha_v3_feature_groups$temporal)) {
        temporal[[paste0(coha_v3_feature_groups$temporal[[j]], ":1950")]] <- sin(seq_len(nrow(rows)) * j / 11)
      }
      write.csv(static, coha_v3_feature_path(input_root, mode, 5000L, "features", "Agnostic", "RAW", tagging, width, 0L), row.names = FALSE, na = "")
      if (width != 10000L) write.csv(temporal, coha_v3_feature_path(input_root, mode, 5000L, "temporal", "Agnostic", "RAW", tagging, width, 0L), row.names = FALSE, na = "")
    }
  }
  rscript <- file.path(R.home("bin"), "Rscript")
  fitter <- file.path(regression_root, "modified_regressions", "regression_exps.r")
  fit <- function(tagging, validation = FALSE) {
    arguments <- c(shQuote(fitter), "--input-root", shQuote(input_root), "--output-dir", shQuote(output_root),
      "--corpus", corpus, "--context-mode", mode, "--context-size", "5000", "--ppmi", "RAW", "--setting", "Agnostic", "--tagging", tagging,
      "--timespans", "10000,50", "--cutoffs", "0", "--targets", "compound", "--feature-sets",
      "all_no_setting,representation_only,cordeiro,sim_cpf_beta,all,all_static,temporal", "--cores", "1", "--pilot")
    if (validation) arguments <- c(arguments, "--validate-only")
    log <- file.path(task_root, paste0(tagging, if (validation) "-check" else "-fit", ".log"))
    status <- system2(rscript, arguments, stdout = log, stderr = log)
    if (status != 0L) stop(paste(tail(readLines(log), 50L), collapse = "\n"))
    invisible(readLines(log))
  }
  for (tagging in c("UnTagged", "Tagged")) fit(tagging, validation = TRUE)
  for (tagging in c("UnTagged", "Tagged")) fit(tagging)
  receipts <- list.files(output_root, pattern = paste0("^complete_", corpus, "_v3_.*\\.rds$"), full.names = TRUE)
  stopifnot(length(receipts) == 6L)
  files <- list.files(output_root, pattern = "\\.(csv|rds)$", full.names = TRUE)
  identities <- tools::md5sum(files)
  mtimes <- file.info(files)$mtime
  for (tagging in c("UnTagged", "Tagged")) {
    log <- fit(tagging)
    stopifnot(sum(grepl("Skipping complete configuration:", log)) == if (tagging == "Tagged") 4L else 2L)
  }
  stopifnot(identical(identities, tools::md5sum(files)), identical(mtimes, file.info(files)$mtime))
  predictions <- do.call(rbind, lapply(list.files(output_root, pattern = "^predictions_.*\\.csv$", full.names = TRUE), read.csv))
  stopifnot(all(predictions$corpus == corpus), all(predictions$context_mode == mode),
            all(is.finite(predictions$predicted_spearman)),
            identical(sort(unique(predictions$pattern)), c("all", "an", "nn")),
            all(predictions$train_variant == "modified"), all(predictions$test_variant == "modified"))
  keys <- do.call(paste, c(predictions[c("tag", "pattern", "timespan", "features")], sep = "\r"))
  stopifnot(all(table(keys) == 100L))
  scores <- do.call(rbind, lapply(list.files(output_root, pattern = "^rsquared_.*\\.csv$", full.names = TRUE), read.csv))
  stopifnot(all(scores$train_dim == 180L), all(scores$test_dim == 100L), all(is.finite(scores$CVTrainSpearman)))
  report_env <- c(GRID_CORPUS = corpus, GRID_OUT = output_root, GRID_INPUT_ROOT = input_root, GRID_MODES = mode, GRID_VOCABS = "5000",
    GRID_WEIGHTINGS = "RAW", GRID_SETTINGS = "Agnostic", GRID_TAGGINGS = "UnTagged Tagged", GRID_TARGETS = "compound",
    GRID_TIMESPANS = "10000,50", GRID_CUTOFFS = "0", GRID_CUTOFF = "0", GRID_TRAIN_VARIANT = "modified", GRID_TEST_VARIANT = "modified",
    GRID_N_BOOT = "100", GRID_N_PERM = "100", GRID_REFERENCE_MODE = mode, GRID_REFERENCE_VOCAB = "5000",
    GRID_REFERENCE_WEIGHTING = "RAW", GRID_REFERENCE_SETTING = "Agnostic", GRID_RUN_TEMPORAL = "1", GRID_RUN_ENCODINGS = "0",
    GRID_FOLDS = "3", GRID_REPEATS = "1", GRID_ALPHA_COUNT = "3", GRID_LAMBDA_COUNT = "12", GRID_PERMUTATIONS = "1")
  report <- file.path(regression_root, "modified_regressions", "workflow_report.R")
  report_log <- file.path(task_root, "report.log")
  status <- system2(rscript, shQuote(report), env = paste0(names(report_env), "=", shQuote(report_env)), stdout = report_log, stderr = report_log)
  if (status != 0L) stop(paste(tail(readLines(report_log), 50L), collapse = "\n"))
  contrasts <- read.csv(file.path(output_root, "report", "paired_contrasts.csv"))
  stopifnot(all(c("rho_p_primary_holm", "mae_p_family_holm") %in% names(contrasts)),
            identical(sort(unique(contrasts$pattern)), c("all", "an", "nn")))
  cat(corpus, mode, "regression smoke passed: pilot fits, three interpretations, 180/100 roster, training CV, reports and unchanged resumed checkpoints\n")
}

main()
