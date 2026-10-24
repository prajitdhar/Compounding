# Retain the requested grid when an output directory contains older fits.
coha_v3_report_grid <- function(frame) {
  fields <- c(context_mode = "GRID_MODES", context_size = "GRID_VOCABS",
              ppmi = "GRID_WEIGHTINGS", setting = "GRID_SETTINGS", tag = "GRID_TAGGINGS",
              timespan = "GRID_TIMESPANS", cutoff = "GRID_CUTOFFS", y = "GRID_TARGETS",
              train_variant = "GRID_TRAIN_VARIANT", test_variant = "GRID_TEST_VARIANT")
  keep <- rep(TRUE, nrow(frame))
  for (field in names(fields)) {
    value <- trimws(Sys.getenv(fields[[field]], unset = ""))
    if (nzchar(value)) {
      if (!field %in% names(frame)) stop("Missing report grid column: ", field)
      allowed <- strsplit(value, "[,[:space:]]+")[[1L]]
      keep <- keep & as.character(frame[[field]]) %in% allowed
    }
  }
  frame[keep, , drop = FALSE]
}

# A successful array dependency does not imply that every scientific cell fitted:
# the fitter can legitimately skip individual recipes lacking variable predictors.
# Require all requested cell receipts before presenting a complete-grid report,
# while recording those recipe-level skips rather than inventing model results.
google_v3_validate_results <- function(directory, input_root = Sys.getenv(
    "GRID_INPUT_ROOT", unset = "runs/compositionality/regressions/inputs_snapshot")) {
  words <- function(name, default) {
    value <- trimws(Sys.getenv(name, unset = default))
    if (!nzchar(value)) stop("Empty report grid: ", name)
    unique(strsplit(value, "[,[:space:]]+")[[1L]])
  }
  modes <- words("GRID_MODES", "local5")
  vocabs <- words("GRID_VOCABS", "1000 3000 5000 10000 30000 50000")
  weights <- words("GRID_WEIGHTINGS", "RAW PPMI")
  settings <- words("GRID_SETTINGS", "Agnostic Pooled Aware")
  tags <- words("GRID_TAGGINGS", "UnTagged Tagged")
  widths <- words("GRID_TIMESPANS", "10000 10 20 50 100")
  cutoffs <- words("GRID_CUTOFFS", "0 10 50 100 500 1000")
  targets <- words("GRID_TARGETS", "compound head modifier")
  train_variant <- Sys.getenv("GRID_TRAIN_VARIANT", unset = "modified")
  test_variant <- Sys.getenv("GRID_TEST_VARIANT", unset = "modified")
  if (any(modes != "local5")) stop("Google reports support only local5 context")
  script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
  # Normally called by the fitter's companion report/statistics scripts.
  if (!file.exists(file.path(script_dir, "regression_exps.r"))) {
    script_dir <- file.path(script_dir, "..", "modified_regressions")
  }
  code_md5 <- unname(tools::md5sum(file.path(script_dir,
    c("regression_exps.r", "coha_v3_regression_data.R"))))
  if (anyNA(code_md5)) stop("Cannot identify Google regression source files")
  input_hashes <- new.env(parent = emptyenv())
  input_md5 <- function(paths) {
    vapply(paths, function(path) {
      if (!exists(path, envir = input_hashes, inherits = FALSE)) {
        assign(path, unname(tools::md5sum(path)), envir = input_hashes)
      }
      get(path, envir = input_hashes, inherits = FALSE)
    }, character(1), USE.NAMES = FALSE)
  }
  rows <- list()
  problems <- character()
  tuning_fields <- c("folds", "repeats", "alpha_count", "lambda_count", "permutations")
  tuning_environment <- c(folds = "GRID_FOLDS", repeats = "GRID_REPEATS", alpha_count = "GRID_ALPHA_COUNT",
    lambda_count = "GRID_LAMBDA_COUNT", permutations = "GRID_PERMUTATIONS")
  reference_tuning <- NULL
  output_types <- c("rsquared", "coefimp", "permimp", "familyimp", "predictions", "cvgrid", "cvfolds")
  for (tag in tags) for (mode in modes) for (vocab in vocabs) for (weight in weights) {
    for (setting in settings) for (width in widths) for (cutoff in cutoffs) {
      for (pattern in if (tag == "Tagged") c("nn", "an") else "all") {
        suffix <- sprintf("google_v3_%s_ctx%s_%s_%s_%s_%s_%s_%s_freq_prod",
          mode, vocab, tag, weight, setting, width, cutoff, pattern)
        if (train_variant != "original" || test_variant != "original") {
          suffix <- paste0(suffix, "_train", train_variant, "_test", test_variant)
        }
        receipt <- file.path(directory, paste0("complete_", suffix, ".rds"))
        paths <- file.path(directory, paste0(output_types, "_", suffix, ".csv"))
        if (!file.exists(receipt) || !all(file.exists(paths))) {
          problems <- c(problems, paste0(suffix, " (missing completion receipt or result CSV)"))
          next
        }
        config <- readRDS(receipt)
        inputs <- file.path(input_root, mode, sprintf(
          "features_Compound%s_withSetting_%s_%s_ctx%s_%s_%s_na.csv",
          setting, weight, tag, vocab, width, cutoff))
        if (width != "10000") inputs <- c(inputs, file.path(input_root, mode, sprintf(
          "temporal_Compound%s_withSetting_%s_%s_ctx%s_%s_%s_na.csv",
          setting, weight, tag, vocab, width, cutoff)))
        if (!all(file.exists(inputs)) || !identical(config$input_md5, input_md5(inputs)) ||
            !identical(config$code_md5, code_md5) || !all(targets %in% config$targets) ||
            !identical(config$train_variant, train_variant) ||
            !identical(config$test_variant, test_variant)) {
          problems <- c(problems, paste0(suffix, " (input/source identity or target/variant mismatch)"))
          next
        }
        if (!all(c(tuning_fields, "frequency_family", "productivity_family") %in% names(config)) ||
            anyNA(unlist(config[tuning_fields])) ||
            !identical(config$frequency_family, "freq") || !identical(config$productivity_family, "prod")) {
          problems <- c(problems, paste0(suffix, " (missing tuning identity or wrong predictor encoding)"))
          next
        }
        tuning <- c(lapply(config[tuning_fields], as.integer),
          config[c("frequency_family", "productivity_family")])
        mismatch <- FALSE
        for (field in tuning_fields) {
          expected <- trimws(Sys.getenv(tuning_environment[[field]], unset = ""))
          if (nzchar(expected)) {
            expected <- suppressWarnings(as.integer(expected))
            if (is.na(expected) || !identical(tuning[[field]], expected)) mismatch <- TRUE
          }
        }
        if (mismatch || (!is.null(reference_tuning) && !identical(tuning, reference_tuning))) {
          problems <- c(problems, paste0(suffix, " (mixed or unexpected CV/tuning/importance settings)"))
          next
        }
        if (is.null(reference_tuning)) reference_tuning <- tuning
        scores <- read.csv(paths[[1L]], check.names = FALSE)
        expected_cell <- list(corpus = "google", context_mode = mode, context_size = as.integer(vocab),
          tag = tag, ppmi = weight, setting = setting, timespan = as.integer(width), cutoff = as.integer(cutoff),
          pattern = pattern, train_variant = train_variant, test_variant = test_variant,
          frequency_family = "freq", productivity_family = "prod", impute = "fold_median")
        required <- c(names(expected_cell), "features", "y", "train_dim", "test_dim")
        if (!all(required %in% names(scores)) || !nrow(scores) ||
            anyNA(scores[required]) || any(scores$train_dim != 180L) || any(scores$test_dim != 100L) ||
            !all(scores$y %in% config$targets) || !all(scores$features %in% config$feature_sets) ||
            anyDuplicated(paste(scores$features, scores$y, sep = "\r")) ||
            !all(vapply(names(expected_cell), function(field) {
              all(as.character(scores[[field]]) == as.character(expected_cell[[field]]))
            }, logical(1)))) {
          problems <- c(problems, paste0(suffix, " (invalid/duplicate model results, cell metadata or rating roster)"))
          next
        }
        recipes <- expand.grid(features = config$feature_sets, y = targets, stringsAsFactors = FALSE)
        observed <- paste(scores$features, scores$y, sep = " -> ")
        skipped <- setdiff(paste(recipes$features, recipes$y, sep = " -> "), observed)
        rows[[length(rows) + 1L]] <- data.frame(context_mode = mode, context_size = as.integer(vocab),
          tag = tag, ppmi = weight, setting = setting, timespan = as.integer(width), cutoff = as.integer(cutoff),
          pattern = pattern, train_variant = train_variant, test_variant = test_variant,
          folds = tuning$folds, repeats = tuning$repeats, alpha_count = tuning$alpha_count,
          lambda_count = tuning$lambda_count, permutations = tuning$permutations,
          frequency_family = tuning$frequency_family, productivity_family = tuning$productivity_family,
          n_fitted_models = sum(scores$y %in% targets), n_skipped_recipes = length(skipped),
          skipped_recipes = paste(skipped, collapse = "; "), receipt = basename(receipt),
          stringsAsFactors = FALSE)
      }
    }
  }
  if (length(problems)) stop("Requested Google regression grid is incomplete or incompatible (",
    length(problems), " cells):\n", paste(head(problems, 20L), collapse = "\n"),
    if (length(problems) > 20L) "\n..." else "",
    "\nFinish/resume those cells, or explicitly narrow the report grid. A cell with no usable models has no completion receipt.")
  result <- do.call(rbind, rows)
  if (sum(result$n_skipped_recipes)) message("Some requested recipes lacked usable predictors; inspect configuration_coverage.csv")
  result
}
