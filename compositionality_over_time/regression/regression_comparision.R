# Source this file in RStudio. It does not run any comparison automatically.
# Using these held-out results to select a winner makes later tests exploratory.

read_regression_predictions <- function(experiment,
                                        root = "runs/compositionality/regressions/results") {
  directory <- file.path(root, experiment)
  files <- list.files(directory, pattern = "^predictions_.*\\.csv$", full.names = TRUE)
  if (!length(files)) stop("No prediction CSVs found in: ", directory)
  frames <- lapply(files, function(path) {
    frame <- read.csv(path, check.names = FALSE)
    frame$input_file <- basename(path)
    frame
  })
  do.call(rbind, frames)
}

filter_prediction_rows <- function(data, filters) {
  if (!length(filters)) return(data)
  if (is.null(names(filters)) || any(!nzchar(names(filters)))) {
    stop("Filters must be a named list")
  }
  for (name in names(filters)) {
    if (!name %in% names(data)) stop("Unknown filter column: ", name)
    data <- data[!is.na(data[[name]]) & data[[name]] %in% filters[[name]], , drop = FALSE]
  }
  data
}

safe_spearman_comparison <- function(observed, predicted) {
  if (length(observed) < 3L || sd(observed) == 0 || sd(predicted) == 0) return(NA_real_)
  suppressWarnings(cor(observed, predicted, method = "spearman"))
}

compare_regression_predictions <- function(experiment,
                                           y = "compound",
                                           by = NULL,
                                           features = NULL,
                                           conditions = NULL,
                                           filters = list(),
                                           levels = NULL,
                                           prediction_col = "predicted_spearman",
                                           root = "runs/compositionality/regressions/results",
                                           n_boot = 2000L,
                                           n_perm = 2000L,
                                           seed = 1111L,
                                           prediction_data = NULL,
                                           pairing_columns = c("source", "modifier", "head")) {
  if (length(y) != 1L || !y %in% c("compound", "head", "modifier")) stop("Invalid target")
  if (!prediction_col %in% c("predicted_spearman", "predicted_r2")) stop("Invalid prediction column")
  if (length(n_boot) != 1L || is.na(n_boot) || n_boot < 100L ||
      length(n_perm) != 1L || is.na(n_perm) || n_perm < 100L) {
    stop("n_boot and n_perm must each be at least 100")
  }
  if (is.null(by) == is.null(conditions)) stop("Supply exactly one of by or conditions")
  data <- if (is.null(prediction_data)) read_regression_predictions(experiment, root) else prediction_data
  required <- c("y", "features", "source", "modifier", "head", "observed", prediction_col)
  if (!all(required %in% names(data))) stop("Prediction files are missing required columns")
  if (!length(pairing_columns) || anyDuplicated(pairing_columns) ||
      !all(pairing_columns %in% names(data))) stop("Invalid pairing columns")
  common <- c(list(y = y), if (!is.null(features)) list(features = features), filters)
  if (anyDuplicated(names(common))) stop("Do not repeat y/features in filters")
  data <- filter_prediction_rows(data, common)
  if (!nrow(data)) stop("No rows match the requested target, features, and filters")

  if (!is.null(by)) {
    if (length(by) != 1L || !by %in% names(data)) stop("Unknown comparison column")
    if (by %in% names(common)) stop("The comparison column is also fixed by a filter")
    values <- if (is.null(levels)) sort(unique(data[[by]])) else levels
    if (anyDuplicated(as.character(values))) stop("Duplicate comparison levels")
    conditions <- setNames(lapply(values, function(value) setNames(list(value), by)),
                           as.character(values))
  }
  if (!is.list(conditions) || length(conditions) < 2L ||
      is.null(names(conditions)) || any(!nzchar(names(conditions))) ||
      anyDuplicated(names(conditions))) {
    stop("At least two uniquely named conditions are required")
  }

  config_columns <- intersect(c("corpus", "tag", "ppmi", "setting", "context_mode",
    "context_size", "frequency_family", "productivity_family", "timespan", "cutoff",
    "impute", "pattern", "features", "y", "train_variant", "test_variant"), names(data))
  key_columns <- pairing_columns
  selected <- lapply(names(conditions), function(label) {
    rows <- filter_prediction_rows(data, conditions[[label]])
    if (!nrow(rows)) stop("No predictions for condition: ", label)
    if (nrow(unique(rows[config_columns])) != 1L) {
      stop("Condition ", label, " matches multiple configurations; add filters")
    }
    key <- do.call(paste, c(rows[key_columns], sep = "\r"))
    if (anyDuplicated(key)) stop("Duplicate compound in condition: ", label)
    rows[order(key), , drop = FALSE]
  })
  names(selected) <- names(conditions)
  reference <- selected[[1L]]
  reference_key <- do.call(paste, c(reference[key_columns], sep = "\r"))
  for (label in names(selected)[-1L]) {
    rows <- selected[[label]]
    key <- do.call(paste, c(rows[key_columns], sep = "\r"))
    if (!identical(key, reference_key)) {
      stop("Different held-out compound rosters for ", label, "; comparison would be unpaired")
    }
    if (!isTRUE(all.equal(rows$observed, reference$observed, tolerance = 1e-10))) {
      stop("Observed ratings differ for condition: ", label)
    }
  }
  observed <- reference$observed
  predicted <- do.call(cbind, lapply(selected, function(rows) rows[[prediction_col]]))
  colnames(predicted) <- names(selected)
  if (nrow(predicted) < 3L || any(!is.finite(observed)) || any(!is.finite(predicted))) {
    stop("At least three finite, paired predictions are required")
  }
  scores <- vapply(seq_len(ncol(predicted)), function(j) {
    safe_spearman_comparison(observed, predicted[, j])
  }, numeric(1))
  undefined_spearman <- names(selected)[is.na(scores)]
  if (length(undefined_spearman)) {
    warning("Spearman is undefined for condition(s) with constant predictions: ",
            paste(undefined_spearman, collapse = ", "),
            ". Their Spearman contrasts will be NA; MAE remains comparable.")
  }
  errors <- abs(sweep(predicted, 1L, observed, "-"))
  maes <- colMeans(errors)
  model_summary <- data.frame(condition = names(selected), n_compounds = length(observed),
                              spearman = scores, mae = maes, row.names = NULL)
  configurations <- do.call(rbind, lapply(names(selected), function(label) {
    data.frame(condition = label, selected[[label]][1L, config_columns, drop = FALSE],
               row.names = NULL)
  }))

  set.seed(seed)
  bootstrap_rows <- replicate(n_boot, sample.int(length(observed), replace = TRUE), simplify = FALSE)
  bootstrap_spearman <- vapply(bootstrap_rows, function(i) {
    vapply(seq_len(ncol(predicted)), function(j) {
      safe_spearman_comparison(observed[i], predicted[i, j])
    }, numeric(1))
  }, numeric(ncol(predicted)))
  bootstrap_mae <- vapply(bootstrap_rows, function(i) colMeans(errors[i, , drop = FALSE]),
                          numeric(ncol(predicted)))

  pairs <- combn(seq_along(selected), 2L)
  pairwise <- lapply(seq_len(ncol(pairs)), function(column) {
    a <- pairs[1L, column]
    b <- pairs[2L, column]
    rho_estimable <- !is.na(scores[a]) && !is.na(scores[b])
    delta_spearman <- if (rho_estimable) scores[a] - scores[b] else NA_real_
    delta_mae <- maes[a] - maes[b]
    rho_ci <- if (rho_estimable) {
      quantile(bootstrap_spearman[a, ] - bootstrap_spearman[b, ],
               c(0.025, 0.975), na.rm = TRUE, names = FALSE)
    } else c(NA_real_, NA_real_)
    mae_ci <- quantile(bootstrap_mae[a, ] - bootstrap_mae[b, ],
                       c(0.025, 0.975), na.rm = TRUE, names = FALSE)
    perm_rho <- if (rho_estimable) numeric(n_perm) else NULL
    perm_mae <- numeric(n_perm)
    for (iteration in seq_len(n_perm)) {
      swap <- sample(c(FALSE, TRUE), length(observed), replace = TRUE)
      left <- predicted[, a]
      right <- predicted[, b]
      left[swap] <- predicted[swap, b]
      right[swap] <- predicted[swap, a]
      if (rho_estimable) {
        perm_rho[iteration] <- safe_spearman_comparison(observed, left) -
                               safe_spearman_comparison(observed, right)
      }
      perm_mae[iteration] <- mean(ifelse(swap, errors[, b] - errors[, a],
                                         errors[, a] - errors[, b]))
    }
    valid <- !is.na(perm_rho)
    if (rho_estimable && !any(valid)) {
      stop("Permutation Spearman was undefined for ", names(selected)[a],
           " vs ", names(selected)[b])
    }
    data.frame(left = names(selected)[a], right = names(selected)[b],
               delta_spearman = delta_spearman,
               rho_ci_low = rho_ci[1L], rho_ci_high = rho_ci[2L],
               rho_p = if (rho_estimable) {
                 (1 + sum(abs(perm_rho[valid]) >= abs(delta_spearman))) / (1 + sum(valid))
               } else NA_real_,
               delta_mae = delta_mae, mae_ci_low = mae_ci[1L], mae_ci_high = mae_ci[2L],
               mae_p = (1 + sum(abs(perm_mae) >= abs(delta_mae))) / (n_perm + 1))
  })
  pairwise <- do.call(rbind, pairwise)
  pairwise$rho_p_holm <- p.adjust(pairwise$rho_p, method = "holm")
  pairwise$mae_p_holm <- p.adjust(pairwise$mae_p, method = "holm")
  rownames(pairwise) <- NULL
  global_error_p <- if (ncol(errors) > 2L) stats::friedman.test(errors)$p.value else NA_real_
  structure(list(models = model_summary, pairwise = pairwise,
                 configurations = configurations,
                 friedman_mae_p = global_error_p,
                 undefined_spearman = undefined_spearman,
                 paired_predictions = data.frame(reference[key_columns], observed,
                                                 predicted, check.names = FALSE)),
            class = "regression_comparison")
}

print.regression_comparison <- function(x, ...) {
  cat("Model performance on identical held-out compounds:\n")
  print(x$models, row.names = FALSE, digits = 4)
  if (length(x$undefined_spearman)) {
    cat("Spearman undefined (constant predictions): ",
        paste(x$undefined_spearman, collapse = ", "), "\n", sep = "")
  }
  if (!is.na(x$friedman_mae_p)) {
    cat("Global Friedman test of absolute errors: p = ",
        format.pval(x$friedman_mae_p), "\n", sep = "")
  }
  cat("Pairwise differences (left minus right; Holm adjustment within this comparison):\n")
  print(x$pairwise, row.names = FALSE, digits = 4)
  cat("Permutation p-values assume exchangeable model labels per compound; ",
      "Holm correction covers only the pairs shown.\n", sep = "")
  invisible(x)
}

# Example after source("regression_comparision.R"):
# compare_regression_predictions(".", by = "ppmi", features = "all_no_setting",
#   levels = c("RAW", "PPMI"),
#   filters = list(setting = "Agnostic", context_mode = "local5",
#                  context_size = 5000, timespan = 10000, cutoff = 0,
#                  frequency_family = "freq", productivity_family = "prod"))
