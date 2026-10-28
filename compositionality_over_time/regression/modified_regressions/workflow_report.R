#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else getwd()
source(file.path(dirname(script_dir), "regression_comparision.R"))
source(file.path(script_dir, "report_grid_filter.R"))

env_words <- function(name, default) {
  value <- trimws(Sys.getenv(name, unset = default))
  if (!nzchar(value)) stop("Empty ", name)
  strsplit(value, "[[:space:]]+")[[1L]]
}
env_integer <- function(name, default, minimum) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, unset = default)))
  if (length(value) != 1L || is.na(value) || value < minimum) stop("Invalid ", name)
  value
}

output_dir <- Sys.getenv("GRID_OUT", unset = "runs/compositionality/regressions/results")
corpus <- Sys.getenv("GRID_CORPUS", unset = "google")
report_dir <- file.path(output_dir, "report")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
taggings <- env_words("GRID_TAGGINGS", "UnTagged Tagged")
train_variant <- Sys.getenv("GRID_TRAIN_VARIANT", unset = "modified")
test_variant <- Sys.getenv("GRID_TEST_VARIANT", unset = "modified")
strata <- do.call(rbind, lapply(taggings, function(tag) {
  data.frame(tag = tag, pattern = if (tag == "Tagged") c("nn", "an") else "all")
}))
modes <- env_words("GRID_MODES", "local5")
vocabs <- as.integer(env_words("GRID_VOCABS", "1000 3000 5000 10000 30000 50000"))
weightings <- env_words("GRID_WEIGHTINGS", "RAW PPMI")
settings <- env_words("GRID_SETTINGS", "Agnostic Pooled Aware")
targets <- env_words("GRID_TARGETS", "compound head modifier")
cutoff <- env_integer("GRID_CUTOFF", "0", 0L)
timespans <- as.integer(strsplit(gsub(",", " ",
                         Sys.getenv("GRID_TIMESPANS", unset = "10000,10,20,50,100")),
                         "[[:space:]]+")[[1L]])
n_boot <- env_integer("GRID_N_BOOT", "2000", 100L)
n_perm <- env_integer("GRID_N_PERM", "5000", 100L)
if (anyNA(vocabs) || anyNA(timespans) || !10000L %in% timespans) {
  stop("Invalid vocabulary or timespan values")
}
reference <- list(
  context_mode = Sys.getenv("GRID_REFERENCE_MODE", unset = modes[[1L]]),
  context_size = env_integer("GRID_REFERENCE_VOCAB", "5000", 0L),
  ppmi = Sys.getenv("GRID_REFERENCE_WEIGHTING", unset = "PPMI"),
  setting = Sys.getenv("GRID_REFERENCE_SETTING", unset = "Agnostic")
)

coverage <- google_v3_validate_results(output_dir)
write.csv(coverage, file.path(report_dir, "configuration_coverage.csv"), row.names = FALSE)

predictions <- read_regression_predictions(".", root = output_dir)
required <- c("corpus", "tag", "ppmi", "setting", "context_mode", "context_size",
              "frequency_family", "productivity_family", "train_variant", "test_variant",
              "timespan", "cutoff",
              "impute", "pattern", "features", "y", "source", "modifier", "head",
              "observed", "predicted_spearman")
if (!all(required %in% names(predictions))) {
  stop("Prediction files are missing columns: ",
       paste(setdiff(required, names(predictions)), collapse = ", "))
}
direct_baseline <- Sys.getenv("GRID_DIRECT_BASELINE", unset = "1") == "1" &&
                   "compound" %in% targets
predictions <- coha_v3_report_grid(predictions)
if (direct_baseline) {
  source(file.path(script_dir, "coha_v3_regression_data.R"))
  input_root <- Sys.getenv("GRID_INPUT_ROOT",
                           unset = "runs/compositionality/regressions/inputs_snapshot")
  direct_rows <- list()
  for (stratum in seq_len(nrow(strata))) {
  tag <- strata$tag[[stratum]]
  pattern <- strata$pattern[[stratum]]
  for (mode in modes) for (vocab in vocabs) for (weighting in weightings) {
    for (setting in settings) {
      filters <- list(y = "compound", features = "all_no_setting",
                      tag = tag, pattern = pattern,
                      train_variant = train_variant, test_variant = test_variant,
                      context_mode = mode, context_size = vocab,
                      ppmi = weighting, setting = setting, timespan = 10000L,
                      cutoff = cutoff, frequency_family = "freq",
                      productivity_family = "prod")
      reference_rows <- filter_prediction_rows(predictions, filters)
      if (nrow(reference_rows) != 100L) {
        stop("Expected 100 held-out predictions for direct baseline in ",
             paste(mode, vocab, weighting, setting, collapse = "/"))
      }
      input <- coha_v3_read_input(input_root, mode, vocab, setting, weighting,
                                  tag, 10000L, cutoff)
      split <- coha_v3_variant_split(input, train_variant, test_variant, tag, pattern)
      test <- split$test
      key <- function(rows) do.call(paste, c(rows[c("source", "modifier", "head")],
                                               sep = "\r"))
      order <- match(key(reference_rows), key(test))
      if (anyNA(order) || nrow(test) != nrow(reference_rows) ||
          !isTRUE(all.equal(reference_rows$observed,
                            test$compositionality[order], tolerance = 1e-10))) {
        stop("Direct baseline and regression held-out rosters differ")
      }
      baseline <- test[["sim_cpf_beta:0"]][order]
      # Match the regression treatment of sparse features: learn the fallback
      # from training data only, without fitting a model to the direct score.
      training_scores <- split$train[["sim_cpf_beta:0"]]
      fallback <- if (any(is.finite(training_scores))) {
        median(training_scores[is.finite(training_scores)])
      } else 0
      baseline[!is.finite(baseline)] <- fallback
      reference_rows$features <- "cordeiro_unfitted"
      reference_rows$predicted_spearman <- baseline
      reference_rows$input_file <- "derived_from_feature_input"
      if ("predictor_count" %in% names(reference_rows)) reference_rows$predictor_count <- 1L
      direct_rows[[length(direct_rows) + 1L]] <- reference_rows
    }
  }
  } # stratum
  predictions <- rbind(predictions, do.call(rbind, direct_rows))
}
config_fields <- required[seq_len(match("y", required))]
config_key <- do.call(paste, c(predictions[config_fields], sep = "\r"))
groups <- split(seq_len(nrow(predictions)), config_key)
model_rows <- lapply(groups, function(index) {
  rows <- predictions[index, , drop = FALSE]
  data.frame(rows[1L, config_fields, drop = FALSE],
             n_compounds = nrow(rows),
             test_spearman = safe_spearman_comparison(rows$observed,
                                                      rows$predicted_spearman),
             test_mae = mean(abs(rows$observed - rows$predicted_spearman)),
             row.names = NULL)
})
model_grid <- do.call(rbind, model_rows)
rownames(model_grid) <- NULL
write.csv(model_grid, file.path(report_dir, "model_grid.csv"), row.names = FALSE)

score_files <- list.files(output_dir, pattern = "^rsquared_.*\\.csv$", full.names = TRUE)
if (!length(score_files)) stop("No training-CV score files found in ", output_dir)
scores <- do.call(rbind, lapply(score_files, function(path) read.csv(path, check.names = FALSE)))
scores <- coha_v3_report_grid(scores)
cv_fields <- c(config_fields, "predictor_count", "CVTrainSpearman",
               "CVTrainSpearmanSD", "CVTrainRsquared", "CVTrainRsquaredSD")
if (!all(cv_fields %in% names(scores))) {
  stop("Score files are missing training-CV columns: ",
       paste(setdiff(cv_fields, names(scores)), collapse = ", "))
}
cv_scores <- scores[cv_fields]
cv_scores$cv_rank_within_target <- ave(
  cv_scores$CVTrainSpearman, interaction(cv_scores$tag, cv_scores$pattern, cv_scores$y),
  FUN = function(value) rank(-value, ties.method = "min", na.last = "keep"))
cv_scores <- cv_scores[order(cv_scores$y, cv_scores$cv_rank_within_target,
                            na.last = TRUE), , drop = FALSE]
write.csv(cv_scores, file.path(report_dir, "training_cv_scores.csv"), row.names = FALSE)

common_base <- list(corpus = corpus, timespan = 10000L,
               cutoff = cutoff, impute = "fold_median",
               train_variant = train_variant, test_variant = test_variant,
               frequency_family = "freq", productivity_family = "prod")
results <- list()
add_comparison <- function(family, target, filters, by = NULL, levels = NULL,
                           features = NULL, conditions = NULL, primary = FALSE) {
  if (!is.null(levels) && length(levels) < 2L) return(invisible(NULL))
  comparison <- compare_regression_predictions(
    ".", y = target, by = by, features = features, conditions = conditions,
    filters = filters, levels = levels, root = output_dir,
    n_boot = n_boot, n_perm = n_perm, prediction_data = predictions)
  rows <- comparison$pairwise
  rows$tag <- filters$tag
  rows$pattern <- filters$pattern
  rows$train_variant <- train_variant
  rows$test_variant <- test_variant
  rows$family <- family
  rows$target <- target
  rows$primary <- primary
  rows$fixed_context_mode <- if (!is.null(filters$context_mode)) filters$context_mode else NA_character_
  rows$fixed_context_size <- if (!is.null(filters$context_size)) filters$context_size else NA_integer_
  rows$fixed_ppmi <- if (!is.null(filters$ppmi)) filters$ppmi else NA_character_
  rows$fixed_setting <- if (!is.null(filters$setting)) filters$setting else NA_character_
  rows$feature_set <- if (is.null(features)) "condition_specific" else features
  results[[length(results) + 1L]] <<- rows
}

# Primary comparisons hold every other grid factor fixed. The same held-out
# compounds are used throughout; per-stratum tests are correlated.
for (stratum in seq_len(nrow(strata))) {
common <- c(common_base, list(tag = strata$tag[[stratum]],
                              pattern = strata$pattern[[stratum]]))
for (target in targets) {
  for (mode in modes) for (vocab in vocabs) for (weighting in weightings) {
    filters <- c(common, list(context_mode = mode, context_size = vocab, ppmi = weighting))
    add_comparison("setting", target, filters, by = "setting", levels = settings,
                   features = "all_no_setting", primary = TRUE)
    add_comparison("representation_setting", target, filters, by = "setting",
                   levels = settings, features = "representation_only")
  }
  for (mode in modes) for (vocab in vocabs) for (setting in settings) {
    filters <- c(common, list(context_mode = mode, context_size = vocab, setting = setting))
    add_comparison("weighting", target, filters, by = "ppmi", levels = weightings,
                   features = "all_no_setting", primary = TRUE)
  }
  for (vocab in vocabs) for (weighting in weightings) for (setting in settings) {
    filters <- c(common, list(context_size = vocab, ppmi = weighting, setting = setting))
    add_comparison("context", target, filters, by = "context_mode", levels = modes,
                   features = "all_no_setting", primary = TRUE)
  }
  for (mode in modes) for (weighting in weightings) for (setting in settings) {
    filters <- c(common, list(context_mode = mode, ppmi = weighting, setting = setting))
    add_comparison("vocabulary", target, filters, by = "context_size", levels = vocabs,
                   features = "all_no_setting", primary = TRUE)
  }

  for (mode in modes) for (vocab in vocabs) for (weighting in weightings) {
    for (setting in settings) {
      filters <- c(common, list(context_mode = mode, context_size = vocab,
                                ppmi = weighting, setting = setting))
      baseline_levels <- c("all_no_setting", "cordeiro", "sim_cpf_beta")
      if (direct_baseline && target == "compound") {
        baseline_levels <- c(baseline_levels, "cordeiro_unfitted")
      }
      add_comparison("feature_baseline", target, filters, by = "features",
                     levels = baseline_levels)
    }
  }
  reference_filters <- c(common, reference)
  for (setting in intersect(settings, c("Pooled", "Aware"))) {
    filters <- c(common, reference[c("context_mode", "context_size", "ppmi")],
                 list(setting = setting))
    add_comparison("setting_feature_ablation", target, filters, by = "features",
                   levels = c("all", "all_no_setting"))
  }

  if (Sys.getenv("GRID_RUN_ENCODINGS", unset = "0") == "1") {
    filters <- reference_filters[setdiff(names(reference_filters),
                                         c("frequency_family", "productivity_family"))]
    add_comparison("encoding", target, filters, conditions = list(
      freq_prod = list(frequency_family = "freq", productivity_family = "prod",
                       features = "all_no_setting"),
      log_freq_prod = list(frequency_family = "log_freq", productivity_family = "prod",
                           features = "all_no_setting"),
      tf_prod = list(frequency_family = "tf", productivity_family = "prod",
                     features = "all_no_setting"),
      freq_family_size = list(frequency_family = "freq",
                              productivity_family = "family_size",
                              features = "all_no_setting")))
  }

  if (Sys.getenv("GRID_RUN_TEMPORAL", unset = "0") == "1") {
    filters <- reference_filters[setdiff(names(reference_filters), "timespan")]
    for (timespan in setdiff(timespans, 10000L)) {
      add_comparison("temporal_resolution", target, filters, conditions = list(
        pooled = list(timespan = 10000L, features = "all_no_setting"),
        binned_static = list(timespan = timespan, features = "all_static")))
      add_comparison("temporal_feature_ablation", target, filters, conditions = list(
        binned_static = list(timespan = timespan, features = "all_static"),
        binned_all = list(timespan = timespan, features = "all")))
    }
  }
}

} # stratum

contrasts <- do.call(rbind, results)
rownames(contrasts) <- NULL
# Correct all primary factor contrasts together per target. The family correction
# covers every stratum and pair for one contrast family, including baselines.
contrasts$rho_p_primary_holm <- NA_real_
contrasts$mae_p_primary_holm <- NA_real_
contrasts$rho_p_family_holm <- NA_real_
contrasts$mae_p_family_holm <- NA_real_
for (target in targets) {
  index <- which(contrasts$primary & contrasts$target == target)
  contrasts$rho_p_primary_holm[index] <- p.adjust(contrasts$rho_p[index], method = "holm")
  contrasts$mae_p_primary_holm[index] <- p.adjust(contrasts$mae_p[index], method = "holm")
  for (family in unique(contrasts$family[contrasts$target == target])) {
    index <- which(contrasts$target == target & contrasts$family == family)
    contrasts$rho_p_family_holm[index] <- p.adjust(contrasts$rho_p[index], method = "holm")
    contrasts$mae_p_family_holm[index] <- p.adjust(contrasts$mae_p[index], method = "holm")
  }
}
write.csv(contrasts, file.path(report_dir, "paired_contrasts.csv"), row.names = FALSE)

primary <- contrasts[contrasts$primary, , drop = FALSE]
summary_rows <- lapply(split(primary, list(primary$tag, primary$pattern, primary$target, primary$family), drop = TRUE),
                       function(rows) {
  data.frame(tag = rows$tag[[1L]], pattern = rows$pattern[[1L]],
             train_variant = train_variant, test_variant = test_variant,
             target = rows$target[[1L]], factor = rows$family[[1L]],
             n_contrasts = nrow(rows),
             n_rho_holm_below_05 = sum(rows$rho_p_primary_holm < 0.05, na.rm = TRUE),
             median_absolute_delta_spearman = median(abs(rows$delta_spearman), na.rm = TRUE),
             row.names = NULL)
})
factor_summary <- do.call(rbind, summary_rows)
rownames(factor_summary) <- NULL
write.csv(factor_summary, file.path(report_dir, "factor_summary.csv"), row.names = FALSE)
cat("Wrote ", nrow(model_grid), " model rows and ", nrow(contrasts),
    " paired contrasts to ", report_dir, "\n", sep = "")
