#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(optparse))
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1]]))) else getwd()
source(file.path(script_dir, "coha_v3_regression_data.R"))

options(scipen = 999)
glmnet_padding_column <- ".glmnet_padding"

option_list <- list(
  make_option("--input-root", dest = "input_root", default = "runs/compositionality/regressions/inputs_snapshot"),
  make_option("--output-dir", dest = "output_dir", default = "runs/compositionality/regressions/results"),
  make_option("--context-mode", dest = "context_mode", default = "local5"),
  make_option("--context-size", dest = "context_size", type = "integer", default = 10000L),
  make_option("--setting", default = "Agnostic"),
  make_option("--ppmi", default = "PPMI"),
  make_option("--tagging", default = "UnTagged"),
  make_option("--timespans", default = "10000,10,20,50,100"),
  make_option("--cutoffs", default = "0,10,50,100,500,1000"),
  make_option("--targets", default = "compound,modifier,head"),
  make_option("--frequency-family", dest = "frequency_family", default = "freq"),
  make_option("--productivity-family", dest = "productivity_family", default = "prod"),
  make_option("--train-variant", dest = "train_variant", default = "modified"),
  make_option("--test-variant", dest = "test_variant", default = "modified"),
  make_option("--feature-sets", dest = "feature_sets", default = paste(c("all", "all_static", "sim_cpf_beta", "cordeiro",
      "all_no_setting", "representation_only", "cosine_sim", "with_setting", "info_theory", "association_counts", "freq", "tf",
      "log_freq", "family_size", "prod", "temporal"), collapse = ",")),
  make_option("--cores", type = "integer", default = 1L),
  make_option("--folds", type = "integer", default = 5L),
  make_option("--repeats", type = "integer", default = 10L),
  make_option("--alpha-count", dest = "alpha_count", type = "integer", default = 10L),
  make_option("--lambda-count", dest = "lambda_count", type = "integer", default = 200L),
  make_option("--permutations", type = "integer", default = 5L),
  make_option("--pilot", action = "store_true", default = FALSE),
  make_option("--validate-only", dest = "validate_only", action = "store_true", default = FALSE),
  make_option("--force", action = "store_true", default = FALSE)
)

parse_choices <- function(value, allowed, label) {
  result <- unique(trimws(strsplit(value, ",", fixed = TRUE)[[1]]))
  if (!length(result) || any(!result %in% allowed)) stop("Invalid ", label, ": ", value)
  result
}

safe_spearman <- function(pred, obs) {
  if (length(pred) < 3L || any(!is.finite(pred)) || any(!is.finite(obs)) ||
      sd(pred) == 0 || sd(obs) == 0) return(NA_real_)
  suppressWarnings(cor(pred, obs, method = "spearman"))
}

safe_r2 <- function(pred, obs) {
  if (length(pred) < 3L || any(!is.finite(pred)) || any(!is.finite(obs)) ||
      sd(pred) == 0 || sd(obs) == 0) return(NA_real_)
  cor(pred, obs)^2
}

safe_adjusted_r2 <- function(pred, obs, p) {
  r2 <- safe_r2(pred, obs)
  n <- length(obs)
  if (is.na(r2) || n <= p + 1L) return(NA_real_)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

caret_spearman <- function(data, lev = NULL, model = NULL) {
  # caret serializes this callback to PSOCK workers; keep it self-contained.
  pred <- data$pred
  obs <- data$obs
  if (length(pred) < 3L || any(!is.finite(pred)) || any(!is.finite(obs)) ||
      sd(pred) == 0 || sd(obs) == 0) return(c(Spearman = NA_real_))
  c(Spearman = suppressWarnings(stats::cor(pred, obs, method = "spearman")))
}

usable_columns <- function(df) {
  names(df)[vapply(df, function(x) sum(!is.na(x)) >= 10L && is.finite(sd(x, na.rm = TRUE)) &&
                     sd(x, na.rm = TRUE) > 0, logical(1))]
}

metadata_frame <- function(metadata, n) {
  as.data.frame(lapply(metadata, rep, n), stringsAsFactors = FALSE)
}

cv_results <- function(model, objective, metadata) {
  results <- model$results
  best <- model$bestTune
  selected <- results$alpha == best$alpha & results$lambda == best$lambda
  if (sum(selected) != 1L) stop("Cannot identify selected ", objective, " tuning row")
  sd_column <- paste0(objective, "SD")
  if (!all(c(objective, sd_column) %in% names(results))) {
    stop("Missing ", objective, " training-CV scores")
  }
  grid <- data.frame(metadata_frame(metadata, nrow(results)),
                     cv_objective = objective, alpha = results$alpha, lambda = results$lambda,
                     mean_cv_score = results[[objective]], sd_cv_score = results[[sd_column]],
                     selected = selected, check.names = FALSE)
  folds <- model$resample
  if (is.null(folds) || !all(c("Resample", objective) %in% names(folds))) {
    stop("Missing per-fold ", objective, " training-CV scores")
  }
  if ("alpha" %in% names(folds) && any(folds$alpha != best$alpha)) {
    stop("Unexpected alpha in selected ", objective, " folds")
  }
  if ("lambda" %in% names(folds) && any(folds$lambda != best$lambda)) {
    stop("Unexpected lambda in selected ", objective, " folds")
  }
  fold_scores <- data.frame(metadata_frame(metadata, nrow(folds)),
                            cv_objective = objective, resample = folds$Resample,
                            alpha = best$alpha, lambda = best$lambda,
                            cv_score = folds[[objective]], check.names = FALSE)
  list(mean = results[[objective]][selected], sd = results[[sd_column]][selected],
       alpha = best$alpha, lambda = best$lambda, grid = grid, folds = fold_scores)
}

coefficient_importance <- function(model, metadata) {
  coefficients <- as.matrix(coef(model$finalModel, s = model$bestTune$lambda))
  features <- rownames(coefficients)[-1L]
  values <- abs(as.numeric(coefficients[-1L, 1L]))
  real <- features != glmnet_padding_column
  features <- features[real]
  values <- values[real]
  values[!is.finite(values)] <- 0
  total <- sum(values)
  data.frame(metadata_frame(metadata, length(values)), feature = features,
             feature_family = vapply(features, coha_v3_feature_family, character(1)),
             coefficient_importance = values,
             coefficient_importance_share = if (total > 0) values / total else 0,
             coefficient_rank = rank(-values, ties.method = "min"), check.names = FALSE)
}

permutation_importance <- function(r2_model, spearman_model, test_x, test_y, baseline_r2,
                                   baseline_spearman, metadata, repeats) {
  if (repeats == 0L) return(data.frame())
  set.seed(1111)
  rows <- lapply(setdiff(names(test_x), glmnet_padding_column), function(feature) {
    r2_scores <- spearman_scores <- rep(NA_real_, repeats)
    for (i in seq_len(repeats)) {
      altered <- test_x
      altered[[feature]] <- sample(altered[[feature]])
      r2_scores[i] <- safe_r2(predict(r2_model, newdata = altered), test_y)
      spearman_scores[i] <- safe_spearman(predict(spearman_model, newdata = altered), test_y)
    }
    data.frame(feature = feature, feature_family = coha_v3_feature_family(feature),
               mean_permuted_r2 = mean(r2_scores), mean_permuted_spearman = mean(spearman_scores),
               delta_test_r2 = baseline_r2 - mean(r2_scores),
               delta_test_spearman = baseline_spearman - mean(spearman_scores),
               sd_permuted_r2 = sd(r2_scores), sd_permuted_spearman = sd(spearman_scores))
  })
  result <- do.call(rbind, rows)
  result$permutation_rank <- rank(-result$delta_test_r2, ties.method = "min", na.last = "keep")
  cbind(metadata_frame(metadata, nrow(result)), result)
}

family_importance <- function(r2_model, spearman_model, coef_table, test_x, test_y,
                              baseline_r2, baseline_spearman, metadata, repeats) {
  if (repeats == 0L) return(data.frame())
  real_features <- setdiff(names(test_x), glmnet_padding_column)
  families <- split(real_features, vapply(real_features, coha_v3_feature_family, character(1)))
  set.seed(1111)
  rows <- lapply(names(families), function(family) {
    columns <- families[[family]]
    r2_scores <- spearman_scores <- rep(NA_real_, repeats)
    for (i in seq_len(repeats)) {
      altered <- test_x
      # A shared row permutation preserves relationships within a feature family.
      altered[columns] <- altered[sample(seq_len(nrow(altered))), columns, drop = FALSE]
      r2_scores[i] <- safe_r2(predict(r2_model, newdata = altered), test_y)
      spearman_scores[i] <- safe_spearman(predict(spearman_model, newdata = altered), test_y)
    }
    share <- sum(coef_table$coefficient_importance_share[coef_table$feature_family == family])
    data.frame(feature_family = family, feature_count = length(columns),
               total_coefficient_importance_share = share,
               mean_delta_test_r2 = baseline_r2 - mean(r2_scores),
               mean_delta_test_spearman = baseline_spearman - mean(spearman_scores),
               max_delta_test_r2 = max(baseline_r2 - r2_scores),
               max_delta_test_spearman = max(baseline_spearman - spearman_scores))
  })
  result <- do.call(rbind, rows)
  cbind(metadata_frame(metadata, nrow(result)), result)
}

fit_one <- function(train, test, feature_set, target, args, timespan, cutoff, pattern, grid) {
  raw_train <- coha_v3_select_features(train, feature_set, args$frequency_family, args$productivity_family)
  raw_test <- coha_v3_select_features(test, feature_set, args$frequency_family, args$productivity_family)
  columns <- usable_columns(raw_train)
  if (!length(columns)) {
    message("Skipping ", feature_set, " (no variable predictors)")
    return(NULL)
  }
  train_x <- raw_train[columns]
  test_x <- raw_test[columns]
  if (length(columns) == 1L) {
    # glmnet requires two columns; an all-zero column carries no information.
    train_x[[glmnet_padding_column]] <- 0
    test_x[[glmnet_padding_column]] <- 0
  }
  target_col <- c(compound = "compositionality", modifier = "avgModifier", head = "avgHead")[[target]]
  train_y <- train[[target_col]]
  test_y <- test[[target_col]]
  if (anyNA(train_y) || anyNA(test_y)) stop("Missing response for ", target)
  metadata <- list(corpus = "google", tag = args$tagging, ppmi = args$ppmi, setting = args$setting,
                   context_mode = args$context_mode, context_size = args$context_size,
                   train_variant = args$train_variant, test_variant = args$test_variant,
                   frequency_family = args$frequency_family,
                   productivity_family = args$productivity_family,
                   timespan = timespan, cutoff = cutoff, impute = "fold_median", pattern = pattern,
                   features = feature_set, y = target, train_dim = nrow(train_x), test_dim = nrow(test_x),
                   predictor_count = length(columns))

  set.seed(1111)
  folds <- caret::createMultiFolds(train_y, k = args$folds, times = args$repeats)
  cv_common <- list(method = "repeatedcv", number = args$folds, repeats = args$repeats,
                    index = folds, returnResamp = "final", allowParallel = args$cores > 1L)
  r2_control <- do.call(caret::trainControl, cv_common)
  spearman_control <- do.call(caret::trainControl, c(cv_common, list(summaryFunction = caret_spearman)))
  # glmnet standardizes the real predictor internally; avoid scaling zero padding.
  preprocess <- if (length(columns) == 1L) c("medianImpute", "center") else
    c("medianImpute", "center", "scale")
  set.seed(1111)
  r2_model <- caret::train(x = train_x, y = train_y, method = "glmnet", metric = "Rsquared",
                           trControl = r2_control, tuneGrid = grid, preProcess = preprocess)
  set.seed(1111)
  spearman_model <- caret::train(x = train_x, y = train_y, method = "glmnet", metric = "Spearman",
                                 trControl = spearman_control, tuneGrid = grid, preProcess = preprocess)
  r2_cv <- cv_results(r2_model, "Rsquared", metadata)
  spearman_cv <- cv_results(spearman_model, "Spearman", metadata)

  r2_train_pred <- predict(r2_model, newdata = train_x)
  r2_test_pred <- predict(r2_model, newdata = test_x)
  spearman_train_pred <- predict(spearman_model, newdata = train_x)
  spearman_test_pred <- predict(spearman_model, newdata = test_x)
  if (any(!is.finite(r2_test_pred)) || any(!is.finite(spearman_test_pred))) {
    stop("Non-finite held-out predictions for ", feature_set, "/", target)
  }
  best_r2 <- r2_model$bestTune$lambda
  p_r2 <- sum(as.matrix(coef(r2_model$finalModel, s = best_r2))[-1L, 1L] != 0)
  best_spearman <- spearman_model$bestTune$lambda
  p_spearman <- sum(as.matrix(coef(spearman_model$finalModel, s = best_spearman))[-1L, 1L] != 0)
  test_r2 <- safe_r2(r2_test_pred, test_y)
  test_spearman <- safe_spearman(spearman_test_pred, test_y)
  performance <- data.frame(metadata_frame(metadata, 1L), num_vars_rsquared = p_r2,
                            num_vars_adjrsquared = p_r2, num_vars_spearman = p_spearman,
                            CVTrainRsquared = r2_cv$mean, CVTrainRsquaredSD = r2_cv$sd,
                            CVTrainSpearman = spearman_cv$mean,
                            CVTrainSpearmanSD = spearman_cv$sd,
                            best_alpha_r2 = r2_cv$alpha, best_lambda_r2 = r2_cv$lambda,
                            best_alpha_spearman = spearman_cv$alpha,
                            best_lambda_spearman = spearman_cv$lambda,
                            TrainRsquared = safe_r2(r2_train_pred, train_y),
                            TrainSpearman = safe_spearman(spearman_train_pred, train_y),
                            TrainAdjRsquared = safe_adjusted_r2(r2_train_pred, train_y, p_r2),
                            TestRsquared = test_r2, TestSpearman = test_spearman,
                            TestAdjRsquared = safe_adjusted_r2(r2_test_pred, test_y, p_r2))
  coefficients <- coefficient_importance(spearman_model, metadata)
  permutations <- permutation_importance(r2_model, spearman_model, test_x, test_y,
                                         test_r2, test_spearman, metadata, args$permutations)
  families <- family_importance(r2_model, spearman_model, coefficients, test_x, test_y,
                                test_r2, test_spearman, metadata, args$permutations)
  predictions <- data.frame(metadata_frame(metadata, nrow(test)), compound = test$compound,
                            canonical_compound = if ("canonical_compound" %in% names(test))
                              test$canonical_compound else test$compound,
                            modifier = test$modifier, head = test$head, source = test$source,
                            observed = test_y, predicted_r2 = as.numeric(r2_test_pred),
                            predicted_spearman = as.numeric(spearman_test_pred))
  list(rsquared = performance, coefimp = coefficients, permimp = permutations,
       familyimp = families, predictions = predictions,
       cvgrid = rbind(r2_cv$grid, spearman_cv$grid),
       cvfolds = rbind(r2_cv$folds, spearman_cv$folds))
}

atomic_csv <- function(df, path) {
  temp <- tempfile(tmpdir = dirname(path), fileext = ".csv")
  on.exit(unlink(temp), add = TRUE)
  write.csv(df, temp, row.names = FALSE, na = "")
  if (!file.rename(temp, path)) stop("Cannot save ", path)
}

run <- function(args) {
  if (args$context_mode != "local5") stop("Google supports only local5 context mode")
  if (!args$setting %in% c("Agnostic", "Aware", "Pooled")) stop("Invalid setting")
  if (!args$ppmi %in% c("PPMI", "RAW")) stop("Invalid ppmi")
  if (!args$tagging %in% c("UnTagged", "Tagged")) stop("Invalid tagging")
  if (!args$frequency_family %in% c("freq", "tf", "log_freq")) stop("Invalid frequency family")
  if (!args$productivity_family %in% c("prod", "family_size")) stop("Invalid productivity family")
  if (!args$train_variant %in% c("original", "modified") ||
      !args$test_variant %in% c("original", "modified")) stop("Invalid variant choice")
  if (args$context_size < 0L || args$cores < 1L || args$folds < 2L || args$repeats < 1L ||
      args$alpha_count < 2L || args$lambda_count < 2L || args$permutations < 1L) stop("Invalid numeric option")
  timespans <- coha_v3_parse_ints(args$timespans, "timespans")
  cutoffs <- coha_v3_parse_ints(args$cutoffs, "cutoffs")
  targets <- parse_choices(args$targets, c("compound", "modifier", "head"), "targets")
  allowed_sets <- c("all", "all_static", "all_no_setting", "representation_only",
                    names(coha_v3_feature_groups),
                    unlist(coha_v3_feature_groups, use.names = FALSE))
  feature_sets <- parse_choices(args$feature_sets, allowed_sets, "feature sets")
  if (args$pilot) {
    args$folds <- 3L
    args$repeats <- 1L
    args$alpha_count <- 3L
    args$lambda_count <- 12L
    args$permutations <- 1L
  }
  if (!args$validate_only) {
    suppressPackageStartupMessages(library(caret))
    suppressPackageStartupMessages(library(glmnet))
    if (args$cores > 1L) {
      suppressPackageStartupMessages(library(doParallel))
      cluster <- parallel::makePSOCKcluster(args$cores)
      doParallel::registerDoParallel(cluster)
      on.exit(parallel::stopCluster(cluster), add = TRUE)
    }
    dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  grid <- expand.grid(alpha = seq(0, 1, length.out = args$alpha_count),
                      lambda = 10^seq(-4, 4, length.out = args$lambda_count))
  patterns <- if (args$tagging == "Tagged") c("nn", "an") else "all"
  reference_rosters <- list()
  for (timespan in timespans) for (cutoff in cutoffs) {
    input <- coha_v3_read_input(args$input_root, args$context_mode, args$context_size,
                                args$setting, args$ppmi, args$tagging, timespan, cutoff)
    for (pattern in patterns) {
      split <- if (args$train_variant == "original" && args$test_variant == "original") {
        coha_v3_original_split(input, args$tagging, pattern)
      } else {
        coha_v3_variant_split(input, args$train_variant, args$test_variant,
                              args$tagging, pattern)
      }
      roster <- list(train = sort(coha_v3_row_key(split$train)),
                     test = sort(coha_v3_row_key(split$test)))
      if (is.null(reference_rosters[[pattern]])) {
        reference_rosters[[pattern]] <- roster
      } else if (!identical(reference_rosters[[pattern]], roster)) {
        stop("Rating roster differs across time-bin/cutoff configurations for ", pattern)
      }
      available <- if (timespan == 10000L) {
        setdiff(feature_sets, c("all_static", "temporal", coha_v3_feature_groups$temporal))
      } else feature_sets
      suffix <- sprintf("google_v3_%s_ctx%s_%s_%s_%s_%s_%s_%s_%s_%s",
                        args$context_mode, args$context_size, args$tagging, args$ppmi,
                        args$setting, timespan, cutoff, pattern,
                        args$frequency_family, args$productivity_family)
      if (args$train_variant != "original" || args$test_variant != "original") {
        suffix <- paste0(suffix, "_train", args$train_variant,
                         "_test", args$test_variant)
      }
      output_types <- c("rsquared", "coefimp", "permimp", "familyimp", "predictions",
                        "cvgrid", "cvfolds")
      paths <- setNames(file.path(args$output_dir, paste0(output_types, "_", suffix, ".csv")),
                        output_types)
      manifest_path <- file.path(args$output_dir, paste0("complete_", suffix, ".rds"))
      input_paths <- coha_v3_feature_path(args$input_root, args$context_mode, args$context_size,
                                          "features", args$setting, args$ppmi, args$tagging, timespan, cutoff)
      if (timespan != 10000L) input_paths <- c(input_paths,
        coha_v3_feature_path(args$input_root, args$context_mode, args$context_size,
                             "temporal", args$setting, args$ppmi, args$tagging, timespan, cutoff))
      code_paths <- file.path(script_dir, c("regression_exps.r", "coha_v3_regression_data.R"))
      config <- list(input_md5 = unname(tools::md5sum(input_paths)),
                     code_md5 = unname(tools::md5sum(code_paths)), targets = targets,
                     feature_sets = available, folds = args$folds, repeats = args$repeats,
                     train_variant = args$train_variant, test_variant = args$test_variant,
                     frequency_family = args$frequency_family,
                     productivity_family = args$productivity_family,
                     alpha_count = args$alpha_count, lambda_count = args$lambda_count,
                     permutations = args$permutations)
      message(suffix, ": ", nrow(split$train), " train / ", nrow(split$test), " test; ",
              ncol(coha_v3_select_features(input, "all", args$frequency_family,
                                           args$productivity_family)), " feature columns")
      if (args$validate_only) {
        for (feature_set in available) {
          predictors <- coha_v3_select_features(split$train, feature_set,
                                                 args$frequency_family, args$productivity_family)
          keep <- usable_columns(predictors)
          missing_rate <- if (length(keep)) mean(is.na(predictors[keep])) else NA_real_
          message("  ", feature_set, ": ", length(keep), "/", ncol(predictors),
                  " usable predictors; training missing fraction ",
                  if (is.na(missing_rate)) "NA" else sprintf("%.3f", missing_rate))
        }
        next
      }
      if (!args$force && file.exists(manifest_path)) {
        previous <- readRDS(manifest_path)
        if (!identical(previous, config)) stop("Configuration changed for ", suffix,
                                             "; choose a new output directory or use --force")
        if (all(file.exists(paths))) {
          message("Skipping complete configuration: ", suffix)
          next
        }
      }
      output <- setNames(vector("list", length(paths)), names(paths))
      for (feature_set in available) for (target in targets) {
        message("  ", feature_set, " -> ", target)
        fitted <- fit_one(split$train, split$test, feature_set, target, args,
                          timespan, cutoff, pattern, grid)
        if (is.null(fitted)) next
        for (name in names(paths)) output[[name]][[length(output[[name]]) + 1L]] <- fitted[[name]]
      }
      if (!length(output$rsquared)) {
        warning("No models had usable predictors for ", suffix)
        next
      }
      for (name in names(paths)) {
        frames <- output[[name]]
        value <- if (length(frames)) do.call(rbind, frames) else data.frame()
        atomic_csv(value, paths[[name]])
      }
      temp_manifest <- tempfile(tmpdir = args$output_dir, fileext = ".rds")
      saveRDS(config, temp_manifest)
      if (!file.rename(temp_manifest, manifest_path)) stop("Cannot save ", manifest_path)
    }
  }
}

run(parse_args(OptionParser(option_list = option_list)))
