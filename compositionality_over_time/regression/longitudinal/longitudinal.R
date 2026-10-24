#!/usr/bin/env Rscript

# Cumulative-history experiment. The modern train/test split and modern-trained
# preprocessing are frozen before applying one Spearman-tuned model to history.
# Sourcing this file exposes its helpers without starting an experiment.
google_longitudinal_script <- local({
  sources <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  own <- sources[grepl("longitudinal\\.R$", unlist(sources))]
  argv <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  path <- if (length(own)) tail(own, 1L)[[1L]] else if (length(argv)) argv[[1L]] else
    file.path(getwd(), "longitudinal.R")
  normalizePath(path, mustWork = TRUE)
})
google_longitudinal_helper <- file.path(dirname(google_longitudinal_script), "..",
                                        "modified_regressions", "coha_v3_regression_data.R")
source(google_longitudinal_helper)
google_longitudinal_padding <- ".glmnet_padding"

google_longitudinal_defaults <- function() list(
  feature_root = "runs/compositionality/regressions/inputs_snapshot",
  cumulative_root = "runs/compositionality/longitudinal/features_cumulative",
  output_dir = "runs/compositionality/longitudinal/results", context_mode = "local5",
  context_size = 5000L, setting = "Pooled", ppmi = "PPMI", tagging = "UnTagged",
  pattern = "all", cutoff = 0L, target = "compound", feature_set = "all_static",
  train_variant = "modified", test_variant = "modified", time_step = 10L,
  frequency_family = "freq", productivity_family = "prod", folds = 5L,
  repeats = 10L, alpha_count = 10L, lambda_count = 200L, pilot = FALSE,
  check = FALSE, validate_only = FALSE, check_environment = FALSE, force = FALSE)

google_longitudinal_parse <- function() {
  if (!requireNamespace("optparse", quietly = TRUE)) stop("Missing R package: optparse")
  defaults <- google_longitudinal_defaults()
  flags <- lapply(names(defaults), function(name) {
    value <- defaults[[name]]
    arguments <- list(opt_str = paste0("--", gsub("_", "-", name)), dest = name, default = value,
                      help = paste0(gsub("_", " ", name), " [default: %default]"))
    if (is.logical(value)) arguments$action <- "store_true"
    else if (is.integer(value)) arguments$type <- "integer"
    do.call(optparse::make_option, arguments)
  })
  parsed <- optparse::parse_args(optparse::OptionParser(option_list = flags))
  parsed[names(defaults)]
}

google_longitudinal_packages <- function(growth = TRUE) {
  packages <- c("optparse", "jsonlite", "digest", "caret", "glmnet", "ggplot2",
                if (growth) "lme4")
  present <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  if (any(!present)) stop("Missing R packages: ", paste(packages[!present], collapse = ", "),
                          ". Use a compatible existing R environment; no packages were installed.")
  if (growth) {
    abi_check <- get0("check_dep_version", envir = asNamespace("lme4"), inherits = FALSE)
    if (is.function(abi_check) && !isTRUE(suppressWarnings(abi_check()))) {
      stop("lme4/Matrix ABI mismatch; use an environment with compatible packages")
    }
  }
  invisible(packages)
}

google_longitudinal_options <- function(args) {
  if (isTRUE(args$pilot)) {
    args$folds <- 3L; args$repeats <- 1L; args$alpha_count <- 3L; args$lambda_count <- 12L
  }
  if (args$context_mode != "local5" ||
      !args$setting %in% c("Pooled", "Aware", "Agnostic") ||
      !args$ppmi %in% c("PPMI", "RAW") || !args$tagging %in% c("UnTagged", "Tagged") ||
      !args$target %in% c("compound", "head", "modifier") ||
      !args$train_variant %in% c("original", "modified") ||
      !args$test_variant %in% c("original", "modified") ||
      !args$frequency_family %in% c("freq", "tf", "log_freq") ||
      !args$productivity_family %in% c("prod", "family_size") ||
      args$context_size < 1L || args$cutoff < 0L || args$time_step != 10L ||
      args$folds < 2L || args$repeats < 1L || args$alpha_count < 2L || args$lambda_count < 2L) {
    stop("Invalid longitudinal option (cumulative endpoints require --time-step 10)")
  }
  coha_v3_pattern_rows(data.frame(modifier = character(), head = character(),
                                 is_adj = logical()), args$tagging, args$pattern)
  args
}

google_longitudinal_sha <- function(path) {
  if (!file.exists(path)) stop("Missing artifact: ", path)
  before <- file.info(path)[c("size", "mtime")]
  value <- digest::digest(file = path, algo = "sha256")
  if (!identical(before, file.info(path)[c("size", "mtime")])) stop("Artifact changed while hashing: ", path)
  value
}

google_longitudinal_hash_text <- function(value) digest::digest(value, algo = "sha256", serialize = FALSE)

google_longitudinal_read_json <- function(path) {
  if (!file.exists(path)) stop("Missing manifest: ", path)
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

google_longitudinal_identity <- function(info, field) {
  value <- info[[field]]
  material <- info[[paste0(field, "_material")]]
  if (!is.character(value) || length(value) != 1L || !grepl("^[0-9a-f]{64}$", value) ||
      !is.character(material) || length(material) != 1L ||
      !identical(google_longitudinal_hash_text(material), value)) {
    stop("Invalid or unverified ", field)
  }
  jsonlite::fromJSON(material, simplifyVector = FALSE)
}

google_longitudinal_record <- function(record, base = ".") {
  if (is.null(record$path) || is.null(record$sha256) || is.null(record$bytes) ||
      !is.character(record$path) || length(record$path) != 1L ||
      !grepl("^[0-9a-f]{64}$", record$sha256)) stop("Invalid artifact record")
  path <- if (startsWith(record$path, "/")) record$path else file.path(base, record$path)
  path <- normalizePath(path, mustWork = TRUE)
  if (file.info(path)$size != record$bytes ||
      !identical(google_longitudinal_sha(path), record$sha256)) stop("Artifact hash/size mismatch: ", path)
  path
}

google_longitudinal_same <- function(left, right, label, tolerance = 0) {
  if (!isTRUE(all.equal(left, right, check.attributes = FALSE, tolerance = tolerance))) stop(label)
}

google_longitudinal_check_roster <- function(modern, historical, label) {
  key <- coha_v3_row_key(modern)
  order <- match(key, coha_v3_row_key(historical))
  if (nrow(modern) != nrow(historical) || anyNA(order)) stop("Rating roster differs: ", label)
  for (column in coha_v3_rating_cols) {
    google_longitudinal_same(modern[[column]], historical[[column]][order],
                              paste0("Rating payload differs: ", label, ", ", column))
  }
  order
}

google_longitudinal_check_parity <- function(modern, historical, tolerance = 1e-6) {
  order <- google_longitudinal_check_roster(modern, historical, "final cumulative endpoint")
  if (!setequal(names(modern), names(historical))) stop("Final cumulative feature columns differ from modern CSV")
  for (column in names(modern)) {
    left <- modern[[column]]; right <- historical[[column]][order]
    if (is.numeric(left)) {
      # An absolute tolerance also catches differences in large frequency values.
      same_na <- identical(is.na(left), is.na(right))
      finite <- is.finite(left) & is.finite(right)
      same <- same_na && identical(is.finite(left), is.finite(right)) &&
        all(abs(left[finite] - right[finite]) <= tolerance) &&
        isTRUE(all.equal(left[!finite], right[!finite], check.attributes = FALSE, tolerance = 0))
      if (!same) stop("Final cumulative feature parity failed: ", column)
    } else google_longitudinal_same(left, right, paste0("Final cumulative parity failed: ", column))
  }
  invisible(TRUE)
}

google_longitudinal_input <- function(args) {
  modern_path <- coha_v3_feature_path(args$feature_root, args$context_mode, args$context_size,
    "features", args$setting, args$ppmi, args$tagging, 10000L, args$cutoff)
  modern <- coha_v3_read_input(args$feature_root, args$context_mode, args$context_size,
    args$setting, args$ppmi, args$tagging, 10000L, args$cutoff)
  split <- coha_v3_variant_split(modern, args$train_variant, args$test_variant,
                                 args$tagging, args$pattern)
  complete_path <- file.path(args$cumulative_root, "features_complete.json")
  complete <- google_longitudinal_read_json(complete_path)
  if (!identical(complete$status, "complete") || !length(complete$endpoints) ||
      !length(complete$manifests)) stop("Cumulative snapshot set is not complete")
  manifests <- vapply(complete$manifests, google_longitudinal_record, character(1),
                     base = args$cumulative_root)
  discovered <- list.files(file.path(args$cumulative_root, args$context_mode),
    pattern = "^cumulative_manifest\\.json$", full.names = TRUE, recursive = TRUE)
  if (!setequal(normalizePath(discovered, mustWork = TRUE), manifests)) {
    stop("Completed manifest inventory differs from cumulative snapshot directories")
  }
  config <- list(context_mode = args$context_mode, context_size = args$context_size,
    setting = args$setting, condition = paste0("Compound", args$setting), ppmi = args$ppmi == "PPMI",
    tagged = args$tagging == "Tagged", cutoff = args$cutoff, time_step = args$time_step,
    feature_definition_version = 3L, beta_policy = "clipped_unit_interval")
  infos <- lapply(manifests, google_longitudinal_read_json)
  frames <- paths <- vector("list", length(infos))
  modern_sha <- google_longitudinal_sha(modern_path)
  for (i in seq_along(infos)) {
    info <- infos[[i]]
    if (info$schema_version != 1L || !identical(info$stage, "google_cumulative_features")) {
      stop("Incompatible cumulative manifest: ", manifests[[i]])
    }
    input_material <- google_longitudinal_identity(info, "input_identity")
    snapshot_material <- google_longitudinal_identity(info, "snapshot_identity")
    for (name in names(config)) {
      google_longitudinal_same(info[[name]], config[[name]], paste0("Snapshot configuration differs: ", name))
      google_longitudinal_same(input_material[[name]], config[[name]], paste0("Input identity configuration differs: ", name))
    }
    for (name in c("input_identity", "endpoint_decade", "end_year", "files")) {
      google_longitudinal_same(snapshot_material[[name]], info[[name]], paste0("Snapshot identity material differs: ", name))
    }
    google_longitudinal_same(input_material$modern_source, info$modern_source, "Modern source identity material differs")
    source_path <- google_longitudinal_record(info$modern_source)
    if (source_path != normalizePath(modern_path) || info$modern_source$sha256 != modern_sha) {
      stop("Snapshot modern source differs from requested frozen CSV")
    }
    if (info$endpoint_decade %% 10L != 0L || info$end_year != info$endpoint_decade + 9L ||
        info$endpoint_decade > info$full_input_last_decade) stop("Invalid cumulative endpoint/year")
    paths[[i]] <- vapply(info$files, google_longitudinal_record, character(1), base = dirname(manifests[[i]]))
    expected <- file.path(dirname(manifests[[i]]), basename(modern_path))
    if (!normalizePath(expected, mustWork = TRUE) %in% paths[[i]]) stop("Snapshot CSV is absent from published artifacts")
    if (basename(dirname(manifests[[i]])) != paste0("through_", info$endpoint_decade)) stop("Snapshot directory and endpoint differ")
    frame <- coha_v3_read_input(dirname(manifests[[i]]), ".", args$context_size, args$setting,
                               args$ppmi, args$tagging, 10000L, args$cutoff)
    google_longitudinal_check_roster(modern, frame, paste0("endpoint ", info$endpoint_decade))
    # Always apply the same variant-aware helper, including its canonical keys.
    coha_v3_variant_split(frame, args$train_variant, args$test_variant, args$tagging, args$pattern)
    eligible <- frame[["eligible:0"]]
    if (is.null(eligible) || any(!is.na(eligible) & !eligible %in% c(0, 1))) {
      stop("Snapshot eligible:0 must be 0/1 or missing (unattested)")
    }
    if (any(!grepl(":0$", setdiff(names(frame), coha_v3_rating_cols)))) stop("Cumulative predictors must use epoch :0")
    frames[[i]] <- frame
  }
  endpoints <- vapply(infos, function(x) as.integer(x$endpoint_decade), integer(1))
  if (anyDuplicated(endpoints) || !setequal(endpoints, as.integer(unlist(complete$endpoints)))) stop("Completed endpoint inventory differs")
  identities <- vapply(infos, function(x) x$input_identity, character(1))
  if (length(unique(identities)) != 1L || complete$input_identity != identities[[1L]]) stop("Mixed cumulative input identities")
  if (anyDuplicated(vapply(infos, function(x) x$snapshot_identity, character(1)))) stop("Duplicate snapshot identities")
  last_decades <- vapply(infos, function(x) as.integer(x$full_input_last_decade), integer(1))
  if (length(unique(last_decades)) != 1L || max(endpoints) != last_decades[[1L]]) stop("Final full-input endpoint is required")
  ord <- order(endpoints)
  endpoints <- endpoints[ord]; infos <- infos[ord]; frames <- frames[ord]; manifests <- manifests[ord]; paths <- paths[ord]
  # A declared endpoint subset may thin the series while retaining all earlier
  # source observations. Its actual calendar spacing enters the growth model.
  final_path <- file.path(dirname(tail(manifests, 1L)), basename(modern_path))
  google_longitudinal_check_parity(
    read.csv(modern_path, check.names = FALSE, na.strings = c("NA", "")),
    read.csv(final_path, check.names = FALSE, na.strings = c("NA", "")))
  list(modern = modern, split = split, modern_path = modern_path, modern_sha256 = modern_sha,
       complete_path = complete_path, input_identity = identities[[1L]], endpoints = endpoints,
       infos = infos, frames = frames, manifests = manifests, artifacts = unlist(paths, use.names = FALSE))
}

google_longitudinal_spearman <- function(pred, obs) {
  if (length(pred) < 3L || any(!is.finite(pred)) || any(!is.finite(obs)) ||
      sd(pred) == 0 || sd(obs) == 0) return(NA_real_)
  suppressWarnings(stats::cor(pred, obs, method = "spearman"))
}

google_longitudinal_r2 <- function(pred, obs) {
  if (length(pred) < 2L || any(!is.finite(pred)) || any(!is.finite(obs))) return(NA_real_)
  denominator <- sum((obs - mean(obs))^2)
  if (denominator == 0) return(NA_real_)
  1 - sum((obs - pred)^2) / denominator
}

google_longitudinal_summary <- function(data, lev = NULL, model = NULL) {
  # Self-contained so caret can serialize the callback without global helpers.
  valid <- length(data$pred) >= 3L && all(is.finite(data$pred)) && all(is.finite(data$obs)) &&
    stats::sd(data$pred) > 0 && stats::sd(data$obs) > 0
  c(Spearman = if (valid) suppressWarnings(stats::cor(data$pred, data$obs, method = "spearman")) else NA_real_)
}

google_longitudinal_select <- function(df, args) coha_v3_select_features(
  df, args$feature_set, args$frequency_family, args$productivity_family)

google_longitudinal_x <- function(df, args, predictors) {
  raw <- google_longitudinal_select(df, args)
  if (length(setdiff(predictors, names(raw)))) stop("Historical model predictors are absent from snapshot schema")
  raw <- raw[predictors]
  for (name in names(raw)) raw[[name]][!is.finite(raw[[name]])] <- NA_real_
  if (length(predictors) == 1L) raw[[google_longitudinal_padding]] <- 0
  raw
}

google_longitudinal_fit <- function(input, args) {
  train <- input$split$train; test <- input$split$test
  raw <- google_longitudinal_select(train, args)
  predictors <- names(raw)[vapply(raw, function(x) {
    x <- x[is.finite(x)]
    length(x) >= 10L && is.finite(sd(x)) && sd(x) > 0
  }, logical(1))]
  if (!length(predictors)) stop("No usable modern training predictors")
  train_x <- google_longitudinal_x(train, args, predictors)
  test_x <- google_longitudinal_x(test, args, predictors)
  target <- c(compound = "compositionality", head = "avgHead", modifier = "avgModifier")[[args$target]]
  train_y <- train[[target]]; test_y <- test[[target]]
  if (any(!is.finite(train_y)) || any(!is.finite(test_y))) stop("Missing/nonfinite modern ratings")
  grid <- expand.grid(alpha = seq(0, 1, length.out = args$alpha_count),
                      lambda = 10^seq(-4, 4, length.out = args$lambda_count))
  set.seed(1111)
  folds <- caret::createMultiFolds(train_y, k = args$folds, times = args$repeats)
  control <- caret::trainControl(method = "repeatedcv", number = args$folds, repeats = args$repeats,
    index = folds, summaryFunction = google_longitudinal_summary, savePredictions = "final",
    returnResamp = "final", allowParallel = FALSE)
  # glmnet requires two predictors; zero padding must not be scaled.
  preprocessing <- if (length(predictors) == 1L) c("medianImpute", "center") else
    c("medianImpute", "center", "scale")
  set.seed(1111)
  model <- caret::train(x = train_x, y = train_y, method = "glmnet", metric = "Spearman",
    trControl = control, tuneGrid = grid, preProcess = preprocessing)
  predicted <- as.numeric(predict(model, newdata = test_x))
  if (any(!is.finite(predicted))) stop("Nonfinite modern held-out predictions")
  selected <- model$results$alpha == model$bestTune$alpha & model$results$lambda == model$bestTune$lambda
  if (sum(selected) != 1L || !is.finite(model$results$Spearman[selected])) stop("No valid Spearman tuning result")
  coefficients <- as.matrix(stats::coef(model$finalModel, s = model$bestTune$lambda))
  cv_predictions <- model$pred
  if (is.null(cv_predictions) || !all(c("rowIndex", "Resample", "pred", "obs") %in% names(cv_predictions))) {
    stop("Training CV fold predictions were not retained")
  }
  cv_predictions$compound <- train$compound[cv_predictions$rowIndex]
  cv_predictions$source <- train$source[cv_predictions$rowIndex]
  canonical_train <- train$compound
  for (i in seq_len(nrow(coha_v3_variant_pairs))) {
    pair <- coha_v3_variant_pairs[i, ]
    canonical_train[train$source == pair$source & train$compound == pair$modified] <- pair$original
  }
  cv_predictions$canonical_compound <- canonical_train[cv_predictions$rowIndex]
  cv_predictions$cv_objective <- "Spearman"
  list(model = model, predictors = predictors, target_column = target,
    preprocessing = model$preProcess, training_fold_indices = folds,
    coefficients = data.frame(feature = rownames(coefficients), coefficient = coefficients[, 1L], row.names = NULL),
    modern_predictions = data.frame(compound = test$compound, canonical_compound = test$canonical_compound,
      source = test$source, observed = test_y, predicted = predicted),
    performance = data.frame(corpus = "google", context_mode = args$context_mode,
      context_size = args$context_size, setting = args$setting, ppmi = args$ppmi,
      tagging = args$tagging, pattern = args$pattern, cutoff = args$cutoff, target = args$target,
      feature_set = args$feature_set, train_variant = args$train_variant, test_variant = args$test_variant,
      time_step = args$time_step, train_n = nrow(train), test_n = nrow(test), predictor_count = length(predictors),
      tuning_objective = "Spearman", cv_spearman = model$results$Spearman[selected],
      cv_spearman_sd = model$results$SpearmanSD[selected], alpha = model$bestTune$alpha,
      lambda = model$bestTune$lambda, test_spearman = google_longitudinal_spearman(predicted, test_y),
      test_r2 = google_longitudinal_r2(predicted, test_y), test_mae = mean(abs(predicted - test_y)),
      r2_definition = "1 - SSE/SST; same Spearman-tuned predictions"),
    cv_grid = transform(model$results, selected = selected), cv_folds = model$resample,
    cv_predictions = cv_predictions)
}

google_longitudinal_trajectories <- function(input, frozen, args) {
  trajectory <- coverage <- per_compound <- vector("list", length(input$endpoints))
  for (i in seq_along(input$endpoints)) {
    frame <- coha_v3_variant_split(input$frames[[i]], args$train_variant, args$test_variant,
                                  args$tagging, args$pattern)$test
    order <- match(coha_v3_row_key(input$split$test), coha_v3_row_key(frame))
    if (anyNA(order)) stop("Selected historical variant roster differs")
    frame <- frame[order, , drop = FALSE]
    x <- google_longitudinal_x(frame, args, frozen$predictors)
    attested <- !is.na(frame[["eligible:0"]]) & frame[["eligible:0"]] == 1
    nonmissing <- rowSums(!is.na(x[frozen$predictors])) > 0L
    eligible <- attested & nonmissing
    status <- ifelse(!attested, "unattested", ifelse(!nonmissing, "all_predictors_missing", "predicted"))
    endpoint <- input$endpoints[[i]]; end_year <- input$infos[[i]]$end_year
    per_compound[[i]] <- data.frame(endpoint_decade = endpoint, end_year = end_year,
      canonical_compound = frame$canonical_compound, compound = frame$compound, source = frame$source,
      eligible = attested, status = status, stringsAsFactors = FALSE)
    coverage[[i]] <- data.frame(endpoint_decade = endpoint, end_year = end_year, total_test = nrow(frame),
      attested_test = sum(attested), unattested_test = sum(!attested),
      all_predictors_missing_test = sum(attested & !nonmissing), predicted_test = sum(eligible))
    if (!any(eligible)) next
    predicted <- as.numeric(predict(frozen$model, newdata = x[eligible, , drop = FALSE]))
    if (any(!is.finite(predicted))) stop("Nonfinite historical predictions at endpoint ", endpoint)
    trajectory[[i]] <- data.frame(compound_id = paste(frame$source[eligible], frame$canonical_compound[eligible], sep = "::"),
      compound = frame$compound[eligible], canonical_compound = frame$canonical_compound[eligible],
      modifier = frame$modifier[eligible], head = frame$head[eligible], source = frame$source[eligible],
      endpoint_decade = endpoint, end_year = end_year, predicted_compositionality = predicted,
      observed_modern_rating = frame[[frozen$target_column]][eligible],
      missing_predictor_fraction = rowMeans(is.na(x[eligible, frozen$predictors, drop = FALSE])))
  }
  pieces <- Filter(Negate(is.null), trajectory)
  combined <- if (length(pieces)) do.call(rbind, pieces) else data.frame(
    compound_id = character(), compound = character(), canonical_compound = character(),
    modifier = character(), head = character(), source = character(), endpoint_decade = integer(),
    end_year = integer(), predicted_compositionality = numeric(), observed_modern_rating = numeric(),
    missing_predictor_fraction = numeric())
  list(trajectories = combined, coverage = do.call(rbind, coverage),
       coverage_by_compound = do.call(rbind, per_compound))
}

google_longitudinal_atomic <- function(path, writer) {
  temporary <- tempfile(pattern = ".longitudinal-", tmpdir = dirname(path), fileext = paste0(".", tools::file_ext(path)))
  on.exit(unlink(temporary), add = TRUE)
  writer(temporary)
  if (!file.rename(temporary, path)) stop("Cannot publish artifact: ", path)
  invisible(path)
}

google_longitudinal_csv <- function(value, path) google_longitudinal_atomic(path, function(temp) {
  write.csv(value, temp, row.names = FALSE, na = "")
})

google_longitudinal_rds <- function(value, path) google_longitudinal_atomic(path, function(temp) saveRDS(value, temp))

google_longitudinal_checkpoint_source_info <- function(checkpoint, config, model_path) {
  if (is.null(checkpoint$frozen_identity) ||
      !identical(checkpoint$frozen_identity, digest::digest(checkpoint$frozen, algo = "sha256"))) {
    stop("Frozen model checkpoint failed its content identity check")
  }
  if (!identical(checkpoint$config, config)) {
    stop("Frozen model identity changed; use a new output directory or --force")
  }
  invisible(NULL)
}

google_longitudinal_growth <- function(trajectory, endpoints) {
  software <- list(R = R.version.string,
    lme4 = tryCatch(as.character(utils::packageVersion("lme4")), error = function(e) NA_character_),
    Matrix = as.character(utils::packageVersion("Matrix")))
  software_summary <- c(paste("R:", software$R), paste("lme4:", software$lme4, "; Matrix:", software$Matrix))
  caveat <- paste("Descriptive model-implied proxy, not observed historical ratings.",
    "Uncertainty is conditional on the frozen modern regression and its preprocessing;",
    "it excludes modern-model estimation uncertainty. Cumulative snapshots reuse earlier",
    "corpus counts, so serial dependence can make conventional mixed-model uncertainty optimistic.")
  counts <- table(trajectory$compound_id)
  growth <- trajectory[trajectory$compound_id %in% names(counts[counts >= 3L]), , drop = FALSE]
  if (length(unique(growth$compound_id)) < 3L || length(unique(growth$endpoint_decade)) < 3L) {
    return(list(status = "too_few_eligible_compounds_or_endpoints", fit = NULL, fitted = growth, software = software,
      slope = data.frame(term = "time_decades", estimate = NA_real_, std_error = NA_real_,
        conf_low = NA_real_, conf_high = NA_real_, units = "rating units per decade"),
      summary = c("Growth model not fitted: need >=3 endpoints per compound and >=3 compounds.", software_summary, caveat)))
  }
  google_longitudinal_packages(growth = TRUE)
  origin <- min(endpoints)
  growth$time_decades <- (growth$endpoint_decade - origin) / 10
  growth$compound_id <- factor(growth$compound_id)
  warnings <- character()
  fit <- withCallingHandlers(lme4::lmer(predicted_compositionality ~ time_decades +
    (1 + time_decades | compound_id), data = growth, REML = TRUE),
    warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
  coefficients <- lme4::fixef(fit)
  # vcov.merMod returns an S4 Matrix object. Base diag() may not dispatch
  # without attaching Matrix, so convert this small fixed-effect covariance.
  errors <- sqrt(diag(as.matrix(stats::vcov(fit))))
  slope <- data.frame(term = names(coefficients), estimate = unname(coefficients),
    std_error = unname(errors), conf_low = unname(coefficients - 1.96 * errors),
    conf_high = unname(coefficients + 1.96 * errors),
    units = c("rating units at first endpoint", "rating units per decade"),
    interval = "conditional normal-approximation 95%")
  growth$fitted <- as.numeric(predict(fit))
  messages <- unlist(fit@optinfo$conv$lme4$messages)
  optimizer_code <- fit@optinfo$conv$opt
  converged <- all(optimizer_code == 0) && !length(messages)
  singular <- lme4::isSingular(fit)
  list(status = "fitted", fit = fit, fitted = growth, slope = slope, origin = origin, software = software,
    converged = converged, singular = singular,
    summary = c(paste("Eligible compounds:", nlevels(growth$compound_id), "; observations:", nrow(growth)),
      paste("Converged:", converged, "; optimizer code:", paste(optimizer_code, collapse = ",")),
      paste("Singular:", singular), paste("Convergence messages:", paste(messages, collapse = "; ")),
      paste("Warnings:", paste(warnings, collapse = "; ")), software_summary, caveat, capture.output(summary(fit))))
}

google_longitudinal_plot <- function(trajectory, growth, path) {
  plot <- ggplot2::ggplot(trajectory, ggplot2::aes(x = end_year, y = predicted_compositionality, group = compound_id)) +
    ggplot2::geom_line(color = "grey65", alpha = 0.3, linewidth = 0.35) +
    ggplot2::geom_point(color = "grey45", alpha = 0.25, size = 0.6) +
    ggplot2::labs(title = "Model-implied compositionality across cumulative Google history",
      subtitle = "Gray points/lines: frozen-model predictions; blue: compound growth fits; red: population fit",
      x = "Cumulative history through year", y = "Predicted compositionality (unclipped)",
      caption = "Descriptive proxy. Intervals are conditional on the modern model; cumulative serial dependence is not modeled.") +
    ggplot2::theme_minimal(base_size = 12)
  if (identical(growth$status, "fitted")) {
    fixed <- data.frame(end_year = seq(min(growth$fitted$end_year), max(growth$fitted$end_year), length.out = 100L))
    fixed$time_decades <- (fixed$end_year - 9 - growth$origin) / 10
    design <- cbind(1, fixed$time_decades)
    fixed$predicted <- as.numeric(design %*% lme4::fixef(growth$fit))
    uncertainty <- sqrt(rowSums((design %*% as.matrix(stats::vcov(growth$fit))) * design))
    fixed$lower <- fixed$predicted - 1.96 * uncertainty; fixed$upper <- fixed$predicted + 1.96 * uncertainty
    plot <- plot + ggplot2::geom_line(data = growth$fitted, ggplot2::aes(y = fitted),
      color = "#2878a1", alpha = 0.2, linewidth = 0.4) +
      ggplot2::geom_ribbon(data = fixed, ggplot2::aes(x = end_year, ymin = lower, ymax = upper),
        inherit.aes = FALSE, fill = "#c52822", alpha = 0.12) +
      ggplot2::geom_line(data = fixed, ggplot2::aes(x = end_year, y = predicted),
        inherit.aes = FALSE, color = "#c52822", linewidth = 1)
  }
  google_longitudinal_atomic(path, function(temp) ggplot2::ggsave(temp, plot, width = 10, height = 6, dpi = 160))
}

google_longitudinal_run <- function(args) {
  args <- google_longitudinal_options(args)
  if (isTRUE(args$check_environment)) {
    google_longitudinal_packages(growth = TRUE)
    message("Longitudinal R environment is compatible")
    return(invisible(NULL))
  }
  google_longitudinal_packages(growth = FALSE)
  input <- google_longitudinal_input(args)
  message("Verified cumulative inputs: ", length(input$endpoints), " endpoints; ", nrow(input$split$train),
          "/", nrow(input$split$test), " modern train/test rows; final full-feature parity passed")
  if (isTRUE(args$check) || isTRUE(args$validate_only)) {
    google_longitudinal_packages(growth = TRUE)
    message("Longitudinal input and environment checks passed")
    return(invisible(input))
  }
  output <- normalizePath(args$output_dir, mustWork = FALSE)
  for (root in c(args$feature_root, args$cumulative_root)) {
    root <- normalizePath(root, mustWork = TRUE)
    if (output == root || startsWith(output, paste0(root, "/")) || startsWith(root, paste0(output, "/"))) {
      stop("Longitudinal outputs must be separate from feature inputs")
    }
  }
  config_args <- args[setdiff(names(args), c("check", "validate_only", "check_environment", "force",
                                           "output_dir"))]
  artifact_paths <- unique(c(input$modern_path, input$complete_path, input$manifests, input$artifacts))
  cv_packages <- c("caret", "glmnet", "Matrix", "digest", "jsonlite", "ggplot2", "optparse")
  config <- list(options = config_args, input_identity = input$input_identity,
    software = list(R = R.version.string, packages = setNames(vapply(cv_packages,
      function(name) as.character(utils::packageVersion(name)), character(1)), cv_packages)),
    input_sha256 = setNames(vapply(artifact_paths, google_longitudinal_sha, character(1)), artifact_paths),
    source_sha256 = setNames(vapply(c(google_longitudinal_script, google_longitudinal_helper),
      google_longitudinal_sha, character(1)), c(google_longitudinal_script, google_longitudinal_helper)))
  receipt_path <- file.path(output, "complete_longitudinal.rds")
  previous <- NULL
  if (file.exists(receipt_path) && !isTRUE(args$force)) {
    previous <- readRDS(receipt_path)
    if (!identical(previous$config, config)) stop("Longitudinal configuration/source/input changed; choose a new output directory or use --force")
    intact <- all(file.exists(names(previous$output_sha256))) &&
      identical(previous$output_sha256, setNames(vapply(names(previous$output_sha256),
        google_longitudinal_sha, character(1)), names(previous$output_sha256)))
    if (intact) {
      message("Skipping complete Google longitudinal configuration")
      return(invisible(previous))
    }
  }
  model_path <- file.path(output, "frozen_modern_model.rds")
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  checkpoint <- if (file.exists(model_path) && !isTRUE(args$force)) readRDS(model_path) else NULL
  checkpoint_sha <- NULL
  if (!is.null(checkpoint)) {
    google_longitudinal_checkpoint_source_info(checkpoint, config, model_path)
    checkpoint_sha <- google_longitudinal_sha(model_path)
    message("Reusing verified frozen modern model; no modern-model training")
  }
  if (!is.null(checkpoint) && !is.null(previous) && !is.null(previous$output_sha256[[model_path]]) &&
      !identical(google_longitudinal_sha(model_path), previous$output_sha256[[model_path]])) {
    stop("Frozen model checkpoint differs from completed artifact hash")
  }
  frozen <- if (is.null(checkpoint)) google_longitudinal_fit(input, args) else checkpoint$frozen
  if (is.null(checkpoint)) google_longitudinal_rds(list(config = config, frozen = frozen,
    frozen_identity = digest::digest(frozen, algo = "sha256")), model_path)
  histories <- google_longitudinal_trajectories(input, frozen, args)
  tables <- c(histories, list(modern_predictions = frozen$modern_predictions, modern_performance = frozen$performance,
    coefficients = frozen$coefficients, cv_grid = frozen$cv_grid, cv_folds = frozen$cv_folds,
    cv_predictions = frozen$cv_predictions))
  written <- model_path
  for (name in names(tables)) {
    path <- file.path(output, paste0(name, ".csv"))
    google_longitudinal_csv(tables[[name]], path); written <- c(written, path)
  }
  preprocess_path <- file.path(output, "frozen_preprocessing.rds")
  google_longitudinal_rds(frozen$preprocessing, preprocess_path); written <- c(written, preprocess_path)
  growth <- google_longitudinal_growth(histories$trajectories, input$endpoints)
  growth_path <- file.path(output, "growth_model.rds")
  google_longitudinal_rds(growth, growth_path); written <- c(written, growth_path)
  summary_path <- file.path(output, "growth_model_summary.txt")
  lines <- c(paste("Growth status:", growth$status),
    paste("Modern held-out Spearman:", frozen$performance$test_spearman),
    paste("Modern held-out residual R2:", frozen$performance$test_r2),
    paste("Modern held-out MAE:", frozen$performance$test_mae), growth$summary)
  google_longitudinal_atomic(summary_path, function(temp) writeLines(lines, temp)); written <- c(written, summary_path)
  for (name in c("slope", "fitted")) {
    path <- file.path(output, paste0("growth_", name, ".csv"))
    google_longitudinal_csv(growth[[name]], path); written <- c(written, path)
  }
  plot_path <- file.path(output, "growth_trajectories.png")
  google_longitudinal_plot(histories$trajectories, growth, plot_path); written <- c(written, plot_path)
  # Recheck inputs after fitting before publishing the atomic completion receipt.
  if (!identical(config$input_sha256, setNames(vapply(artifact_paths, google_longitudinal_sha, character(1)), artifact_paths))) {
    stop("Longitudinal input changed during fitting; no completion receipt published")
  }
  if (!identical(config$source_sha256, setNames(vapply(names(config$source_sha256),
      google_longitudinal_sha, character(1)), names(config$source_sha256)))) {
    stop("Longitudinal source changed during fitting; no completion receipt published")
  }
  if (!is.null(checkpoint_sha) && !identical(checkpoint_sha, google_longitudinal_sha(model_path))) {
    stop("Frozen model checkpoint changed during resume; no completion receipt published")
  }
  receipt <- list(config = config, status = "complete", growth_status = growth$status,
    output_sha256 = setNames(vapply(written, google_longitudinal_sha, character(1)), written))
  google_longitudinal_rds(receipt, receipt_path)
  message("Saved frozen-model trajectories and growth status: ", growth$status)
  invisible(receipt)
}

if (sys.nframe() == 0L) google_longitudinal_run(google_longitudinal_parse())
