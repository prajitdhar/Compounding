#!/usr/bin/env Rscript

# Cumulative-prefix temporal model built on the common longitudinal driver.
google_longitudinal_temporal_script <- local({
  sources <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  own <- sources[grepl("longitudinal_temporal\\.R$", unlist(sources))]
  argv <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  path <- if (length(own)) tail(own, 1L)[[1L]] else if (length(argv)) argv[[1L]] else
    file.path(getwd(), "longitudinal_temporal.R")
  normalizePath(path, mustWork = TRUE)
})
google_longitudinal_base_script <- file.path(dirname(google_longitudinal_temporal_script),
                                               "longitudinal.R")
source(google_longitudinal_base_script)
google_longitudinal_script <- google_longitudinal_temporal_script
google_longitudinal_base_sha <- google_longitudinal_sha(google_longitudinal_base_script)
google_longitudinal_temporal_policy <- "cumulative_prefix_cosine_lag10_v1"
google_longitudinal_temporal_columns <- c("change_comp", "change_mod", "change_head")
google_longitudinal_temporal_epochs <- paste0(google_longitudinal_temporal_columns, ":0")

google_longitudinal_temporal_base_defaults <- google_longitudinal_defaults
google_longitudinal_temporal_base_options <- google_longitudinal_options
google_longitudinal_temporal_base_trajectories <- google_longitudinal_trajectories

google_longitudinal_defaults <- function() {
  args <- google_longitudinal_temporal_base_defaults()
  args$context_size <- 30000L; args$setting <- "Aware"; args$cutoff <- 500L
  args$feature_set <- "all"
  args$cumulative_root <- "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_500_cumulative_lag10/features_cumulative"
  args$output_dir <- "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_500_cumulative_lag10/results"
  args
}

google_longitudinal_options <- function(args) {
  args <- google_longitudinal_temporal_base_options(args)
  required <- list(context_mode = "local5", context_size = 30000L, setting = "Aware",
    ppmi = "PPMI", tagging = "UnTagged", cutoff = 500L, time_step = 10L,
    pattern = "all", target = "compound", train_variant = "modified", test_variant = "modified")
  for (name in names(required)) {
    google_longitudinal_same(args[[name]], required[[name]],
      paste0("Cumulative temporal experiment is restricted to the requested configuration: ", name))
  }
  if (!args$feature_set %in% c("all", "all_static", "representation_only", "temporal")) {
    stop("Temporal feature set must be all, all_static, representation_only, or temporal")
  }
  if (!identical(google_longitudinal_base_sha, google_longitudinal_sha(google_longitudinal_base_script))) {
    stop("Sourced base longitudinal driver changed; restart with unchanged source")
  }
  args$temporal_policy <- google_longitudinal_temporal_policy
  args$temporal_lag <- 10L
  args$base_driver_sha256 <- google_longitudinal_base_sha
  args
}

google_longitudinal_temporal_metadata <- function(info, label) {
  expected <- list(temporal_policy = google_longitudinal_temporal_policy, temporal_lag = 10L)
  for (name in names(expected)) {
    google_longitudinal_same(info[[name]], expected[[name]],
      paste0("Temporal policy differs in ", label, ": ", name))
  }
  google_longitudinal_same(unlist(info$temporal_columns, use.names = FALSE),
    google_longitudinal_temporal_columns, paste0("Temporal columns differ in ", label))
}

google_longitudinal_temporal_schema <- function(frame, no_lag = FALSE) {
  if (!all(google_longitudinal_temporal_epochs %in% names(frame)) ||
      !setequal(grep("^change_", names(frame), value = TRUE), google_longitudinal_temporal_epochs)) {
    stop("Cumulative temporal schema must contain exactly change_comp:0, change_mod:0, change_head:0")
  }
  for (name in google_longitudinal_temporal_epochs) {
    value <- frame[[name]]
    if (!is.numeric(value) || any(is.nan(value)) ||
        any(!is.na(value) & (!is.finite(value) | value < 0 | value > 1))) {
      stop("Temporal cosine similarity must be finite in [0,1] or NA: ", name)
    }
    if (no_lag && any(!is.na(value))) stop("First cumulative prefix must have missing temporal similarities")
  }
  invisible(TRUE)
}

google_longitudinal_temporal_sources <- function(material) {
  directory <- normalizePath(file.path(dirname(google_longitudinal_script), "..", ".."))
  records <- c(material$software$code_sha256, material$temporal_software$code_sha256)
  paths <- character()
  google_sources <- c("cumulative_features.py", "cumulative_temporal.py",
    "feature_math.py", "features.py", "pipeline_source_info.py")
  coha_sources <- c("coha_feature_pipeline.py", "coha_features.py")
  shared_sources <- c("source_info.py", "io.py", "progress.py", "partitioned_parquet.py")
  shared_directory <- file.path(dirname(directory), "src")
  if (!setequal(names(material$software$code_sha256),
                setdiff(c(google_sources, coha_sources, shared_sources), "cumulative_temporal.py")) ||
      !setequal(names(material$temporal_software$code_sha256), c("cumulative_temporal.py", shared_sources))) {
    stop("Missing or incompatible cumulative temporal generator source identities")
  }
  for (name in shared_sources) {
    if (!identical(material$software$code_sha256[[name]], material$temporal_software$code_sha256[[name]])) {
      stop("Shared generator source identities disagree: ", name)
    }
  }
  records <- records[!duplicated(names(records))]
  for (name in names(records)) {
    if (!name %in% c(google_sources, coha_sources, shared_sources) || !grepl("^[0-9a-f]{64}$", records[[name]])) {
      stop("Invalid generator source identity")
    }
    # Project helpers and shared infrastructure have separate source roots.
    # Follow the exact
    # generator import locations rather than treating matching basenames as
    # ambiguous (or selecting the wrong copy by directory order).
    found <- if (name %in% shared_sources) file.path(shared_directory, name) else
      if (name %in% coha_sources) file.path(directory, "feature_helpers", name) else file.path(directory, name)
    if (!file.exists(found) || !identical(google_longitudinal_sha(found), records[[name]])) {
      stop("Cumulative temporal generator source identity changed: ", name)
    }
    paths <- c(paths, normalizePath(found))
  }
  unique(paths)
}

google_longitudinal_input <- function(args) {
  args <- google_longitudinal_options(args)
  modern_path <- coha_v3_feature_path(args$feature_root, args$context_mode, args$context_size,
    "features", args$setting, args$ppmi, args$tagging, 10000L, args$cutoff)
  original <- coha_v3_read_input(args$feature_root, args$context_mode, args$context_size,
    args$setting, args$ppmi, args$tagging, 10000L, args$cutoff)
  if (any(grepl("^change_", names(original)))) stop("Original modern CSV must be static; use the augmented final snapshot for temporal training")
  complete_path <- file.path(args$cumulative_root, "features_complete.json")
  complete <- google_longitudinal_read_json(complete_path)
  google_longitudinal_temporal_metadata(complete, "completion")
  if (!identical(complete$status, "complete") || !length(complete$endpoints) ||
      !length(complete$manifests) || !isTRUE(complete$final_modern_parity_verified)) {
    stop("Cumulative temporal snapshot set is not complete")
  }
  manifests <- vapply(complete$manifests, google_longitudinal_record, character(1), base = args$cumulative_root)
  discovered <- list.files(file.path(args$cumulative_root, args$context_mode),
    pattern = "^cumulative_manifest\\.json$", full.names = TRUE, recursive = TRUE)
  if (!setequal(normalizePath(discovered, mustWork = TRUE), manifests)) {
    stop("Completed temporal manifest inventory differs from cumulative snapshot directories")
  }
  config <- list(context_mode = args$context_mode, context_size = args$context_size,
    setting = args$setting, condition = paste0("Compound", args$setting), ppmi = TRUE,
    tagged = FALSE, cutoff = args$cutoff, time_step = args$time_step,
    feature_definition_version = 3L, beta_policy = "clipped_unit_interval")
  infos <- lapply(manifests, google_longitudinal_read_json)
  frames <- paths <- vector("list", length(infos))
  generator_paths <- character()
  modern_sha <- google_longitudinal_sha(modern_path)
  for (i in seq_along(infos)) {
    info <- infos[[i]]
    if (!identical(info$schema_version, 1L) || !identical(info$stage, "google_cumulative_temporal_features")) {
      stop("Incompatible cumulative temporal manifest: ", manifests[[i]])
    }
    material <- google_longitudinal_identity(info, "input_identity")
    snapshot <- google_longitudinal_identity(info, "snapshot_identity")
    google_longitudinal_temporal_metadata(info, "snapshot")
    google_longitudinal_temporal_metadata(material, "input identity")
    google_longitudinal_temporal_metadata(snapshot, "snapshot identity")
    if (!"lag_endpoint_decade" %in% names(info) || !"lag_endpoint_decade" %in% names(snapshot)) {
      stop("Cumulative temporal manifests must explicitly declare the lag endpoint or null")
    }
    generator_paths <- unique(c(generator_paths, google_longitudinal_temporal_sources(material)))
    for (name in names(config)) {
      google_longitudinal_same(info[[name]], config[[name]], paste0("Snapshot configuration differs: ", name))
      google_longitudinal_same(material[[name]], config[[name]], paste0("Input identity configuration differs: ", name))
    }
    for (name in c("input_identity", "count_input_identity", "endpoint_decade", "end_year", "files", "lag_endpoint_decade")) {
      google_longitudinal_same(snapshot[[name]], info[[name]], paste0("Snapshot identity material differs: ", name))
    }
    google_longitudinal_same(material$modern_source, info$modern_source, "Modern source identity material differs")
    source_path <- google_longitudinal_record(info$modern_source)
    if (source_path != normalizePath(modern_path) || info$modern_source$sha256 != modern_sha) {
      stop("Snapshot modern source differs from requested frozen CSV")
    }
    if (!is.character(info$count_input_identity) || length(info$count_input_identity) != 1L ||
        !grepl("^[0-9a-f]{64}$", info$count_input_identity)) stop("Missing count-input preparation identity")
    google_longitudinal_same(material$count_input_identity, info$count_input_identity,
                             "Count preparation identity differs")
    first <- info$full_input_first_decade
    google_longitudinal_same(material$full_input_first_decade, first, "First source decade identity differs")
    if (length(first) != 1L || !is.numeric(first) || !is.finite(first) || first %% 10L != 0L ||
        info$endpoint_decade %% 10L != 0L || info$end_year != info$endpoint_decade + 9L ||
        first > info$endpoint_decade || info$endpoint_decade > info$full_input_last_decade) {
      stop("Invalid cumulative temporal endpoint/year/source range")
    }
    lag <- if (info$endpoint_decade - 10L < first) NULL else info$endpoint_decade - 10L
    google_longitudinal_same(info$lag_endpoint_decade, lag, "Temporal lag must be exactly ten calendar years")
    paths[[i]] <- vapply(info$files, google_longitudinal_record, character(1), base = dirname(manifests[[i]]))
    expected <- normalizePath(file.path(dirname(manifests[[i]]), basename(modern_path)), mustWork = TRUE)
    if (!expected %in% paths[[i]]) stop("Temporal snapshot CSV absent from published artifacts")
    if (basename(dirname(manifests[[i]])) != paste0("through_", info$endpoint_decade)) stop("Snapshot directory and endpoint differ")
    frame <- coha_v3_read_input(dirname(manifests[[i]]), ".", args$context_size, args$setting,
                               args$ppmi, args$tagging, 10000L, args$cutoff)
    google_longitudinal_check_roster(original, frame, paste0("endpoint ", info$endpoint_decade))
    coha_v3_variant_split(frame, args$train_variant, args$test_variant, args$tagging, args$pattern)
    eligible <- frame[["eligible:0"]]
    if (is.null(eligible) || any(!is.na(eligible) & !eligible %in% c(0, 1))) stop("Snapshot eligible:0 must be 0/1 or missing")
    if (any(!grepl(":0$", setdiff(names(frame), coha_v3_rating_cols)))) stop("Cumulative predictors must use epoch :0")
    google_longitudinal_temporal_schema(frame, no_lag = is.null(lag))
    frames[[i]] <- frame
  }
  endpoints <- vapply(infos, function(x) as.integer(x$endpoint_decade), integer(1))
  if (anyDuplicated(endpoints) || !setequal(endpoints, as.integer(unlist(complete$endpoints)))) stop("Completed endpoint inventory differs")
  identities <- vapply(infos, function(x) x$input_identity, character(1))
  if (length(unique(identities)) != 1L || complete$input_identity != identities[[1L]]) stop("Mixed temporal input identities")
  count_ids <- vapply(infos, function(x) x$count_input_identity, character(1))
  if (length(unique(count_ids)) != 1L || !identical(complete$count_input_identity, count_ids[[1L]])) stop("Mixed count preparation identities")
  if (anyDuplicated(vapply(infos, function(x) x$snapshot_identity, character(1)))) stop("Duplicate snapshot identities")
  firsts <- vapply(infos, function(x) as.integer(x$full_input_first_decade), integer(1))
  lasts <- vapply(infos, function(x) as.integer(x$full_input_last_decade), integer(1))
  if (length(unique(firsts)) != 1L || length(unique(lasts)) != 1L || max(endpoints) != lasts[[1L]]) stop("Final full-input endpoint is required")
  ord <- order(endpoints)
  endpoints <- endpoints[ord]; infos <- infos[ord]; frames <- frames[ord]; manifests <- manifests[ord]; paths <- paths[ord]
  final_path <- file.path(dirname(tail(manifests, 1L)), basename(modern_path))
  original_raw <- read.csv(modern_path, check.names = FALSE, na.strings = c("NA", ""))
  final_raw <- read.csv(final_path, check.names = FALSE, na.strings = c("NA", ""))
  final_static <- final_raw[setdiff(names(final_raw), google_longitudinal_temporal_epochs)]
  google_longitudinal_check_parity(original_raw, final_static)
  # Training uses the augmented FINAL prefix; the ordinary modern static CSV
  # is only a roster/static-parity reference, never a temporal proxy.
  modern <- tail(frames, 1L)[[1L]]
  split <- coha_v3_variant_split(modern, args$train_variant, args$test_variant, args$tagging, args$pattern)
  list(modern = modern, split = split, modern_path = modern_path, modern_sha256 = modern_sha,
    complete_path = complete_path, input_identity = identities[[1L]], endpoints = endpoints,
    infos = infos, frames = frames, manifests = manifests,
    # The unchanged runner also hashes and rechecks every artifact after fitting.
    # Including its sourced base implementation here protects both code files.
    artifacts = unique(c(unlist(paths, use.names = FALSE), google_longitudinal_base_script, generator_paths)))
}

google_longitudinal_checkpoint_source_info <- function(checkpoint, config, model_path) {
  if (is.null(checkpoint$frozen_identity) ||
      !identical(checkpoint$frozen_identity, digest::digest(checkpoint$frozen, algo = "sha256"))) {
    stop("Frozen model checkpoint failed its content identity check")
  }
  if (!identical(checkpoint$config, config)) stop("Frozen temporal model identity changed; use a new output directory or --force")
  invisible(NULL)
}

google_longitudinal_trajectories <- function(input, frozen, args) {
  result <- google_longitudinal_temporal_base_trajectories(input, frozen, args)
  for (i in seq_along(input$endpoints)) {
    frame <- coha_v3_variant_split(input$frames[[i]], args$train_variant, args$test_variant,
                                  args$tagging, args$pattern)$test
    order <- match(coha_v3_row_key(input$split$test), coha_v3_row_key(frame))
    frame <- frame[order, , drop = FALSE]
    available <- !is.na(frame[google_longitudinal_temporal_epochs])
    attested <- !is.na(frame[["eligible:0"]]) & frame[["eligible:0"]] == 1
    coverage_row <- result$coverage$endpoint_decade == input$endpoints[[i]]
    result$coverage$temporal_all_observed_test[coverage_row] <- sum(rowSums(available) == 3L)
    result$coverage$temporal_any_missing_test[coverage_row] <- sum(rowSums(available) < 3L)
    result$coverage$attested_temporal_all_observed_test[coverage_row] <- sum(attested & rowSums(available) == 3L)
    result$coverage$attested_temporal_any_missing_test[coverage_row] <- sum(attested & rowSums(available) < 3L)
    rows <- which(result$coverage_by_compound$endpoint_decade == input$endpoints[[i]])
    for (j in seq_along(google_longitudinal_temporal_columns)) {
      result$coverage_by_compound[rows, paste0(google_longitudinal_temporal_columns[[j]], "_observed")] <- available[, j]
    }
  }
  result
}

if (sys.nframe() == 0L) google_longitudinal_run(google_longitudinal_parse())
