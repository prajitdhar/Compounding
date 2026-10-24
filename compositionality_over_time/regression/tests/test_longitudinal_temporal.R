#!/usr/bin/env Rscript
# Published rating identities with synthetic counts/representations only. No
# production output, model or feature snapshot is modified by these tests.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
test_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
source(file.path(test_dir, "..", "longitudinal", "longitudinal_temporal.R"))

expect_error <- function(expression, pattern = NULL) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  if (is.null(error) || (!is.null(pattern) && !grepl(pattern, conditionMessage(error), fixed = TRUE))) {
    stop("Expected error containing: ", pattern, "; got: ", if (is.null(error)) "no error" else conditionMessage(error))
  }
}
json_write <- function(value, path) jsonlite::write_json(value, path, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null")
identity_json <- function(value) as.character(jsonlite::toJSON(value, auto_unbox = TRUE, digits = NA, null = "null"))
record <- function(path) list(path = normalizePath(path), sha256 = google_longitudinal_sha(path), bytes = file.info(path)$size)

make_test_data <- function(root, frame_transform = NULL, info_transform = NULL, complete_transform = NULL) {
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
  ratings[["sim_cpf_beta:0"]] <- ratings$compositionality + sin(seq_len(nrow(ratings)) / 7)
  ratings[["perc_token_mod:0"]] <- (seq_len(nrow(ratings)) %% 23L) / 23
  ratings[["eligible:0"]] <- 1L
  modern_path <- coha_v3_feature_path(args$feature_root, "local5", 30000L, "features", "Aware", "PPMI", "UnTagged", 10000L, 500L)
  write.csv(ratings, modern_path, row.names = FALSE, na = "NA")
  modern_source <- record(modern_path)
  directory <- normalizePath(file.path(dirname(google_longitudinal_script), "..", ".."))
  code_paths <- c(file.path(directory, c("cumulative_features.py", "feature_math.py",
    "features.py", "pipeline_source_info.py")),
    file.path(directory, "feature_helpers", c("coha_feature_pipeline.py", "coha_features.py")))
  shared_names <- c("source_info.py", "io.py", "progress.py", "partitioned_parquet.py")
  shared_paths <- file.path(dirname(directory), "src", shared_names)
  code_paths <- c(code_paths, shared_paths)
  shared_hashes <- setNames(lapply(shared_paths, google_longitudinal_sha), shared_names)
  code_hashes <- setNames(lapply(code_paths, google_longitudinal_sha), basename(code_paths))
  policy <- list(temporal_policy = google_longitudinal_temporal_policy, temporal_lag = 10L,
                 temporal_columns = as.list(google_longitudinal_temporal_columns))
  material <- c(list(context_mode = "local5", context_size = 30000L, setting = "Aware",
    condition = "CompoundAware", ppmi = TRUE, tagged = FALSE, cutoff = 500L, time_step = 10L,
    feature_definition_version = 3L, beta_policy = "clipped_unit_interval", modern_source = modern_source,
    count_input_identity = strrep("a", 64L), full_input_first_decade = 1900L,
    software = list(code_sha256 = code_hashes),
    temporal_software = list(code_sha256 = c(list(cumulative_temporal.py =
      google_longitudinal_sha(file.path(directory, "cumulative_temporal.py"))), shared_hashes))), policy)
  material_text <- identity_json(material); input_id <- google_longitudinal_hash_text(material_text)
  manifests <- vector("list", 2L)
  for (i in seq_along(manifests)) {
    endpoint <- c(1900L, 1910L)[[i]]
    directory <- file.path(args$cumulative_root, "local5", paste0("through_", endpoint))
    dir.create(directory, recursive = TRUE)
    frame <- ratings
    if (i == 1L) frame[["sim_cpf_beta:0"]] <- frame[["sim_cpf_beta:0"]] - 0.5
    for (j in seq_along(google_longitudinal_temporal_epochs)) {
      # All three modern controls vary, so model feature screening must retain
      # them independently of any elastic-net coefficient being set to zero.
      frame[[google_longitudinal_temporal_epochs[[j]]]] <- if (i == 1L) NA_real_ else
        (seq_len(nrow(frame)) %% c(17L, 19L, 29L)[[j]]) / c(17, 19, 29)[[j]]
    }
    if (!is.null(frame_transform)) frame <- frame_transform(frame, i)
    csv <- file.path(directory, basename(modern_path))
    write.csv(frame, csv, row.names = FALSE, na = "NA")
    files <- list(record(csv))
    lag <- if (i == 1L) NULL else 1900L
    snapshot <- c(list(input_identity = input_id, count_input_identity = material$count_input_identity,
      endpoint_decade = endpoint, end_year = endpoint + 9L, lag_endpoint_decade = lag, files = files), policy)
    snapshot_text <- identity_json(snapshot)
    info <- c(list(schema_version = 1L, stage = "google_cumulative_temporal_features"), material,
      list(input_identity = input_id, input_identity_material = material_text,
        snapshot_identity = google_longitudinal_hash_text(snapshot_text), snapshot_identity_material = snapshot_text,
        endpoint_decade = endpoint, end_year = endpoint + 9L, lag_endpoint_decade = lag,
        full_input_last_decade = 1910L, files = files))
    if (!is.null(info_transform)) info <- info_transform(info, i)
    path <- file.path(directory, "cumulative_manifest.json")
    json_write(info, path); manifests[[i]] <- record(path)
  }
  complete <- c(list(status = "complete", input_identity = input_id, count_input_identity = material$count_input_identity,
    endpoints = c(1900L, 1910L), manifests = manifests, final_modern_parity_verified = TRUE), policy)
  if (!is.null(complete_transform)) complete <- complete_transform(complete)
  json_write(complete, file.path(args$cumulative_root, "features_complete.json"))
  list(args = args, ratings = ratings, modern_path = modern_path, manifests = manifests)
}

main <- function() {
  google_longitudinal_packages(growth = TRUE)
  root <- tempfile("google-longitudinal-temporal-test-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  data <- make_test_data(file.path(root, "valid")); args <- data$args
  stopifnot(!dir.exists(args$output_dir), args$feature_set == "all", args$context_size == 30000L)
  for (name in c("context_size", "cutoff", "setting", "ppmi", "tagging", "time_step", "target", "pattern")) {
    changed <- args
    changed[[name]] <- switch(name, context_size = 5000L, cutoff = 0L, setting = "Pooled", ppmi = "RAW",
      tagging = "Tagged", time_step = 20L, target = "head", pattern = "nn")
    expect_error(google_longitudinal_options(changed))
  }
  input <- google_longitudinal_input(args)
  stopifnot(nrow(input$split$train) == 180L, nrow(input$split$test) == 100L,
    all(google_longitudinal_temporal_epochs %in% names(input$modern)),
    all(vapply(input$frames[[1L]][google_longitudinal_temporal_epochs], function(x) all(is.na(x)), logical(1))),
    all(vapply(input$modern[google_longitudinal_temporal_epochs], function(x) all(is.finite(x)), logical(1))),
    normalizePath(google_longitudinal_base_script) %in% input$artifacts)
  material <- google_longitudinal_identity(input$infos[[1L]], "input_identity")
  generator_paths <- google_longitudinal_temporal_sources(material)
  # The analysis wrapper is distinct from the authoritative shared helpers;
  # exact source mapping must retain it without ambiguous-name rejection.
  stopifnot(length(generator_paths) == 11L,
    file.path(normalizePath(file.path(dirname(google_longitudinal_script), "..", "..")),
      "pipeline_source_info.py") %in% generator_paths)
  bad_material <- material; bad_material$temporal_software$code_sha256$cumulative_temporal.py <- strrep("b", 64L)
  expect_error(google_longitudinal_temporal_sources(bad_material), "source identity changed")
  bad_material <- material; bad_material$software$code_sha256$pipeline_source_info.py <- NULL
  expect_error(google_longitudinal_temporal_sources(bad_material), "Missing or incompatible")
  for (set in c("all", "all_static", "representation_only", "temporal")) {
    selected_args <- args; selected_args$feature_set <- set
    selected <- names(google_longitudinal_select(input$split$train, selected_args))
    stopifnot(all(google_longitudinal_temporal_epochs %in% selected) == set %in% c("all", "temporal"))
    if (set == "all") stopifnot("perc_token_mod:0" %in% selected)
    if (set == "temporal") stopifnot(setequal(selected, google_longitudinal_temporal_epochs))
  }
  # Properly re-hashed invalid tables/manifests must fail semantic checks, not
  # merely file checksums. Static parity excludes exactly the three controls.
  changes <- list(
    missing_column = function(frame, i) { frame[["change_head:0"]] <- NULL; frame },
    out_of_range = function(frame, i) { if (i == 2L) frame[["change_comp:0"]][[1L]] <- 1.1; frame },
    infinite = function(frame, i) { if (i == 2L) frame[["change_mod:0"]][[1L]] <- Inf; frame },
    first_not_missing = function(frame, i) { if (i == 1L) frame[["change_head:0"]][[1L]] <- 0.5; frame },
    static_changed = function(frame, i) { if (i == 2L) frame[["perc_token_mod:0"]][[1L]] <- 0.99; frame },
    wrong_epoch = function(frame, i) { names(frame)[names(frame) == "change_comp:0"] <- "change_comp:1910"; frame })
  for (name in names(changes)) {
    bad <- make_test_data(file.path(root, name), frame_transform = changes[[name]])
    expect_error(google_longitudinal_input(bad$args))
  }
  bad <- make_test_data(file.path(root, "bad_policy"), info_transform = function(info, i) {
    info$temporal_policy <- "disjoint_decades"; info
  })
  expect_error(google_longitudinal_input(bad$args), "Temporal policy differs")
  bad <- make_test_data(file.path(root, "bad_complete"), complete_transform = function(info) {
    info$temporal_lag <- 20L; info
  })
  expect_error(google_longitudinal_input(bad$args), "Temporal policy differs")
  bad <- make_test_data(file.path(root, "bad_lag"), info_transform = function(info, i) {
    if (i == 2L) info$lag_endpoint_decade <- 1890L; info
  })
  expect_error(google_longitudinal_input(bad$args), "Snapshot identity material differs")
  bad <- make_test_data(file.path(root, "tampered"))
  final_path <- file.path(bad$args$cumulative_root, "local5", "through_1910", basename(bad$modern_path))
  writeLines("tampered", final_path)
  expect_error(google_longitudinal_input(bad$args), "hash/size mismatch")
  # Full pilot CV, one immutable modern fit and frozen historical application.
  receipt <- suppressWarnings(google_longitudinal_run(args))
  stopifnot(receipt$status == "complete", is.null(receipt$checkpoint_source_info),
    google_longitudinal_base_script %in% names(receipt$config$input_sha256),
    google_longitudinal_script %in% names(receipt$config$source_sha256),
    google_longitudinal_helper %in% names(receipt$config$source_sha256),
    receipt$config$options$temporal_policy == google_longitudinal_temporal_policy,
    receipt$config$options$base_driver_sha256 == google_longitudinal_base_sha)
  model_path <- file.path(args$output_dir, "frozen_modern_model.rds")
  checkpoint <- readRDS(model_path)
  frozen <- checkpoint$frozen
  stopifnot(all(google_longitudinal_temporal_epochs %in% frozen$predictors),
    "perc_token_mod:0" %in% frozen$predictors, all(is.finite(frozen$modern_predictions$predicted)))
  histories <- google_longitudinal_trajectories(input, frozen, args)
  stopifnot(all(histories$coverage$predicted_test == 100L),
    histories$coverage$temporal_all_observed_test[[1L]] == 0L,
    histories$coverage$temporal_any_missing_test[[1L]] == 100L,
    histories$coverage$temporal_all_observed_test[[2L]] == 100L,
    !any(histories$coverage_by_compound$change_comp_observed[histories$coverage_by_compound$endpoint_decade == 1900L]))
  # The final historical predictions must be precisely the frozen modern ones.
  final <- histories$trajectories[histories$trajectories$endpoint_decade == 1910L, ]
  ord <- match(frozen$modern_predictions$canonical_compound, final$canonical_compound)
  stopifnot(isTRUE(all.equal(frozen$modern_predictions$predicted, final$predicted_compositionality[ord])))
  before <- google_longitudinal_sha(model_path); model_time <- file.info(model_path)$mtime
  old_fit <- google_longitudinal_fit
  assign("google_longitudinal_fit", function(...) stop("A verified resume must not retrain"), envir = .GlobalEnv)
  on.exit(assign("google_longitudinal_fit", old_fit, envir = .GlobalEnv), add = TRUE)
  google_longitudinal_run(args)
  stopifnot(before == google_longitudinal_sha(model_path), identical(model_time, file.info(model_path)$mtime))
  # Recreate a missing derived product by reusing the same frozen temporal fit.
  unlink(file.path(args$output_dir, "growth_model.rds"))
  google_longitudinal_run(args)
  stopifnot(before == google_longitudinal_sha(model_path), file.exists(file.path(args$output_dir, "growth_model.rds")))
  bad_config <- checkpoint$config; bad_config$options$feature_set <- "all_static"
  expect_error(google_longitudinal_checkpoint_source_info(checkpoint, bad_config, model_path), "identity changed")
  changed_checkpoint <- checkpoint; changed_checkpoint$frozen$predictors <- "tampered"
  expect_error(google_longitudinal_checkpoint_source_info(changed_checkpoint, checkpoint$config, model_path), "content identity")
  message("Google cumulative temporal R tests passed")
}
main()
