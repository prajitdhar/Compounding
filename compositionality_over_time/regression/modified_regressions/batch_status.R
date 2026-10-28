#!/usr/bin/env Rscript

# Read-only checkpoint inspection for the fine-grained regression batch planner.
# Keep this separate from the fitter: its recorded code MD5s must not change
# when the scheduling policy changes. Only the requested TSV output is written.

google_v3_status_script_dir <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg)) dirname(normalizePath(sub("^--file=", "", arg[[1]]))) else getwd()
}

google_v3_status_words <- function(value, label) {
  value <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  if (!length(value) || anyNA(value) || any(!nzchar(value)) || anyDuplicated(value)) {
    stop("Invalid ", label)
  }
  value
}

google_v3_status_integer <- function(value, label, minimum = 0L) {
  if (length(value) != 1L || is.na(value) || !grepl("^[0-9]+$", value)) {
    stop("Invalid ", label)
  }
  result <- suppressWarnings(as.integer(value))
  if (is.na(result) || result < minimum) stop("Invalid ", label)
  result
}

google_v3_status_required <- c(
  "task_id", "tagging", "vocab", "weight", "setting", "timespan", "cutoff",
  "input_root", "out", "feature_sets", "targets", "frequency_family",
  "productivity_family", "folds", "repeats", "alpha_count", "lambda_count",
  "permutations", "train_variant", "test_variant"
)

google_v3_status_task <- function(row) {
  if (!all(google_v3_status_required %in% names(row))) {
    stop("Missing task fields: ", paste(setdiff(google_v3_status_required, names(row)), collapse = ","))
  }
  for (name in google_v3_status_required) {
    if (length(row[[name]]) != 1L || is.na(row[[name]]) || !nzchar(row[[name]])) {
      stop("Empty task field: ", name)
    }
  }
  row$context_mode <- if (is.null(row$context_mode)) "local5" else row$context_mode
  row$corpus <- if (is.null(row$corpus)) "google" else row$corpus
  coha_v3_validate_study(row$corpus, row$context_mode)
  choices <- list(tagging = c("UnTagged", "Tagged"), weight = c("RAW", "PPMI"),
                  setting = c("Agnostic", "Aware", "Pooled"),
                  frequency_family = c("freq", "tf", "log_freq"),
                  productivity_family = c("prod", "family_size"),
                  train_variant = c("original", "modified"),
                  test_variant = c("original", "modified"))
  for (name in names(choices)) {
    if (!row[[name]] %in% choices[[name]]) stop("Invalid ", name)
  }
  minimum <- c(vocab = 1L, timespan = 1L, cutoff = 0L, folds = 2L,
               repeats = 1L, alpha_count = 2L, lambda_count = 2L, permutations = 1L)
  for (name in names(minimum)) {
    row[[name]] <- google_v3_status_integer(row[[name]], name, minimum[[name]])
  }
  row$targets <- google_v3_status_words(row$targets, "targets")
  if (any(!row$targets %in% c("compound", "modifier", "head"))) stop("Invalid targets")
  row$feature_sets <- google_v3_status_words(row$feature_sets, "feature_sets")
  allowed <- c("all", "all_static", "all_no_setting", "representation_only",
               names(coha_v3_feature_groups), unlist(coha_v3_feature_groups, use.names = FALSE))
  if (any(!row$feature_sets %in% allowed)) stop("Invalid feature_sets")
  row
}

google_v3_status_suffix <- function(task, pattern) {
  suffix <- sprintf("%s_v3_%s_ctx%s_%s_%s_%s_%s_%s_%s_%s_%s",
                    task$corpus, task$context_mode, task$vocab, task$tagging, task$weight,
                    task$setting, task$timespan, task$cutoff, pattern,
                    task$frequency_family, task$productivity_family)
  if (task$train_variant != "original" || task$test_variant != "original") {
    suffix <- paste0(suffix, "_train", task$train_variant, "_test", task$test_variant)
  }
  suffix
}

google_v3_status_input_paths <- function(task) {
  paths <- coha_v3_feature_path(task$input_root, task$context_mode, task$vocab,
                               "features", task$setting, task$weight, task$tagging,
                               task$timespan, task$cutoff)
  if (task$timespan != 10000L) paths <- c(paths,
    coha_v3_feature_path(task$input_root, task$context_mode, task$vocab,
                         "temporal", task$setting, task$weight, task$tagging,
                         task$timespan, task$cutoff))
  paths
}

google_v3_status_config <- function(task, input_md5, code_md5) {
  available <- if (task$timespan == 10000L) {
    setdiff(task$feature_sets, c("all_static", "temporal", coha_v3_feature_groups$temporal))
  } else task$feature_sets
  # Field order and integer/vector types match the fitter's saveRDS() exactly.
  list(input_md5 = unname(input_md5), code_md5 = unname(code_md5),
       corpus = task$corpus, context_mode = task$context_mode, targets = task$targets,
       feature_sets = available, folds = task$folds, repeats = task$repeats,
       train_variant = task$train_variant, test_variant = task$test_variant,
       frequency_family = task$frequency_family,
       productivity_family = task$productivity_family,
       alpha_count = task$alpha_count, lambda_count = task$lambda_count,
       permutations = task$permutations)
}

google_v3_status_clean_detail <- function(value) {
  gsub("[\t\r\n]", " ", value)
}

google_v3_status_one <- function(task, code_md5, md5_cache = new.env(parent = emptyenv())) {
  task_id <- as.character(task$task_id)
  tryCatch({
    task <- google_v3_status_task(task)
    patterns <- if (task$tagging == "Tagged") c("nn", "an") else "all"
    input_paths <- google_v3_status_input_paths(task)
    missing <- input_paths[!file.exists(input_paths)]
    if (length(missing)) stop("Missing inputs: ", paste(missing, collapse = ", "))
    input_md5 <- vapply(input_paths, function(path) {
      if (!exists(path, envir = md5_cache, inherits = FALSE)) {
        assign(path, unname(tools::md5sum(path)), envir = md5_cache)
      }
      get(path, envir = md5_cache, inherits = FALSE)
    }, character(1), USE.NAMES = FALSE)
    if (anyNA(input_md5) || anyNA(code_md5)) stop("Cannot hash input or fitter code")
    config <- google_v3_status_config(task, input_md5, code_md5)
    statuses <- setNames(rep("pending", length(patterns)), patterns)
    details <- setNames(rep("", length(patterns)), patterns)
    for (pattern in patterns) {
      suffix <- google_v3_status_suffix(task, pattern)
      marker <- file.path(task$out, paste0("complete_", suffix, ".rds"))
      if (!file.exists(marker)) {
        details[[pattern]] <- "missing completion receipt"
        next
      }
      receipt <- tryCatch(readRDS(marker), error = function(error) error)
      if (inherits(receipt, "error")) {
        statuses[[pattern]] <- "incompatible"
        details[[pattern]] <- paste0("unreadable receipt: ", conditionMessage(receipt))
        next
      }
      if (!identical(receipt, config)) {
        statuses[[pattern]] <- "incompatible"
        changed <- if (is.list(receipt)) {
          fields <- union(names(config), names(receipt))
          fields[!vapply(fields, function(field) identical(config[[field]], receipt[[field]]), logical(1))]
        } else "receipt type"
        if (!length(changed)) changed <- "receipt field order/type"
        details[[pattern]] <- paste0("configuration mismatch: ", paste(changed, collapse = ","))
        next
      }
      output_types <- c("rsquared", "coefimp", "permimp", "familyimp", "predictions", "cvgrid", "cvfolds")
      paths <- file.path(task$out, paste0(output_types, "_", suffix, ".csv"))
      missing_outputs <- output_types[!file.exists(paths)]
      if (length(missing_outputs)) {
        details[[pattern]] <- paste0("missing outputs: ", paste(missing_outputs, collapse = ","))
      } else {
        statuses[[pattern]] <- "completed"
        details[[pattern]] <- "receipt and all seven outputs match"
      }
    }
    status <- if (any(statuses == "incompatible")) "incompatible" else
      if (all(statuses == "completed")) "completed" else "pending"
    data.frame(task_id = task_id, status = status,
               completed_patterns = paste(patterns[statuses == "completed"], collapse = ","),
               patterns = paste(patterns, collapse = ","),
               detail = google_v3_status_clean_detail(paste(paste0(patterns, ": ", details), collapse = "; ")),
               stringsAsFactors = FALSE)
  }, error = function(error) {
    data.frame(task_id = task_id, status = "incompatible", completed_patterns = "", patterns = "",
               detail = google_v3_status_clean_detail(conditionMessage(error)), stringsAsFactors = FALSE)
  })
}

google_v3_status_run <- function(tasks, script_dir) {
  if (!all(google_v3_status_required %in% names(tasks))) {
    stop("Task TSV is missing fields: ", paste(setdiff(google_v3_status_required, names(tasks)), collapse = ","))
  }
  if (anyNA(tasks$task_id) || any(!nzchar(tasks$task_id)) || anyDuplicated(tasks$task_id)) {
    stop("Task IDs must be nonempty and unique")
  }
  helper <- file.path(script_dir, "coha_v3_regression_data.R")
  source(helper)
  code_paths <- file.path(script_dir, c("regression_exps.r", "coha_v3_regression_data.R"))
  if (!all(file.exists(code_paths))) stop("Missing original fitter code")
  code_md5 <- unname(tools::md5sum(code_paths))
  md5_cache <- new.env(parent = emptyenv())
  if (!nrow(tasks)) return(data.frame(task_id = character(), status = character(),
    completed_patterns = character(), patterns = character(), detail = character()))
  do.call(rbind, lapply(seq_len(nrow(tasks)), function(index) {
    google_v3_status_one(as.list(tasks[index, , drop = FALSE]), code_md5, md5_cache)
  }))
}

google_v3_status_main <- function(args = commandArgs(TRUE), script_dir = google_v3_status_script_dir()) {
  if (length(args) == 1L && args[[1]] %in% c("-h", "--help")) {
    cat("Usage: Rscript batch_status.R --tasks-tsv PATH [--output-tsv PATH]\n")
    return(invisible(NULL))
  }
  opts <- list()
  while (length(args)) {
    if (length(args) < 2L || !args[[1]] %in% c("--tasks-tsv", "--output-tsv") ||
        !is.null(opts[[args[[1]]]])) stop("Invalid arguments; use --help")
    opts[[args[[1]]]] <- args[[2]]
    args <- args[-c(1L, 2L)]
  }
  if (is.null(opts[["--tasks-tsv"]])) stop("--tasks-tsv is required")
  tasks <- read.delim(opts[["--tasks-tsv"]], sep = "\t", quote = '"', comment.char = "",
                      colClasses = "character", na.strings = character(), check.names = FALSE)
  result <- google_v3_status_run(tasks, script_dir)
  destination <- if (is.null(opts[["--output-tsv"]])) stdout() else opts[["--output-tsv"]]
  write.table(result, destination, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
  invisible(result)
}

if (sys.nframe() == 0L) google_v3_status_main()
