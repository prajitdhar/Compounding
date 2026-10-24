#!/usr/bin/env Rscript

arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
test_dir <- dirname(normalizePath(sub("^--file=", "", arg[[1]])))
regression_dir <- normalizePath(file.path(test_dir, "..", "modified_regressions"))
source(file.path(regression_dir, "batch_status.R"))
source(file.path(regression_dir, "coha_v3_regression_data.R"))

run_test <- function() {
  test_data <- tempfile("google-batch-status-")
  dir.create(test_data)
  on.exit(unlink(test_data, recursive = TRUE), add = TRUE)
  input_root <- file.path(test_data, "inputs")
  out <- file.path(test_data, "results")
  dir.create(file.path(input_root, "local5"), recursive = TRUE)
  dir.create(out)
  row <- list(task_id = "tagged-task", tagging = "Tagged", vocab = "5000", weight = "PPMI",
              setting = "Agnostic", timespan = "10000", cutoff = "0", input_root = input_root,
              out = out, feature_sets = "all_no_setting,representation_only,all,all_static,temporal",
              targets = "compound,head,modifier", frequency_family = "freq", productivity_family = "prod",
              folds = "5", repeats = "10", alpha_count = "10", lambda_count = "200",
              permutations = "5", train_variant = "modified", test_variant = "modified")
  tasks <- as.data.frame(row, stringsAsFactors = FALSE)
  task <- google_v3_status_task(row)
  code_md5 <- unname(tools::md5sum(file.path(regression_dir,
    c("regression_exps.r", "coha_v3_regression_data.R"))))

  inspect <- function() google_v3_status_run(tasks, regression_dir)
  missing <- inspect()
  stopifnot(missing$status == "incompatible", grepl("Missing inputs", missing$detail))

  input_paths <- google_v3_status_input_paths(task)
  writeLines("test data CSV content, not read as model data", input_paths)
  pending <- inspect()
  stopifnot(pending$status == "pending", pending$patterns == "nn,an", pending$completed_patterns == "")
  config <- google_v3_status_config(task, unname(tools::md5sum(input_paths)), code_md5)
  stopifnot(identical(config$feature_sets, c("all_no_setting", "representation_only", "all")),
            identical(config$targets, c("compound", "head", "modifier")),
            is.integer(config$folds), is.integer(config$lambda_count))

  output_types <- c("rsquared", "coefimp", "permimp", "familyimp", "predictions", "cvgrid", "cvfolds")
  write_complete <- function(pattern, config_value = config, task_value = task) {
    suffix <- google_v3_status_suffix(task_value, pattern)
    paths <- file.path(out, paste0(output_types, "_", suffix, ".csv"))
    invisible(file.create(paths))
    marker <- file.path(out, paste0("complete_", suffix, ".rds"))
    saveRDS(config_value, marker)
    list(marker = marker, paths = paths)
  }
  nn <- write_complete("nn")
  partial <- inspect()
  stopifnot(partial$status == "pending", partial$completed_patterns == "nn")
  an <- write_complete("an")
  completed <- inspect()
  stopifnot(completed$status == "completed", completed$completed_patterns == "nn,an")

  unlink(an$paths[[7]])
  output_missing <- inspect()
  stopifnot(output_missing$status == "pending", output_missing$completed_patterns == "nn",
            grepl("missing outputs: cvfolds", output_missing$detail, fixed = TRUE))
  invisible(file.create(an$paths[[7]]))

  changed <- config
  changed$targets <- c("compound", "modifier", "head")
  saveRDS(changed, an$marker)
  mismatch <- inspect()
  stopifnot(mismatch$status == "incompatible", mismatch$completed_patterns == "nn",
            grepl("configuration mismatch: targets", mismatch$detail, fixed = TRUE))
  saveRDS(config, an$marker)

  writeLines("not an RDS file", an$marker)
  corrupt <- inspect()
  stopifnot(corrupt$status == "incompatible", grepl("unreadable receipt", corrupt$detail))
  saveRDS(config, an$marker)

  writeLines("changed input", input_paths)
  stale <- inspect()
  stopifnot(stale$status == "incompatible", grepl("input_md5", stale$detail))
  writeLines("test data CSV content, not read as model data", input_paths)

  untagged_row <- row
  untagged_row$task_id <- "untagged-task"
  untagged_row$tagging <- "UnTagged"
  untagged_row$timespan <- "10"
  untagged_row$cutoff <- "1000"
  untagged <- google_v3_status_task(untagged_row)
  untagged_inputs <- google_v3_status_input_paths(untagged)
  for (path in untagged_inputs) writeLines("input", path)
  untagged_config <- google_v3_status_config(untagged, unname(tools::md5sum(untagged_inputs)), code_md5)
  stopifnot(length(untagged_config$input_md5) == 2L,
            "all_static" %in% untagged_config$feature_sets,
            "temporal" %in% untagged_config$feature_sets)
  write_complete("all", untagged_config, untagged)
  tasks <- rbind(tasks, as.data.frame(untagged_row, stringsAsFactors = FALSE))
  all_status <- inspect()
  stopifnot(identical(all_status$status, c("completed", "completed")),
            identical(all_status$completed_patterns, c("nn,an", "all")))

  task_tsv <- file.path(test_data, "tasks.tsv")
  output_tsv <- file.path(test_data, "status.tsv")
  write.table(tasks, task_tsv, sep = "\t", row.names = FALSE)
  google_v3_status_main(c("--tasks-tsv", task_tsv, "--output-tsv", output_tsv), regression_dir)
  roundtrip <- read.delim(output_tsv, colClasses = "character", check.names = FALSE)
  stopifnot(identical(roundtrip, all_status))
  empty <- google_v3_status_run(tasks[FALSE, , drop = FALSE], regression_dir)
  stopifnot(nrow(empty) == 0L, identical(names(empty), names(all_status)))
  duplicate <- tasks
  duplicate$task_id[[2]] <- duplicate$task_id[[1]]
  rejected <- tryCatch(google_v3_status_run(duplicate, regression_dir), error = identity)
  stopifnot(inherits(rejected, "error"))
  cat("batch_status checkpoint tests passed\n")
}

run_test()
