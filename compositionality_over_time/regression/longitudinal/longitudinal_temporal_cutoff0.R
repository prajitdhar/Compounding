#!/usr/bin/env Rscript

# Cutoff-zero configuration of the cumulative temporal model.
google_longitudinal_cutoff0_script <- local({
  sources <- Filter(Negate(is.null), lapply(sys.frames(), function(x) x$ofile))
  own <- sources[grepl("longitudinal_temporal_cutoff0\\.R$", unlist(sources))]
  argv <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  path <- if (length(own)) tail(own, 1L)[[1L]] else if (length(argv)) argv[[1L]] else
    file.path(getwd(), "longitudinal_temporal_cutoff0.R")
  normalizePath(path, mustWork = TRUE)
})
google_longitudinal_cutoff0_temporal_script <- file.path(dirname(google_longitudinal_cutoff0_script),
                                                       "longitudinal_temporal.R")
source(google_longitudinal_cutoff0_temporal_script)
google_longitudinal_script <- google_longitudinal_cutoff0_script
google_longitudinal_cutoff0_temporal_sha <- google_longitudinal_sha(google_longitudinal_cutoff0_temporal_script)
google_longitudinal_cutoff0_base_defaults <- google_longitudinal_defaults
google_longitudinal_cutoff0_base_input <- google_longitudinal_input
google_longitudinal_cutoff0_base_sources <- google_longitudinal_temporal_sources

google_longitudinal_defaults <- function() {
  args <- google_longitudinal_cutoff0_base_defaults()
  args$cutoff <- 0L
  args$cumulative_root <- "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_0_cumulative_lag10/features_cumulative"
  args$output_dir <- "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_0_cumulative_lag10/results"
  args
}

google_longitudinal_cutoff0_check_sourced_driver <- function() {
  if (!identical(google_longitudinal_cutoff0_temporal_sha,
                 google_longitudinal_sha(google_longitudinal_cutoff0_temporal_script))) {
    stop("Sourced temporal longitudinal driver changed; restart with unchanged source")
  }
  invisible(TRUE)
}

google_longitudinal_options <- function(args) {
  # Apply common option validation and the cutoff-zero configuration guard.
  args <- google_longitudinal_temporal_base_options(args)
  required <- list(context_mode = "local5", context_size = 30000L, setting = "Aware",
    ppmi = "PPMI", tagging = "UnTagged", cutoff = 0L, time_step = 10L,
    pattern = "all", target = "compound", train_variant = "modified", test_variant = "modified")
  for (name in names(required)) {
    google_longitudinal_same(args[[name]], required[[name]],
      paste0("Cutoff-zero cumulative temporal experiment is restricted to the requested configuration: ", name))
  }
  if (!args$feature_set %in% c("all", "all_static", "representation_only", "temporal")) {
    stop("Temporal feature set must be all, all_static, representation_only, or temporal")
  }
  if (!identical(google_longitudinal_base_sha, google_longitudinal_sha(google_longitudinal_base_script))) {
    stop("Sourced base longitudinal driver changed; restart with unchanged source")
  }
  google_longitudinal_cutoff0_check_sourced_driver()
  args$temporal_policy <- google_longitudinal_temporal_policy
  args$temporal_lag <- 10L
  args$base_driver_sha256 <- google_longitudinal_base_sha
  args$temporal_driver_sha256 <- google_longitudinal_cutoff0_temporal_sha
  args
}

google_longitudinal_temporal_sources <- function(material) {
  # Validate the common generators and the cutoff-zero extension identity.
  paths <- google_longitudinal_cutoff0_base_sources(material)
  records <- material$cutoff_extension_software$code_sha256
  name <- "cumulative_temporal_cutoff0.py"
  shared <- c("source_info.py", "io.py", "progress.py", "partitioned_parquet.py")
  if (!setequal(names(records), c(name, shared)) || !is.character(records[[name]]) ||
      length(records[[name]]) != 1L || !grepl("^[0-9a-f]{64}$", records[[name]])) {
    stop("Missing or incompatible cutoff-zero generator source identity")
  }
  for (item in shared) {
    if (!identical(records[[item]], material$software$code_sha256[[item]])) {
      stop("Cutoff-zero shared source identity differs: ", item)
    }
  }
  directory <- normalizePath(file.path(dirname(google_longitudinal_script), "..", ".."))
  path <- file.path(directory, name)
  if (!file.exists(path) || !identical(google_longitudinal_sha(path), records[[name]])) {
    stop("Cutoff-zero generator source identity changed: ", name)
  }
  unique(c(paths, normalizePath(path)))
}

google_longitudinal_input <- function(args) {
  google_longitudinal_cutoff0_check_sourced_driver()
  input <- google_longitudinal_cutoff0_base_input(args)
  google_longitudinal_cutoff0_check_sourced_driver()
  # The runner hashes and rechecks artifacts after fitting. Include
  # the sourced temporal driver in addition to the static driver, helper,
  # extension driver and all nine Python generator sources.
  input$artifacts <- unique(c(input$artifacts, google_longitudinal_cutoff0_temporal_script))
  input
}

if (sys.nframe() == 0L) google_longitudinal_run(google_longitudinal_parse())
