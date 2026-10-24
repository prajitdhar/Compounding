#!/usr/bin/env Rscript

# Plot-only trajectory styling. Do not source the scientific fitting drivers:
# their hashes and generated artifacts belong to the completed run's receipt.
longitudinal_trajectory_plot <- function(trajectory, growth) {
  required <- c("end_year", "predicted_compositionality", "compound_id")
  if (!all(required %in% names(trajectory)) || !nrow(trajectory) ||
      any(!is.finite(trajectory$end_year)) ||
      any(!is.finite(trajectory$predicted_compositionality))) {
    stop("Expected nonempty finite saved longitudinal predictions")
  }
  colors <- c("frozen-model predictions" = "#737373",
              "compound growth fits" = "#2878a1", "population fit" = "#c52822")
  labels <- names(colors)[1L]
  plot <- ggplot2::ggplot(trajectory,
    ggplot2::aes(x = end_year, y = predicted_compositionality, group = compound_id)) +
    ggplot2::geom_line(ggplot2::aes(color = "frozen-model predictions"),
                       alpha = 0.3, linewidth = 0.35) +
    ggplot2::geom_point(ggplot2::aes(color = "frozen-model predictions"),
                        alpha = 0.25, size = 0.6, show.legend = FALSE)
  if (identical(growth$status, "fitted")) {
    # Use the saved fit, without refitting or changing its time origin or CI.
    fixed <- data.frame(end_year = seq(min(growth$fitted$end_year),
                                      max(growth$fitted$end_year), length.out = 100L))
    fixed$time_decades <- (fixed$end_year - 9 - growth$origin) / 10
    design <- cbind(1, fixed$time_decades)
    fixed$predicted <- as.numeric(design %*% lme4::fixef(growth$fit))
    uncertainty <- sqrt(rowSums((design %*% as.matrix(stats::vcov(growth$fit))) * design))
    fixed$lower <- fixed$predicted - 1.96 * uncertainty
    fixed$upper <- fixed$predicted + 1.96 * uncertainty
    plot <- plot +
      ggplot2::geom_line(data = growth$fitted,
        ggplot2::aes(y = fitted, color = "compound growth fits"), alpha = 0.2, linewidth = 0.4) +
      ggplot2::geom_ribbon(data = fixed,
        ggplot2::aes(x = end_year, ymin = lower, ymax = upper), inherit.aes = FALSE,
        fill = colors[["population fit"]], alpha = 0.12, show.legend = FALSE) +
      ggplot2::geom_line(data = fixed,
        ggplot2::aes(x = end_year, y = predicted, color = "population fit"),
        inherit.aes = FALSE, linewidth = 1)
    labels <- names(colors)
  }
  plot +
    ggplot2::scale_x_continuous(breaks = sort(unique(trajectory$end_year))) +
    ggplot2::scale_color_manual(values = colors, breaks = labels, name = NULL) +
    ggplot2::labs(x = "Cumulative history through year",
      y = "Predicted compound compositionality (unclipped)",
      title = NULL, subtitle = NULL, caption = NULL) +
    ggplot2::guides(color = ggplot2::guide_legend(nrow = 1,
      override.aes = list(alpha = 1, linewidth = 1))) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "top", legend.title = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      axis.ticks.x = ggplot2::element_line(color = "grey40"),
      panel.grid.minor.x = ggplot2::element_blank())
}

google_longitudinal_plot_sha <- function(path) digest::digest(file = path, algo = "sha256")

google_longitudinal_plot_render <- function(results_dir, output_prefix = NULL) {
  results_dir <- normalizePath(results_dir, mustWork = TRUE)
  receipt_path <- file.path(results_dir, "complete_longitudinal.rds")
  if (!file.exists(receipt_path)) stop("Missing study completion receipt: ", receipt_path)
  receipt <- readRDS(receipt_path)
  expected <- receipt$output_sha256
  if (!identical(receipt$status, "complete") || !is.character(expected) ||
      !length(expected) || is.null(names(expected)) || anyDuplicated(names(expected)) ||
      any(!file.exists(names(expected)))) stop("Incomplete or invalid study completion receipt")
  actual <- setNames(vapply(names(expected), google_longitudinal_plot_sha, character(1)), names(expected))
  if (!identical(expected, actual)) stop("Saved study outputs no longer match their completion receipt")
  inputs <- file.path(results_dir, c("trajectories.csv", "growth_model.rds"))
  if (!all(normalizePath(inputs, mustWork = TRUE) %in%
           normalizePath(names(expected), mustWork = TRUE))) {
    stop("Saved trajectories and growth model must be covered by the completion receipt")
  }
  if (is.null(output_prefix)) output_prefix <- file.path(results_dir, "growth_trajectories_styled")
  parent <- dirname(output_prefix)
  if (!dir.exists(parent) && !dir.create(parent, recursive = TRUE)) stop("Cannot create plot output directory")
  output_prefix <- file.path(normalizePath(parent, mustWork = TRUE), basename(output_prefix))
  outputs <- paste0(output_prefix, c(".png", ".pdf", ".source_info.rds"))
  protected <- c(names(expected), receipt_path, names(receipt$config$input_sha256),
                 names(receipt$config$source_sha256))
  if (any(normalizePath(outputs, mustWork = FALSE) %in%
          normalizePath(protected, mustWork = FALSE))) {
    stop("Plot-only rendering cannot overwrite receipt-tracked study files")
  }
  receipt_sha <- google_longitudinal_plot_sha(receipt_path)
  trajectory <- utils::read.csv(inputs[[1L]], stringsAsFactors = FALSE)
  growth <- readRDS(inputs[[2L]])
  plot <- longitudinal_trajectory_plot(trajectory, growth)
  for (format in c("png", "pdf")) {
    destination <- paste0(output_prefix, ".", format)
    temporary <- tempfile(pattern = ".longitudinal-plot-", tmpdir = parent, fileext = paste0(".", format))
    tryCatch({
      ggplot2::ggsave(temporary, plot, width = 10, height = 6, dpi = 300,
                      device = format, bg = "white")
      if (!file.rename(temporary, destination)) stop("Cannot publish plot: ", destination)
    }, finally = if (file.exists(temporary)) unlink(temporary))
    message("Saved ", destination)
  }
  after <- setNames(vapply(names(expected), google_longitudinal_plot_sha, character(1)), names(expected))
  if (!identical(expected, after) || !identical(receipt_sha, google_longitudinal_plot_sha(receipt_path))) {
    stop("Study changed while rendering; plot source information was not published")
  }
  argv <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  source_info <- list(study_receipt = receipt_path, study_receipt_sha256 = receipt_sha,
    input_sha256 = setNames(vapply(inputs, google_longitudinal_plot_sha, character(1)), inputs),
    output_sha256 = setNames(vapply(outputs[1:2], google_longitudinal_plot_sha, character(1)), outputs[1:2]),
    plot_script_sha256 = if (length(argv)) google_longitudinal_plot_sha(argv[[1L]]) else NA_character_,
    rendered_utc = format(Sys.time(), tz = "UTC", usetz = TRUE), refitted = FALSE,
    R = R.version.string, ggplot2 = as.character(utils::packageVersion("ggplot2")))
  temporary <- tempfile(pattern = ".longitudinal-plot-source_info-", tmpdir = parent)
  tryCatch({
    saveRDS(source_info, temporary)
    if (!file.rename(temporary, outputs[[3L]])) stop("Cannot publish plot source information")
  }, finally = if (file.exists(temporary)) unlink(temporary))
  invisible(outputs)
}

if (sys.nframe() == 0L) {
  packages <- c("ggplot2", "lme4", "optparse", "digest")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
  parser <- optparse::OptionParser(option_list = list(
    optparse::make_option("--results-dir", type = "character", help = "Completed longitudinal results directory"),
    optparse::make_option("--output-prefix", type = "character", default = NULL,
      help = "Output path without extension [default: RESULTS/growth_trajectories_styled]")))
  args <- optparse::parse_args(parser)
  if (is.null(args$`results-dir`)) stop("--results-dir is required")
  google_longitudinal_plot_render(args$`results-dir`, args$`output-prefix`)
}
