#!/usr/bin/env Rscript
# Plot-only regression tests. Synthetic data and receipt files stay in a
# temporary directory; no scientific drivers or production results are edited.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
test_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
plot_script <- normalizePath(file.path(test_dir, "..", "longitudinal", "plot_longitudinal.R"))
for (package in c("ggplot2", "lme4", "digest")) {
  if (!requireNamespace(package, quietly = TRUE)) stop(package, " required for plot-only tests")
}
# Sourcing must define helpers without interpreting this test's command line or
# rendering a figure as a side effect.
source(plot_script)

main <- function() {
  root <- tempfile("google-longitudinal-plot-test-")
  dir.create(root)
  on.exit(unlink(root, recursive = TRUE), add = TRUE)
  trajectory <- expand.grid(compound_id = paste0("compound_", seq_len(8L)),
    endpoint_decade = seq(1820L, 2010L, by = 10L), KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE)
  trajectory$end_year <- trajectory$endpoint_decade + 9L
  trajectory$time_decades <- (trajectory$endpoint_decade - 1820L) / 10
  index <- match(trajectory$compound_id, unique(trajectory$compound_id))
  trajectory$predicted_compositionality <- 3 + sin(index) +
    (-0.05 + cos(index) / 30) * trajectory$time_decades +
    sin(seq_len(nrow(trajectory)) * 1.7) / 10
  # Guard against accidentally clipping model projections to a rating scale.
  trajectory$predicted_compositionality[c(1L, nrow(trajectory))] <- c(-1, 8)
  fit <- suppressMessages(suppressWarnings(lme4::lmer(
    predicted_compositionality ~ time_decades + (1 + time_decades | compound_id),
    data = trajectory, REML = TRUE)))
  fitted_rows <- trajectory
  fitted_rows$fitted <- as.numeric(stats::predict(fit))
  growth <- list(status = "fitted", fit = fit, fitted = fitted_rows, origin = 1820L)
  plot <- longitudinal_trajectory_plot(trajectory, growth)
  stopifnot(inherits(plot, "ggplot"),
    identical(plot$labels$x, "Cumulative history through year"),
    identical(plot$labels$y, "Predicted compound compositionality (unclipped)"),
    is.null(plot$labels$title), is.null(plot$labels$subtitle), is.null(plot$labels$caption))
  x_scale <- plot$scales$get_scales("x")
  stopifnot(isTRUE(all.equal(as.numeric(x_scale$breaks),
    sort(unique(trajectory$end_year)))))
  built <- ggplot2::ggplot_build(plot)
  colour <- built$plot$scales$get_scales("colour")
  labels <- c("frozen-model predictions", "compound growth fits", "population fit")
  stopifnot(identical(as.character(colour$get_breaks()), labels),
    identical(unname(colour$map(labels)), c("#737373", "#2878a1", "#c52822")),
    !any(grepl("[,;]", labels)))
  # Both line and point layers preserve every original unclipped projection.
  raw_layers <- which(vapply(plot$layers, function(layer)
    inherits(layer$geom, "GeomPoint"), logical(1)))
  stopifnot(length(raw_layers) == 1L,
    identical(sort(built$data[[raw_layers]]$y), sort(trajectory$predicted_compositionality)))
  ribbon <- which(vapply(plot$layers, function(layer)
    inherits(layer$geom, "GeomRibbon"), logical(1)))
  stopifnot(length(ribbon) == 1L)
  interval <- built$data[[ribbon]]
  design <- cbind(1, (interval$x - 9 - growth$origin) / 10)
  predicted <- as.numeric(design %*% lme4::fixef(fit))
  se <- sqrt(rowSums((design %*% as.matrix(stats::vcov(fit))) * design))
  stopifnot(isTRUE(all.equal(interval$ymin, predicted - 1.96 * se)),
    isTRUE(all.equal(interval$ymax, predicted + 1.96 * se)))
  invisible(ggplot2::ggplotGrob(plot))
  # A valid but unfitted growth object still renders the frozen predictions.
  unfitted <- longitudinal_trajectory_plot(trajectory,
    list(status = "too_few_eligible_compounds_or_endpoints"))
  invisible(ggplot2::ggplot_build(unfitted))

  results <- file.path(root, "results")
  dir.create(results)
  write.csv(trajectory, file.path(results, "trajectories.csv"), row.names = FALSE)
  saveRDS(growth, file.path(results, "growth_model.rds"))
  # The canonical receipt has exactly the original scientific output names.
  names <- c("frozen_modern_model.rds", "trajectories.csv", "coverage.csv",
    "coverage_by_compound.csv", "modern_predictions.csv", "modern_performance.csv",
    "coefficients.csv", "cv_grid.csv", "cv_folds.csv", "cv_predictions.csv",
    "frozen_preprocessing.rds", "growth_model.rds", "growth_model_summary.txt",
    "growth_slope.csv", "growth_fitted.csv", "growth_trajectories.png")
  for (name in setdiff(names, c("trajectories.csv", "growth_model.rds"))) {
    writeLines(paste("Immutable test data", name), file.path(results, name))
  }
  paths <- file.path(results, names)
  hash <- function(path) digest::digest(file = path, algo = "sha256")
  hashes <- setNames(vapply(paths, hash, character(1)), paths)
  receipt_path <- file.path(results, "complete_longitudinal.rds")
  saveRDS(list(status = "complete", output_sha256 = hashes,
    config = list(options = list(cutoff = 0L))), receipt_path)
  receipt_hash <- hash(receipt_path)
  run_cli <- function(prefix) {
    output <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"),
      c(shQuote(plot_script), "--results-dir", shQuote(results),
        "--output-prefix", shQuote(prefix)),
      stdout = TRUE, stderr = TRUE, env = "TZ=Europe/Berlin"))
    list(status = if (is.null(attr(output, "status"))) 0L else attr(output, "status"), output = output)
  }
  prefix <- file.path(results, "growth_trajectories_styled")
  rendered <- run_cli(prefix)
  if (rendered$status != 0L) stop("Plot CLI failed: ", paste(rendered$output, collapse = "\n"))
  stopifnot(all(file.exists(paste0(prefix, c(".png", ".pdf")))) ,
    all(file.info(paste0(prefix, c(".png", ".pdf")))$size > 0),
    identical(hashes, setNames(vapply(paths, hash, character(1)), paths)),
    identical(receipt_hash, hash(receipt_path)))
  # Never overwrite the original figure recorded in the scientific receipt.
  overwrite <- run_cli(file.path(results, "growth_trajectories"))
  stopifnot(overwrite$status != 0L,
    identical(hashes, setNames(vapply(paths, hash, character(1)), paths)))
  # A changed result must be rejected before writing a new styled figure.
  altered <- trajectory
  altered$predicted_compositionality[[1L]] <- 99
  write.csv(altered, file.path(results, "trajectories.csv"), row.names = FALSE)
  bad_prefix <- file.path(results, "tampered")
  tampered <- run_cli(bad_prefix)
  stopifnot(tampered$status != 0L,
    !any(file.exists(paste0(bad_prefix, c(".png", ".pdf")))))
  message("Google longitudinal plot-only tests passed")
}
main()
