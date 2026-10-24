#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "regression_comparision.R"))
root <- tempfile("google-comparison-")
dir.create(file.path(root, "settings"), recursive = TRUE)
dir.create(file.path(root, "time"), recursive = TRUE)
dir.create(file.path(root, "families"), recursive = TRUE)
on.exit(unlink(root, recursive = TRUE), add = TRUE)

make_rows <- function(setting, timespan = 10000L, features = "all") {
  n <- 15L
  observed <- seq_len(n) / 3
  offset <- switch(setting, Agnostic = 0.1, Aware = 0.4, Pooled = 0.2)
  data.frame(source = "cordeiro100", modifier = paste0("m", seq_len(n)),
             head = paste0("h", seq_len(n)), observed = observed,
             predicted_spearman = observed + offset * sin(seq_len(n)),
             predicted_r2 = observed + offset * cos(seq_len(n)),
             setting = setting, context_mode = "sentence", context_size = 10000L,
             timespan = timespan, features = features, y = "compound")
}

for (setting in c("Agnostic", "Aware", "Pooled")) {
  rows <- make_rows(setting)
  if (setting == "Aware") rows <- rows[nrow(rows):1L, ]
  write.csv(rows, file.path(root, "settings", paste0("predictions_", setting, ".csv")),
            row.names = FALSE)
}
result <- compare_regression_predictions("settings", by = "setting", features = "all",
                                         root = root, n_boot = 100L, n_perm = 100L)
stopifnot(nrow(result$models) == 3L, nrow(result$pairwise) == 3L,
          all(result$models$n_compounds == 15L),
          all(result$pairwise$rho_p_holm >= result$pairwise$rho_p),
          all(result$pairwise$mae_p_holm >= result$pairwise$mae_p))

write.csv(make_rows("Agnostic"), file.path(root, "time", "predictions_pooled.csv"), row.names = FALSE)
write.csv(make_rows("Aware", 10L, "all_static"),
          file.path(root, "time", "predictions_decade.csv"), row.names = FALSE)
time_result <- compare_regression_predictions("time", conditions = list(
  pooled = list(timespan = 10000L, features = "all"),
  decade = list(timespan = 10L, features = "all_static")),
  root = root, n_boot = 100L, n_perm = 100L)
stopifnot(nrow(time_result$pairwise) == 1L,
          identical(time_result$pairwise$left, "pooled"))

for (family in c("constant", "variable_a", "variable_b")) {
  rows <- make_rows(switch(family, constant = "Agnostic",
                           variable_a = "Aware", variable_b = "Pooled"))
  rows$features <- family
  if (family == "constant") rows$predicted_spearman <- rep(mean(rows$observed), nrow(rows))
  write.csv(rows, file.path(root, "families", paste0("predictions_", family, ".csv")),
            row.names = FALSE)
}
family_result <- suppressWarnings(compare_regression_predictions(
  "families", by = "features", root = root, n_boot = 100L, n_perm = 100L))
stopifnot(identical(family_result$undefined_spearman, "constant"),
          is.na(family_result$models$spearman[family_result$models$condition == "constant"]),
          all(is.na(family_result$pairwise$rho_p_holm[
            family_result$pairwise$left == "constant"])),
          !is.na(family_result$pairwise$rho_p_holm[
            family_result$pairwise$left == "variable_a"]),
          all(is.finite(family_result$pairwise$mae_p_holm)))

aware_file <- file.path(root, "settings", "predictions_Aware.csv")
aware <- read.csv(aware_file)
aware$head[1L] <- "different"
write.csv(aware, aware_file, row.names = FALSE)
stopifnot(inherits(try(compare_regression_predictions("settings", by = "setting",
  features = "all", root = root, n_boot = 100L, n_perm = 100L), silent = TRUE), "try-error"))
cat("Google regression comparison tests passed\n")
