#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)

root <- tempfile("google-v3-summary-")
dir.create(root)
dir.create(file.path(root, "input"))
dir.create(file.path(root, "output"))
on.exit(unlink(root, recursive = TRUE), add = TRUE)

rows <- expand.grid(context_mode = c("local5", "sentence"), context_size = c(1000, 10000),
                    timespan = c(10000, 10, 20), features = c("all", "cosine_sim"),
                    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
rows$corpus <- "google"
rows$tag <- "UnTagged"
rows$ppmi <- "PPMI"
rows$setting <- "Agnostic"
rows$pattern <- "all"
rows$cutoff <- 0
rows$impute <- "fold_median"
rows$y <- "compound"
rows$TestRsquared <- ifelse(rows$features == "all", 0.5, 0.4)
rows$TestSpearman <- ifelse(rows$features == "all", 0.7, 0.6)
rows$TestAdjRsquared <- rows$TestRsquared - 0.1
rows$CVTrainRsquared <- ifelse(rows$features == "all", 0.35, 0.25)
rows$CVTrainSpearman <- ifelse(rows$features == "all", 0.55, 0.45)
rows$frequency_family <- "freq"
rows$productivity_family <- "prod"
alternative <- rows[rows$context_mode == "local5" & rows$context_size == 1000 &
                    rows$timespan == 10000 & rows$features == "all", , drop = FALSE]
alternative$frequency_family <- "tf"
alternative$TestSpearman <- 0.65
rows <- rbind(rows, alternative)
write.csv(rows, file.path(root, "input", "rsquared_test.csv"), row.names = FALSE)
unspecified <- rows[1, setdiff(names(rows), c("CVTrainRsquared", "CVTrainSpearman")), drop = FALSE]
unspecified$context_mode <- "unspecified"
unspecified$frequency_family <- "unspecified"
unspecified$productivity_family <- "unspecified"
write.csv(unspecified, file.path(root, "input", "rsquared_unspecified.csv"), row.names = FALSE)

status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(file.path(regression_root, "modified_regressions", "regression_statistics.R")), env = c(
  paste0("COMPOUNDING_REGRESSION_PATH=", file.path(root, "input")),
  paste0("COMPOUNDING_SUMMARY_PATH=", file.path(root, "output"))))
stopifnot(status == 0)

sync <- read.csv(file.path(root, "output", "summary_sync_feature_set_consistency.csv"))
dia <- read.csv(file.path(root, "output", "summary_diachronic_feature_set_consistency.csv"))
stopifnot(nrow(sync) == 10, nrow(dia) == 16)
stopifnot(setequal(unique(sync$frequency_family), c("freq", "tf", "unspecified")))
stopifnot(setequal(unique(dia$timespan), c(10, 20)))
stopifnot(setequal(unique(sync$context_mode), c("local5", "sentence", "unspecified")))
stopifnot(setequal(unique(sync$context_size), c(1000, 10000)))
stopifnot(all(sync$mean_cv_train_spearman[sync$features == "all" &
                                            sync$context_mode != "unspecified"] == 0.55))
stopifnot(is.na(sync$mean_cv_train_spearman[sync$context_mode == "unspecified"]))
cat("Google v3 regression summary tests passed\n")
