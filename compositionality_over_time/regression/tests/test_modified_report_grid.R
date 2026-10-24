script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "modified_regressions", "report_grid_filter.R"))

rows <- expand.grid(context_mode=c("sentence", "local5"), context_size=c(3000L, 5000L, 30000L),
                    ppmi=c("RAW", "PPMI"), setting="Agnostic", tag="Tagged",
                    pattern=c("nn", "an"), timespan=c(10000L, 10L, 20L, 50L), cutoff=c(0L, 10L),
                    y=c("compound", "head", "modifier"), train_variant="modified",
                    test_variant=c("modified", "original"), stringsAsFactors=FALSE)
rows$predicted_spearman <- seq_len(nrow(rows)) / nrow(rows)
stopifnot(identical(coha_v3_report_grid(rows), rows))
Sys.setenv(GRID_MODES="sentence local5", GRID_VOCABS="3000 5000 10000",
           GRID_WEIGHTINGS="RAW PPMI", GRID_SETTINGS="Agnostic", GRID_TAGGINGS="Tagged",
           GRID_TIMESPANS="10000,10,50", GRID_CUTOFFS="0 10", GRID_TARGETS="compound head modifier",
           GRID_TRAIN_VARIANT="modified", GRID_TEST_VARIANT="modified")
expected <- rows[rows$context_size != 30000L & rows$timespan != 20L & rows$test_variant == "modified", ]
actual <- coha_v3_report_grid(rows)
stopifnot(identical(actual, expected), nrow(actual) == 288L,
          identical(sort(unique(actual$pattern)), c("an", "nn")),
          identical(sort(unique(actual$context_mode)), c("local5", "sentence")))
cat("Revised report grid excludes removed configurations and retains both tagged patterns\n")
