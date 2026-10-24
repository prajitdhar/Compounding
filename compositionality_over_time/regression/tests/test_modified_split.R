#!/usr/bin/env Rscript
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "modified_regressions", "coha_v3_regression_data.R"))
ratings <- do.call(rbind, lapply(c("reddy90", "cordeiro90", "cordeiro100"), function(name) {
  filename <- c(reddy90 = "reddy_90.txt", cordeiro90 = "cordeiro_90.txt",
                cordeiro100 = "cordeiro_100.txt")[[name]]
  rows <- read.delim(file.path(Sys.getenv("TEST_RATINGS_DIR", unset = file.path(regression_root, "..", "tests", "test_data", "ratings")), filename))
  rows$source <- name
  rows$is_adj <- as.logical(rows$is_adj)
  rows$is_original <- as.logical(rows$is_original)
  rows
}))
# Feature values distinguish POS alternatives so a wrong selection is visible.
ratings[["sim_cpf_beta:0"]] <- seq_len(nrow(ratings))
noun <- ratings
noun$modifier <- paste0(noun$modifier, "_NOUN")
noun$head <- paste0(noun$head, "_NOUN")
mixed <- ratings
mixed$modifier <- paste0(mixed$modifier, ifelse(mixed$is_adj, "_ADJ", "_NOUN"))
mixed$head <- paste0(mixed$head, "_NOUN")
mixed[["sim_cpf_beta:0"]][mixed$is_adj] <- -mixed[["sim_cpf_beta:0"]][mixed$is_adj]
tagged <- unique(rbind(noun, mixed))
for (pattern in c("all", "nn", "an")) {
  tag <- if (pattern == "all") "UnTagged" else "Tagged"
  df <- if (tag == "Tagged") tagged else ratings
  original <- coha_v3_original_split(df, tag, pattern)
  for (train_variant in c("original", "modified")) {
    for (test_variant in c("original", "modified")) {
      split <- coha_v3_variant_split(df, train_variant, test_variant, tag, pattern)
      stopifnot(nrow(split$train) == 180L, nrow(split$test) == 100L)
      stopifnot(sum(!split$train$is_original) == if (train_variant == "modified") 5L else 0L)
      stopifnot(sum(!split$test$is_original) == if (test_variant == "modified") 2L else 0L)
      stopifnot(identical(split$test$canonical_compound, original$test$compound))
      for (column in c("avgModifier", "avgHead", "compositionality", "source", "is_adj")) {
        stopifnot(identical(split$train[[column]], original$train[[column]]))
        stopifnot(identical(split$test[[column]], original$test[[column]]))
      }
      if (pattern == "nn") stopifnot(all(grepl("_NOUN$", split$test$modifier)))
      if (pattern == "an") {
        stopifnot(all(grepl("_ADJ$", split$test$modifier[split$test$is_adj])))
        stopifnot(all(split$test[["sim_cpf_beta:0"]][split$test$is_adj] < 0))
      }
    }
  }
  broken <- df[df$compound != "baby_blue", ]
  stopifnot(inherits(try(coha_v3_variant_split(broken, "modified", "modified", tag, pattern),
                        silent = TRUE), "try-error"))
  broken <- df
  broken$compositionality[broken$compound == "baby_blue"] <- 99
  stopifnot(inherits(try(coha_v3_variant_split(broken, "modified", "modified", tag, pattern),
                        silent = TRUE), "try-error"))
}
cat("Modified split tests passed: all / nn / an, 180 train / 100 test, unchanged ratings\n")
