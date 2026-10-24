#!/usr/bin/env Rscript

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[[1L]]))) else getwd()
regression_root <- dirname(script_dir)
source(file.path(regression_root, "modified_regressions", "coha_v3_regression_data.R"))
root <- tempfile("google-v3-regression-")
dir.create(file.path(root, "local5"), recursive = TRUE)
on.exit(unlink(root, recursive = TRUE), add = TRUE)

ratings <- data.frame(
  modifier = paste0("mod", seq_len(13)), head = paste0("head", seq_len(13)),
  avgModifier = seq_len(13), stdevModifier = 1,
  avgHead = seq_len(13), stdevHead = 1,
  compositionality = seq_len(13), stdevHeadModifier = 1,
  is_adj = FALSE, compound = paste0("compound", seq_len(13)),
  source = c(rep("reddy90", 10), rep("cordeiro100", 3)), is_original = TRUE
)
static <- ratings
static[["sim_with_head:1900"]] <- seq_len(13) / 10
static[["sim_with_head:1910"]] <- NA_real_
static[["sim_bw_settings_head:1900"]] <- seq_len(13) / 20
static[["perc_token_head:1900"]] <- seq_len(13) / 30
static[["assoc_pair_freq:1900"]] <- seq_len(13)
static[["log_mod_freq_new:1900"]] <- seq_len(13)
static[["mod_family_size_new:1900"]] <- seq_len(13)
static[["eligible:1900"]] <- 1
temporal <- ratings[13:1, ]
temporal[["change_comp:1910"]] <- seq_len(13) / 10
temporal[["change_mod:1910"]] <- seq_len(13) / 20
temporal[["change_head:1910"]] <- seq_len(13) / 30
temporal[["eligible:1910"]] <- 1

static_path <- coha_v3_feature_path(root, "local5", 10000, "features", "Aware", "PPMI", "UnTagged", 10, 0)
temporal_path <- coha_v3_feature_path(root, "local5", 10000, "temporal", "Aware", "PPMI", "UnTagged", 10, 0)
write.csv(static, static_path, row.names = FALSE, na = "")
write.csv(temporal, temporal_path, row.names = FALSE, na = "")

input <- coha_v3_read_input(root, "local5", 10000, "Aware", "PPMI", "UnTagged", 10, 0)
split <- coha_v3_original_split(input)
stopifnot(nrow(split$train) == 10, nrow(split$test) == 3)
stopifnot(ncol(coha_v3_select_features(input, "all")) == 8)
stopifnot(ncol(coha_v3_select_features(input, "all_static")) == 5)
stopifnot(ncol(coha_v3_select_features(input, "with_setting")) == 2)
stopifnot(ncol(coha_v3_select_features(input, "association_counts")) == 1)
stopifnot(ncol(coha_v3_select_features(input, "sim_with_head")) == 2)
stopifnot(ncol(coha_v3_select_features(input, "change_head")) == 1)
stopifnot(is.numeric(input[["sim_with_head:1910"]]))
stopifnot(input[["change_comp:1910"]][1] == 1.3)
stopifnot(!"eligible:1900" %in% names(coha_v3_select_features(input, "all")))
stopifnot(!any(grepl("_new", names(input))))

setting_bases <- coha_v3_feature_groups$with_setting
stopifnot(length(setting_bases) == 12, !anyDuplicated(setting_bases))
aware_bases <- setting_bases[!grepl("constituent", setting_bases)]
pooled_bases <- setting_bases[grepl("constituent", setting_bases)]
with_setting_columns <- function(bases) {
  frame <- ratings
  for (base in bases) frame[[paste0(base, ":1900")]] <- seq_len(nrow(frame))
  frame
}
for (bases in list(aware_bases, pooled_bases)) {
  frame <- with_setting_columns(bases)
  selected <- coha_v3_select_features(frame, "with_setting")
  stopifnot(ncol(selected) == 6)
  stopifnot(all(vapply(names(selected), coha_v3_feature_family, character(1)) == "with_setting"))
}
stopifnot(ncol(coha_v3_select_features(with_setting_columns(character()), "with_setting")) == 0)

frequency <- ratings
for (base in c("log_comp_freq", "log_mod_freq", "log_mod_freq_new", "log_head_freq", "log_head_freq_new")) {
  frequency[[paste0(base, ":1900")]] <- seq_len(nrow(frequency))
}
stopifnot(ncol(coha_v3_select_features(frequency, "log_freq")) == 3)
stopifnot(!any(grepl("_new", names(coha_v3_select_features(frequency, "all")))))
frequency[["comp_freq:1900"]] <- seq_len(nrow(frequency))
frequency[["comp_tf:1900"]] <- seq_len(nrow(frequency))
frequency[["mod_prod:1900"]] <- seq_len(nrow(frequency))
frequency[["mod_family_size:1900"]] <- seq_len(nrow(frequency))
stopifnot(identical(names(coha_v3_select_features(frequency, "all")),
                    c("comp_freq:1900", "mod_prod:1900")))
stopifnot(identical(names(coha_v3_select_features(frequency, "all", "log_freq", "family_size")),
                    c("log_comp_freq:1900", "log_mod_freq:1900", "log_head_freq:1900",
                      "mod_family_size:1900")))
stopifnot(ncol(coha_v3_select_features(frequency, "representation_only")) == 0)
stopifnot(ncol(coha_v3_select_features(input, "all_no_setting")) == 6)

temporal$compound[1] <- "wrong"
write.csv(temporal, temporal_path, row.names = FALSE, na = "")
stopifnot(inherits(try(coha_v3_read_input(root, "local5", 10000, "Aware", "PPMI", "UnTagged", 10, 0),
                        silent = TRUE), "try-error"))
cat("Google v3 regression input tests passed\n")
