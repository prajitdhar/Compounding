library(dplyr)
library(tidyr)
library(tibble)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else getwd()
source(file.path(script_dir, "report_grid_filter.R"))

path <- Sys.getenv("COMPOUNDING_REGRESSION_PATH", unset = "runs/compositionality/regressions/results")

save_path <- Sys.getenv("COMPOUNDING_SUMMARY_PATH", unset = "./")
dir.create(save_path, recursive = TRUE, showWarnings = FALSE)

# Standalone exploratory summaries remain available without GRID_MODES. The
# production grid launcher sets it and must not summarize an incomplete study.
if (nzchar(trimws(Sys.getenv("GRID_MODES", unset = "")))) {
  coverage <- google_v3_validate_results(path)
  write.csv(coverage, file.path(save_path, "configuration_coverage.csv"), row.names = FALSE)
}

synchronous_timespan <- 10000

read_matching_csvs <- function(pattern) {
  file_list <- list.files(path = path, pattern = pattern, full.names = TRUE)

  if (length(file_list) == 0) {
    return(tibble())
  }

  frames <- lapply(file_list, function(file) {
    frame <- read.csv(file)
    if (!"context_mode" %in% names(frame)) frame$context_mode <- "unspecified"
    if (!"context_size" %in% names(frame)) frame$context_size <- NA_integer_
    if (!"pattern" %in% names(frame)) frame$pattern <- "all"
    if (!"frequency_family" %in% names(frame)) frame$frequency_family <- "unspecified"
    if (!"productivity_family" %in% names(frame)) frame$productivity_family <- "unspecified"
    if (!"train_variant" %in% names(frame)) frame$train_variant <- "original"
    if (!"test_variant" %in% names(frame)) frame$test_variant <- "original"
    coha_v3_report_grid(frame)
  })
  bind_rows(frames) %>%
    mutate(context_mode = replace_na(as.character(context_mode), "unspecified"),
           context_size = as.integer(context_size),
           pattern = replace_na(as.character(pattern), "all"))
}

split_sync_diachronic <- function(df) {
  if (nrow(df) == 0 || !"timespan" %in% names(df)) {
    return(list(sync = tibble(), diachronic = tibble()))
  }

  df <- df %>%
    mutate(timespan = suppressWarnings(as.numeric(timespan)))

  list(
    sync = df %>% filter(timespan == synchronous_timespan),
    diachronic = df %>% filter(!is.na(timespan), timespan != synchronous_timespan)
  )
}

add_within_group_ranks <- function(df, metric_col, rank_col, grouping_cols) {
  if (nrow(df) == 0) {
    return(df)
  }

  df %>%
    group_by(across(all_of(grouping_cols))) %>%
    mutate("{rank_col}" := min_rank(desc(.data[[metric_col]]))) %>%
    ungroup()
}

rsquared_df <- read_matching_csvs("^rsquared_.*\\.csv$")
coefimp_df <- read_matching_csvs("^coefimp_.*\\.csv$")
permimp_df <- read_matching_csvs("^permimp_.*\\.csv$")
familyimp_df <- read_matching_csvs("^familyimp_.*\\.csv$")

write_feature_set_summary <- function(df, output_name) {
  if (nrow(df) == 0) {
    return(invisible(NULL))
  }
  if (!"CVTrainRsquared" %in% names(df)) df$CVTrainRsquared <- NA_real_
  if (!"CVTrainSpearman" %in% names(df)) df$CVTrainSpearman <- NA_real_

  feature_set_group_cols <- c("corpus", "tag", "ppmi", "setting", "context_mode", "context_size",
                              "pattern", "train_variant", "test_variant", "frequency_family", "productivity_family", "timespan", "cutoff", "impute", "y")

  rsquared_ranked_df <- df %>%
    add_within_group_ranks("TestRsquared", "test_r2_rank", feature_set_group_cols) %>%
    add_within_group_ranks("TestSpearman", "test_spearman_rank", feature_set_group_cols) %>%
    add_within_group_ranks("TestAdjRsquared", "test_adj_r2_rank", feature_set_group_cols)

  feature_set_consistency_df <- rsquared_ranked_df %>%
    group_by(corpus, tag, ppmi, setting, context_mode, context_size, pattern, train_variant, test_variant,
             frequency_family, productivity_family, timespan, features, y) %>%
    summarise(
      runs = n(),
      mean_test_r2 = mean(TestRsquared, na.rm = TRUE),
      median_test_r2 = median(TestRsquared, na.rm = TRUE),
      sd_test_r2 = sd(TestRsquared, na.rm = TRUE),
      mean_test_spearman = mean(TestSpearman, na.rm = TRUE),
      median_test_spearman = median(TestSpearman, na.rm = TRUE),
      sd_test_spearman = sd(TestSpearman, na.rm = TRUE),
      mean_cv_train_r2 = if (all(is.na(CVTrainRsquared))) NA_real_ else mean(CVTrainRsquared, na.rm = TRUE),
      mean_cv_train_spearman = if (all(is.na(CVTrainSpearman))) NA_real_ else mean(CVTrainSpearman, na.rm = TRUE),
      mean_test_adj_r2 = mean(TestAdjRsquared, na.rm = TRUE),
      mean_r2_rank = mean(test_r2_rank, na.rm = TRUE),
      mean_spearman_rank = mean(test_spearman_rank, na.rm = TRUE),
      win_rate_r2 = mean(test_r2_rank == 1, na.rm = TRUE),
      win_rate_spearman = mean(test_spearman_rank == 1, na.rm = TRUE),
      top3_rate_r2 = mean(test_r2_rank <= 3, na.rm = TRUE),
      top3_rate_spearman = mean(test_spearman_rank <= 3, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(mean_spearman_rank, desc(mean_test_spearman), mean_r2_rank, desc(mean_test_r2))

  write.csv(
    feature_set_consistency_df,
    file.path(save_path, output_name),
    row.names = FALSE
  )
}

write_feature_coefficient_summary <- function(df, output_name) {
  if (nrow(df) == 0) {
    return(invisible(NULL))
  }

  feature_group_cols <- c("corpus", "tag", "ppmi", "setting", "context_mode", "context_size",
                          "pattern", "train_variant", "test_variant", "frequency_family", "productivity_family", "timespan", "cutoff", "impute", "features", "y")

  coefimp_ranked_df <- df %>%
    group_by(across(all_of(feature_group_cols))) %>%
    mutate(coefficient_rank = min_rank(desc(coefficient_importance))) %>%
    ungroup()

  feature_coefficient_consistency_df <- coefimp_ranked_df %>%
    group_by(corpus, tag, ppmi, setting, context_mode, context_size, pattern, train_variant, test_variant,
             frequency_family, productivity_family, timespan,
             features, y, feature_family, feature) %>%
    summarise(
      runs = n(),
      mean_coefficient_importance = mean(coefficient_importance, na.rm = TRUE),
      median_coefficient_importance = median(coefficient_importance, na.rm = TRUE),
      mean_coefficient_share = mean(coefficient_importance_share, na.rm = TRUE),
      mean_coefficient_rank = mean(coefficient_rank, na.rm = TRUE),
      top5_coef_rate = mean(coefficient_rank <= 5, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(mean_coefficient_rank, desc(mean_coefficient_share), desc(mean_coefficient_importance))

  write.csv(
    feature_coefficient_consistency_df,
    file.path(save_path, output_name),
    row.names = FALSE
  )
}

write_feature_permutation_summary <- function(df, output_name) {
  if (nrow(df) == 0) {
    return(invisible(NULL))
  }

  permimp_ranked_df <- df %>%
    group_by(corpus, tag, ppmi, setting, context_mode, context_size, pattern, train_variant, test_variant,
             frequency_family, productivity_family,
             timespan, cutoff, impute, features, y) %>%
    mutate(
      permutation_r2_rank = min_rank(desc(delta_test_r2)),
      permutation_spearman_rank = min_rank(desc(delta_test_spearman))
    ) %>%
    ungroup()

  feature_permutation_consistency_df <- permimp_ranked_df %>%
    group_by(corpus, tag, ppmi, setting, context_mode, context_size, pattern, train_variant, test_variant,
             frequency_family, productivity_family, timespan,
             features, y, feature_family, feature) %>%
    summarise(
      runs = n(),
      mean_delta_test_r2 = mean(delta_test_r2, na.rm = TRUE),
      median_delta_test_r2 = median(delta_test_r2, na.rm = TRUE),
      mean_delta_test_spearman = mean(delta_test_spearman, na.rm = TRUE),
      median_delta_test_spearman = median(delta_test_spearman, na.rm = TRUE),
      mean_permutation_r2_rank = mean(permutation_r2_rank, na.rm = TRUE),
      mean_permutation_spearman_rank = mean(permutation_spearman_rank, na.rm = TRUE),
      top5_permutation_r2_rate = mean(permutation_r2_rank <= 5, na.rm = TRUE),
      top5_permutation_spearman_rate = mean(permutation_spearman_rank <= 5, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(mean_permutation_spearman_rank, desc(mean_delta_test_spearman), mean_permutation_r2_rank)

  write.csv(
    feature_permutation_consistency_df,
    file.path(save_path, output_name),
    row.names = FALSE
  )
}

write_feature_family_summary <- function(df, output_name) {
  if (nrow(df) == 0) {
    return(invisible(NULL))
  }

  family_consistency_df <- df %>%
    group_by(corpus, tag, ppmi, setting, context_mode, context_size, pattern, train_variant, test_variant,
             frequency_family, productivity_family, timespan,
             features, y, feature_family) %>%
    summarise(
      runs = n(),
      mean_total_coefficient_share = mean(total_coefficient_importance_share, na.rm = TRUE),
      mean_family_delta_test_r2 = mean(mean_delta_test_r2, na.rm = TRUE),
      mean_family_delta_test_spearman = mean(mean_delta_test_spearman, na.rm = TRUE),
      max_family_delta_test_r2 = max(max_delta_test_r2, na.rm = TRUE),
      max_family_delta_test_spearman = max(max_delta_test_spearman, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_family_delta_test_spearman), desc(mean_total_coefficient_share))

  write.csv(
    family_consistency_df,
    file.path(save_path, output_name),
    row.names = FALSE
  )
}

rsquared_split <- split_sync_diachronic(rsquared_df)
coefimp_split <- split_sync_diachronic(coefimp_df)
permimp_split <- split_sync_diachronic(permimp_df)
familyimp_split <- split_sync_diachronic(familyimp_df)

write_feature_set_summary(rsquared_df, "summary_feature_set_consistency.csv")
write_feature_set_summary(rsquared_split$sync, "summary_sync_feature_set_consistency.csv")
write_feature_set_summary(rsquared_split$diachronic, "summary_diachronic_feature_set_consistency.csv")

write_feature_coefficient_summary(coefimp_df, "summary_feature_coefficient_consistency.csv")
write_feature_coefficient_summary(coefimp_split$sync, "summary_sync_feature_coefficient_consistency.csv")
write_feature_coefficient_summary(coefimp_split$diachronic, "summary_diachronic_feature_coefficient_consistency.csv")

write_feature_permutation_summary(permimp_df, "summary_feature_permutation_consistency.csv")
write_feature_permutation_summary(permimp_split$sync, "summary_sync_feature_permutation_consistency.csv")
write_feature_permutation_summary(permimp_split$diachronic, "summary_diachronic_feature_permutation_consistency.csv")

write_feature_family_summary(familyimp_df, "summary_feature_family_consistency.csv")
write_feature_family_summary(familyimp_split$sync, "summary_sync_feature_family_consistency.csv")
write_feature_family_summary(familyimp_split$diachronic, "summary_diachronic_feature_family_consistency.csv")
