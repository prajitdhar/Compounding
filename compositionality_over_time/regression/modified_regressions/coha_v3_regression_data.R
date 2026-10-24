coha_v3_rating_cols <- c(
  "modifier", "head", "avgModifier", "stdevModifier", "avgHead",
  "stdevHead", "compositionality", "stdevHeadModifier", "is_adj",
  "compound", "source", "is_original"
)

coha_v3_feature_groups <- list(
  sim_cpf_beta = c("sim_cpf_beta"),
  cordeiro = c("arith_mean_sim", "beta", "geom_mean_sim", paste0("sim_cpf_", c(0, 25, 50, 75, 100)), "sim_cpf_beta"),
  cosine_sim = c("sim_bw_constituents", "sim_with_head", "sim_with_modifier"),
  with_setting = c(
    "sim_bw_settings_modifier", "sim_bw_settings_head",
    "sim_bw_settings_constituent_mod", "sim_bw_settings_constituent_head",
    "perc_token_mod", "perc_token_head", "perc_type_mod", "perc_type_head",
    "perc_token_constituent_mod", "perc_token_constituent_head",
    "perc_type_constituent_mod", "perc_type_constituent_head"
  ),
  info_theory = c("ppmi", "local_mi", "log_ratio"),
  association_counts = c("assoc_pair_freq", "assoc_left_freq", "assoc_right_freq", "assoc_population_size"),
  log_freq = c("log_comp_freq", "log_mod_freq", "log_head_freq"),
  tf = c("comp_tf", "mod_tf", "head_tf"),
  freq = c("comp_freq", "mod_freq", "head_freq"),
  prod = c("mod_prod", "head_prod"),
  family_size = c("mod_family_size", "head_family_size"),
  temporal = c("change_comp", "change_mod", "change_head")
)

coha_v3_feature_family <- function(feature) {
  base <- sub(":.*$", "", feature)
  for (name in names(coha_v3_feature_groups)) {
    if (base %in% coha_v3_feature_groups[[name]]) return(name)
  }
  stop("Unmapped COHA v3 feature: ", feature)
}

coha_v3_is_new_feature <- function(feature) {
  grepl("_new$", sub(":.*$", "", feature))
}

coha_v3_parse_ints <- function(value, label) {
  parts <- strsplit(value, ",", fixed = TRUE)[[1]]
  result <- suppressWarnings(as.integer(parts))
  if (!length(result) || anyNA(result) || any(result < 0L)) stop("Invalid ", label, ": ", value)
  unique(result)
}

coha_v3_feature_path <- function(root, context_mode, context_size, kind, setting, ppmi,
                                 tagging, timespan, cutoff) {
  tag <- paste0(tagging, if (context_size > 0L) paste0("_ctx", context_size) else "")
  file.path(root, context_mode, sprintf("%s_Compound%s_withSetting_%s_%s_%s_%s_na.csv",
                                        kind, setting, ppmi, tag, timespan, cutoff))
}

coha_v3_row_key <- function(df) {
  keys <- c("modifier", "head", "source", "compound", "is_adj", "is_original")
  do.call(paste, c(df[keys], sep = "\r"))
}

coha_v3_read_input <- function(root, context_mode, context_size, setting, ppmi,
                               tagging, timespan, cutoff) {
  static_path <- coha_v3_feature_path(root, context_mode, context_size, "features", setting,
                                      ppmi, tagging, timespan, cutoff)
  if (!file.exists(static_path)) stop("Missing feature file: ", static_path)
  static <- read.csv(static_path, check.names = FALSE, na.strings = c("NA", ""))
  if (!all(coha_v3_rating_cols %in% names(static))) stop("Missing rating columns in ", static_path)
  static$is_adj <- as.logical(static$is_adj)
  static$is_original <- as.logical(static$is_original)
  if (anyNA(static$is_adj) || anyNA(static$is_original)) stop("Invalid rating flags in ", static_path)
  if (anyDuplicated(names(static))) stop("Duplicate columns in ", static_path)
  key <- coha_v3_row_key(static)
  if (anyDuplicated(key)) stop("Duplicate rating rows in ", static_path)

  if (timespan != 10000L) {
    temporal_path <- coha_v3_feature_path(root, context_mode, context_size, "temporal", setting,
                                          ppmi, tagging, timespan, cutoff)
    if (!file.exists(temporal_path)) stop("Missing temporal file: ", temporal_path)
    temporal <- read.csv(temporal_path, check.names = FALSE, na.strings = c("NA", ""))
    if (!all(coha_v3_rating_cols %in% names(temporal))) stop("Missing rating columns in ", temporal_path)
    temporal$is_adj <- as.logical(temporal$is_adj)
    temporal$is_original <- as.logical(temporal$is_original)
    if (anyDuplicated(names(temporal))) stop("Duplicate columns in ", temporal_path)
    temporal_key <- coha_v3_row_key(temporal)
    if (anyDuplicated(temporal_key)) stop("Duplicate rating rows in ", temporal_path)
    order <- match(key, temporal_key)
    if (anyNA(order) || length(temporal_key) != length(key)) stop("Static/temporal rating rows differ")
    for (column in coha_v3_rating_cols) {
      if (!isTRUE(all.equal(static[[column]], temporal[[column]][order], check.attributes = FALSE))) {
        stop("Static/temporal rating mismatch: ", column)
      }
    }
    change_cols <- grep("^change_(comp|mod|head):[0-9]+$", names(temporal), value = TRUE)
    if (!length(change_cols)) stop("No temporal-change columns in ", temporal_path)
    static <- cbind(static, temporal[order, change_cols, drop = FALSE])
  }

  candidates <- setdiff(names(static), coha_v3_rating_cols)
  candidates <- setdiff(candidates, grep("^eligible:[0-9]+$", candidates, value = TRUE))
  # Retain the source file unchanged, but never admit redundant *_new predictors.
  redundant <- candidates[coha_v3_is_new_feature(candidates)]
  static[redundant] <- NULL
  candidates <- setdiff(candidates, redundant)
  invalid <- candidates[!grepl("^[[:alnum:]_]+:[0-9]+$", candidates)]
  if (length(invalid)) stop("Unexpected feature columns: ", paste(invalid, collapse = ", "))
  invisible(vapply(candidates, coha_v3_feature_family, character(1)))
  for (column in candidates) {
    if (all(is.na(static[[column]]))) static[[column]] <- as.numeric(static[[column]])
  }
  if (any(!vapply(static[candidates], is.numeric, logical(1)))) stop("Non-numeric feature column")
  static
}

coha_v3_pattern_rows <- function(df, tagging = "UnTagged", pattern = "all") {
  if (tagging == "Tagged" && pattern == "nn") {
    df <- df[grepl("_NOUN$", df$modifier) & grepl("_NOUN$", df$head), , drop = FALSE]
  } else if (tagging == "Tagged" && pattern == "an") {
    selected <- ifelse(df$is_adj, grepl("_ADJ$", df$modifier), grepl("_NOUN$", df$modifier))
    df <- df[selected & grepl("_NOUN$", df$head), , drop = FALSE]
  } else if (tagging != "UnTagged" || pattern != "all") {
    stop("Invalid tagging/pattern combination")
  }
  df
}

coha_v3_original_split <- function(df, tagging = "UnTagged", pattern = "all") {
  df <- coha_v3_pattern_rows(df, tagging, pattern)
  df <- df[df$is_original %in% TRUE, , drop = FALSE]
  train <- df[df$source %in% c("reddy90", "cordeiro90"), , drop = FALSE]
  test <- df[df$source == "cordeiro100", , drop = FALSE]
  unexpected <- setdiff(unique(df$source), c("reddy90", "cordeiro90", "cordeiro100"))
  if (length(unexpected)) stop("Unexpected rating source: ", paste(unexpected, collapse = ", "))
  if (nrow(train) < 10L || nrow(test) < 3L) stop("Too few original train/test rows")
  if (anyDuplicated(coha_v3_row_key(train)) || anyDuplicated(coha_v3_row_key(test))) {
    stop("Duplicate train/test rating rows")
  }
  train_pairs <- paste(train$modifier, train$head, sep = "\r")
  test_pairs <- paste(test$modifier, test$head, sep = "\r")
  if (length(intersect(train_pairs, test_pairs))) stop("Compound pair overlaps train and test")
  list(train = train, test = test)
}

# The seven manually added American/singular alternatives correspond to
# published rating items. Keep the rating source in the key: equal scores alone
# are not sufficient to identify a pair.
coha_v3_variant_pairs <- data.frame(
  source = c("reddy90", rep("cordeiro90", 4L), rep("cordeiro100", 2L)),
  original = c("call_centre", "benign_tumour", "computer_program",
               "grey_matter", "labour_union", "baby_blues", "contact_lenses"),
  modified = c("call_center", "benign_tumor", "computer_programme",
               "gray_matter", "labor_union", "baby_blue", "contact_lens"),
  stringsAsFactors = FALSE
)

coha_v3_variant_split <- function(df, train_variant = "original",
                                  test_variant = "original",
                                  tagging = "UnTagged", pattern = "all") {
  if (!train_variant %in% c("original", "modified") ||
      !test_variant %in% c("original", "modified")) {
    stop("Variants must be original or modified")
  }
  # Choose the POS interpretation before matching original/alternative pairs.
  # Tagged adjective items otherwise have both NOUN and ADJ rows.
  df <- coha_v3_pattern_rows(df, tagging, pattern)
  split <- coha_v3_original_split(df, tagging, pattern)
  split$test$canonical_compound <- split$test$compound
  rating_cols <- c("avgModifier", "stdevModifier", "avgHead", "stdevHead",
                   "compositionality", "stdevHeadModifier", "is_adj", "source")
  for (i in seq_len(nrow(coha_v3_variant_pairs))) {
    pair <- coha_v3_variant_pairs[i, ]
    original <- which(df$source == pair$source & df$compound == pair$original &
                      df$is_original)
    modified <- which(df$source == pair$source & df$compound == pair$modified &
                      !df$is_original)
    if (length(original) != 1L || length(modified) != 1L ||
        !isTRUE(all.equal(df[original, rating_cols],
                          df[modified, rating_cols], check.attributes = FALSE))) {
      stop("Missing or mismatched rating pair: ", pair$original, " / ", pair$modified)
    }
    part <- if (pair$source == "cordeiro100") "test" else "train"
    index <- which(split[[part]]$source == pair$source &
                   split[[part]]$compound == pair$original)
    if (length(index) != 1L) stop("Original pair is absent from ", part)
    use_modified <- if (part == "train") train_variant == "modified" else
      test_variant == "modified"
    if (use_modified) {
      replacement <- df[modified, names(df), drop = FALSE]
      if (part == "test") replacement$canonical_compound <- pair$original
      split[[part]][index, ] <- replacement[1L, names(split[[part]]), drop = FALSE]
    }
  }
  if (nrow(split$train) != 180L || nrow(split$test) != 100L ||
      anyDuplicated(paste(split$test$source, split$test$canonical_compound, sep = "\r"))) {
    stop("Variant split changed the rating roster")
  }
  train_pairs <- paste(split$train$modifier, split$train$head, sep = "\r")
  test_pairs <- paste(split$test$modifier, split$test$head, sep = "\r")
  if (length(intersect(train_pairs, test_pairs))) {
    stop("A modified compound overlaps the opposite side of the split")
  }
  split
}

coha_v3_select_features <- function(df, feature_set, frequency_family = "freq",
                                    productivity_family = "prod") {
  if (!frequency_family %in% c("freq", "tf", "log_freq")) stop("Invalid frequency family")
  if (!productivity_family %in% c("prod", "family_size")) stop("Invalid productivity family")
  candidates <- grep("^[[:alnum:]_]+:[0-9]+$", names(df), value = TRUE)
  candidates <- setdiff(candidates, grep("^eligible:[0-9]+$", candidates, value = TRUE))
  candidates <- candidates[!coha_v3_is_new_feature(candidates)]
  bases <- sub(":.*$", "", candidates)
  families <- vapply(candidates, coha_v3_feature_family, character(1))
  if (feature_set %in% c("all", "all_static", "all_no_setting")) {
    excluded <- c(setdiff(c("freq", "tf", "log_freq"), frequency_family),
                  setdiff(c("prod", "family_size"), productivity_family))
    if (feature_set == "all_static") excluded <- c(excluded, "temporal")
    if (feature_set == "all_no_setting") excluded <- c(excluded, "with_setting")
    selected <- candidates[!families %in% excluded]
  } else if (feature_set == "representation_only") {
    selected <- candidates[families %in% c("sim_cpf_beta", "cordeiro", "cosine_sim")]
  } else if (feature_set %in% unique(unlist(coha_v3_feature_groups, use.names = FALSE))) {
    selected <- candidates[bases == feature_set]
  } else if (feature_set %in% names(coha_v3_feature_groups)) {
    selected <- candidates[bases %in% coha_v3_feature_groups[[feature_set]]]
  } else {
    stop("Unknown feature set: ", feature_set)
  }
  df[selected]
}
