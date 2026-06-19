args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || "-h" %in% args) {
  cat(
    paste(
      "Usage:",
      "Rscript r/modeling/run_tvpvar_connectedness.R <input_csv> <output_dir> [nlag] [nfore] [columns_csv] [rds_output_dir]",
      "",
      "Runs ConnectednessApproach TVP-VAR using complete cases for the selected system.",
      "When columns_csv is supplied, only those comma-separated variables",
      "plus the date column are used.",
      sep = "\n"
    ),
    "\n"
  )
  quit(status = 0)
}

if (length(args) < 2) {
  stop(
    paste(
      "Usage:",
      "Rscript r/modeling/run_tvpvar_connectedness.R",
      "<input_csv> <output_dir> [nlag] [nfore] [columns_csv] [rds_output_dir]",
      sep = " "
    )
  )
}

source("r/requirements.R")

input_csv <- args[[1]]
output_dir <- args[[2]]
nlag <- ifelse(length(args) >= 3, as.integer(args[[3]]), 1L)
nfore <- ifelse(length(args) >= 4, as.integer(args[[4]]), 10L)
selected_columns <- if (length(args) >= 5) {
  trimws(strsplit(args[[5]], ",", fixed = TRUE)[[1]])
} else {
  character(0)
}
rds_output_dir <- ifelse(length(args) >= 6, args[[6]], output_dir)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rds_output_dir, recursive = TRUE, showWarnings = FALSE)

raw_data <- readr::read_csv(input_csv, show_col_types = FALSE)

if (!("date" %in% names(raw_data))) {
  stop("Input CSV must contain a 'date' column.")
}

if (length(selected_columns) > 0) {
  missing_columns <- setdiff(selected_columns, names(raw_data))
  if (length(missing_columns) > 0) {
    stop("Input CSV is missing selected columns: ", paste(missing_columns, collapse = ", "))
  }
  raw_data <- raw_data |>
    dplyr::select(date, dplyr::all_of(selected_columns))
}

raw_data <- raw_data |>
  dplyr::filter(stats::complete.cases(dplyr::across(dplyr::everything())))

dates <- as.Date(raw_data$date)
model_data <- raw_data |>
  dplyr::select(-date) |>
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric))

if (anyNA(dates)) {
  stop("Input CSV contains dates that could not be parsed.")
}

if (nrow(model_data) <= nlag + 5) {
  stop("Too few complete observations for the selected TVP-VAR system and lag order.")
}

zoo_data <- zoo::zoo(model_data, order.by = dates)

kappa1 <- 0.99
kappa2 <- 0.99
prior <- ConnectednessApproach::BayesPrior(
  zoo_data,
  size = nrow(zoo_data),
  nlag = nlag
)
tvpvar_fit <- ConnectednessApproach::TVPVAR(
  zoo_data,
  configuration = list(
    l = c(kappa1, kappa2),
    nlag = nlag,
    prior = prior
  )
)
dca <- ConnectednessApproach::TimeConnectedness(
  Phi = tvpvar_fit$B_t,
  Sigma = tvpvar_fit$Q_t,
  nfore = nfore,
  generalized = TRUE
)
dca$config <- list(
  nfore = nfore,
  approach = "Time",
  generalized = TRUE,
  corrected = FALSE
)

saveRDS(dca, file.path(rds_output_dir, paste0("tvpvar_connectedness_h", nfore, ".rds")))
saveRDS(tvpvar_fit, file.path(rds_output_dir, paste0("tvpvar_model_h", nfore, ".rds")))

write_component <- function(component, name) {
  if (is.null(component)) {
    return(invisible(NULL))
  }

  output_path <- file.path(output_dir, paste0(name, "_h", nfore, ".csv"))

  if (zoo::is.zoo(component)) {
    values <- as.data.frame(zoo::coredata(component), check.names = FALSE)
    values <- add_date_column(values, zoo::index(component))
    readr::write_csv(values, output_path)
    return(invisible(NULL))
  }

  if (is.data.frame(component) || is.matrix(component)) {
    values <- as.data.frame(component, check.names = FALSE)
    if (nrow(values) == length(dates) && !("date" %in% names(values))) {
      values <- add_date_column(values, dates)
    }
    readr::write_csv(values, output_path)
    return(invisible(NULL))
  }

  if (is.array(component)) {
    dimensions <- dim(component)
    dimension_names <- dimnames(component)
    if (is.null(dimension_names)) {
      dimension_names <- lapply(dimensions, seq_len)
    } else {
      dimension_names <- Map(
        function(names, size) {
          if (is.null(names)) {
            return(seq_len(size))
          }
          names
        },
        dimension_names,
        dimensions
      )
    }
    grid <- expand.grid(dimension_names, KEEP.OUT.ATTRS = FALSE)
    names(grid) <- paste0("dimension_", seq_along(dimensions))
    time_dimension <- which(dimensions == length(dates))[1]
    if (!is.na(time_dimension)) {
      date_column <- paste0("dimension_", time_dimension)
      grid[[date_column]] <- resolve_component_dates(grid[[date_column]], dates)
      names(grid)[names(grid) == date_column] <- "date"
    }
    grid$value <- as.vector(component)
    readr::write_csv(grid, output_path)
    return(invisible(NULL))
  }

  try(readr::write_csv(as.data.frame(component), output_path), silent = TRUE)
}

add_date_column <- function(values, component_dates) {
  values$date <- as.Date(component_dates)
  values |>
    dplyr::select(date, dplyr::everything())
}

resolve_component_dates <- function(values, fallback_dates) {
  numeric_index <- suppressWarnings(as.integer(as.character(values)))
  if (all(!is.na(numeric_index)) && all(numeric_index >= 1) && all(numeric_index <= length(fallback_dates))) {
    return(as.Date(fallback_dates[numeric_index]))
  }
  as.Date(values)
}

write_component(dca$TABLE, "connectedness_table")
write_component(dca$TCI, "tci")
write_component(dca$TO, "to")
write_component(dca$FROM, "from")
write_component(dca$NET, "net")
write_component(dca$NPDC, "npdc")

standardized_innovations <- function(values, fit, lag_order, observation_dates) {
  centered <- scale(as.matrix(values), center = TRUE, scale = FALSE)
  innovations <- matrix(
    NA_real_,
    nrow = nrow(centered),
    ncol = ncol(centered),
    dimnames = list(NULL, colnames(centered))
  )

  for (i in seq.int(lag_order + 1L, nrow(centered))) {
    lag_vector <- unlist(
      lapply(seq_len(lag_order), function(lag) centered[i - lag, ]),
      use.names = FALSE
    )
    prediction <- fit$B_t[, , i, drop = FALSE][, , 1] %*% lag_vector
    residual <- centered[i, ] - as.numeric(prediction)
    innovation_sd <- sqrt(pmax(diag(fit$Q_t[, , i]), .Machine$double.eps))
    innovations[i, ] <- residual / innovation_sd
  }

  tibble::as_tibble(innovations) |>
    dplyr::mutate(date = observation_dates, .before = 1)
}

companion_radius <- function(coefficient_matrix, variable_count, lag_order) {
  if (lag_order == 1L) {
    companion <- coefficient_matrix
  } else {
    companion <- rbind(
      coefficient_matrix,
      cbind(
        diag(variable_count * (lag_order - 1L)),
        matrix(0, nrow = variable_count * (lag_order - 1L), ncol = variable_count)
      )
    )
  }
  max(Mod(eigen(companion, only.values = TRUE)$values))
}

build_post_estimation_diagnostics <- function(values, fit, lag_order, observation_dates) {
  innovations <- standardized_innovations(values, fit, lag_order, observation_dates)
  diagnostic_rows <- list()

  for (variable in names(values)) {
    series <- stats::na.omit(innovations[[variable]])
    for (lag in c(10L, 20L)) {
      if (length(series) <= lag) {
        next
      }
      test <- stats::Box.test(series, lag = lag, type = "Ljung-Box")
      diagnostic_rows[[length(diagnostic_rows) + 1L]] <- tibble::tibble(
        diagnostic = "ljung_box_standardized_innovation",
        variable = variable,
        lag = lag,
        statistic = as.numeric(test$statistic),
        p_value = test$p.value,
        nobs = length(series),
        note = ""
      )
    }
  }

  radii <- vapply(
    seq_len(dim(fit$B_t)[3]),
    function(i) companion_radius(fit$B_t[, , i], ncol(values), lag_order),
    numeric(1)
  )
  diagnostic_rows[[length(diagnostic_rows) + 1L]] <- tibble::tibble(
    diagnostic = c(
      "time_varying_coefficient_max_spectral_radius",
      "time_varying_coefficient_unstable_date_count",
      "time_varying_coefficient_mean_absolute_change"
    ),
    variable = "",
    lag = NA_integer_,
    statistic = c(
      max(radii, na.rm = TRUE),
      sum(radii > 1 + 1e-8, na.rm = TRUE),
      mean(
        abs(
          fit$B_t[, , 2:dim(fit$B_t)[3], drop = FALSE] -
            fit$B_t[, , 1:(dim(fit$B_t)[3] - 1L), drop = FALSE]
        ),
        na.rm = TRUE
      )
    ),
    p_value = NA_real_,
    nobs = nrow(values),
    note = c(
      "Spectral radius at each date; values below or equal to one indicate stability",
      "Number of dates with spectral radius above one",
      "Average absolute one-period change across all time-varying coefficients"
    )
  )

  list(
    innovations = innovations,
    diagnostics = dplyr::bind_rows(diagnostic_rows)
  )
}

build_output_completeness <- function(connectedness, expected_rows) {
  components <- c("TABLE", "TCI", "TO", "FROM", "NET", "NPDC")
  dplyr::bind_rows(lapply(components, function(component) {
    value <- connectedness[[component]]
    dimensions <- if (is.null(value)) integer(0) else dim(value)
    tibble::tibble(
      component = component,
      present = !is.null(value),
      dimensions = paste(dimensions, collapse = "x"),
      expected_time_observations = expected_rows,
      contains_missing = if (is.null(value)) NA else anyNA(value),
      note = if (is.null(value)) "Missing component" else ""
    )
  }))
}

post_estimation <- build_post_estimation_diagnostics(
  model_data,
  tvpvar_fit,
  nlag,
  dates
)
readr::write_csv(
  post_estimation$innovations,
  file.path(output_dir, paste0("standardized_innovations_h", nfore, ".csv"))
)
readr::write_csv(
  post_estimation$diagnostics,
  file.path(output_dir, paste0("tvpvar_post_estimation_diagnostics_h", nfore, ".csv"))
)
readr::write_csv(
  build_output_completeness(dca, nrow(model_data)),
  file.path(output_dir, paste0("tvpvar_output_completeness_h", nfore, ".csv"))
)

message("Wrote TVP-VAR connectedness outputs to ", output_dir)
