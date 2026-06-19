args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || "-h" %in% args) {
  cat(
    paste(
      "Usage:",
      "Rscript r/modeling/run_egarch_volatility.R [input_csv] [output_dir]",
      "",
      "Fits plain EGARCH(1,1) models with constant mean, no ARMA terms,",
      "no sentiment regressors, and Student-t innovations. Outputs a",
      "conditional-volatility panel for TVP-VAR connectedness models.",
      sep = "\n"
    ),
    "\n"
  )
  quit(status = 0)
}

source("r/requirements.R")

input_csv <- ifelse(
  length(args) >= 1,
  args[[1]],
  "results/tables/tvpvar_connectedness/tvpvar_connectedness_return_dataset.csv"
)
output_dir <- ifelse(
  length(args) >= 2,
  args[[2]],
  "results/tables/tvpvar_connectedness"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- readr::read_csv(input_csv, show_col_types = FALSE) |>
  dplyr::mutate(date = as.Date(.data$date))

asset_specs <- c(
  bitcoin = "bitcoin_adj_close_log_return",
  ndx = "ndx_adj_close_log_return",
  nvda = "nvda_adj_close_log_return",
  googl = "googl_adj_close_log_return",
  msft = "msft_adj_close_log_return"
)

fit_plain_egarch <- function(series) {
  spec <- rugarch::ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )
  rugarch::ugarchfit(
    spec = spec,
    data = series * 100,
    solver = "hybrid",
    solver.control = list(trace = 0)
  )
}

fit_converged <- function(fit) {
  isTRUE(fit@fit$convergence == 0)
}

assert_fit_converged <- function(fit, asset_name) {
  if (!fit_converged(fit)) {
    stop(
      "Plain EGARCH(1,1) fit did not converge for ",
      asset_name,
      ". Convergence code: ",
      fit@fit$convergence
    )
  }
}

extract_coefficients <- function(fit, asset_name, nobs) {
  coefficients <- as.data.frame(fit@fit$matcoef)
  names(coefficients) <- trimws(names(coefficients))
  coefficients$term <- rownames(coefficients)
  rownames(coefficients) <- NULL
  coefficients |>
    dplyr::rename(
      estimate = `Estimate`,
      standard_error = `Std. Error`,
      t_value = `t value`,
      p_value = `Pr(>|t|)`
    ) |>
    dplyr::mutate(
      asset = asset_name,
      model = "egarch_1_1_constant_mean",
      nobs = nobs,
      .before = 1
    )
}

extract_diagnostics <- function(fit, asset_name, nobs) {
  residuals_std <- as.numeric(rugarch::residuals(fit, standardize = TRUE))
  info <- rugarch::infocriteria(fit)
  info_names <- rownames(info)
  if (is.null(info_names)) {
    info_names <- names(info)
  }

  info_rows <- tibble::tibble(
    asset = asset_name,
    diagnostic = paste0("information_criterion_", info_names),
    lag = NA_integer_,
    statistic = as.numeric(info),
    p_value = NA_real_,
    nobs = nobs,
    note = ""
  )

  lb_rows <- dplyr::bind_rows(lapply(c(10L, 20L), function(lag) {
    residual_test <- stats::Box.test(residuals_std, lag = lag, type = "Ljung-Box")
    squared_test <- stats::Box.test(residuals_std^2, lag = lag, type = "Ljung-Box")
    tibble::tibble(
      asset = asset_name,
      diagnostic = c("ljung_box_standardized_residuals", "ljung_box_squared_standardized_residuals"),
      lag = lag,
      statistic = c(as.numeric(residual_test$statistic), as.numeric(squared_test$statistic)),
      p_value = c(residual_test$p.value, squared_test$p.value),
      nobs = nobs,
      note = ""
    )
  }))

  convergence <- tibble::tibble(
    asset = asset_name,
    diagnostic = c("convergence_code", "log_likelihood"),
    lag = NA_integer_,
    statistic = c(fit@fit$convergence, rugarch::likelihood(fit)),
    p_value = NA_real_,
    nobs = nobs,
    note = c(
      ifelse(fit_converged(fit), "Solver converged", "Non-zero solver convergence code"),
      ""
    )
  )

  dplyr::bind_rows(info_rows, lb_rows, convergence)
}

passthrough_columns <- intersect(c("date", "expectation_adjusted_ais"), names(data))
volatility_panel <- data[passthrough_columns]
coefficient_rows <- list()
diagnostic_rows <- list()
long_volatility_rows <- list()

for (asset_name in names(asset_specs)) {
  return_column <- asset_specs[[asset_name]]
  model_data <- data |>
    dplyr::select(date, dplyr::all_of(return_column)) |>
    tidyr::drop_na()

  fit <- fit_plain_egarch(model_data[[return_column]])
  assert_fit_converged(fit, asset_name)
  volatility_column <- paste0(asset_name, "_conditional_volatility")
  asset_volatility <- tibble::tibble(
    date = model_data$date,
    !!volatility_column := as.numeric(rugarch::sigma(fit)) / 100
  )
  volatility_panel <- dplyr::left_join(volatility_panel, asset_volatility, by = "date")

  long_volatility_rows[[length(long_volatility_rows) + 1]] <- tibble::tibble(
    date = model_data$date,
    asset = asset_name,
    return_column = return_column,
    conditional_volatility = as.numeric(rugarch::sigma(fit)) / 100,
    fitted_mean = as.numeric(rugarch::fitted(fit)) / 100,
    residual = as.numeric(rugarch::residuals(fit)) / 100,
    standardized_residual = as.numeric(rugarch::residuals(fit, standardize = TRUE))
  )
  coefficient_rows[[length(coefficient_rows) + 1]] <- extract_coefficients(fit, asset_name, nrow(model_data))
  diagnostic_rows[[length(diagnostic_rows) + 1]] <- extract_diagnostics(fit, asset_name, nrow(model_data))
}

readr::write_csv(volatility_panel, file.path(output_dir, "tvpvar_connectedness_dataset.csv"))
readr::write_csv(dplyr::bind_rows(long_volatility_rows), file.path(output_dir, "egarch_volatility_long.csv"))
readr::write_csv(dplyr::bind_rows(coefficient_rows), file.path(output_dir, "egarch_volatility_coefficients.csv"))
readr::write_csv(dplyr::bind_rows(diagnostic_rows), file.path(output_dir, "egarch_volatility_post_estimation_diagnostics.csv"))

message("Wrote plain EGARCH(1,1) volatility outputs to ", output_dir)
