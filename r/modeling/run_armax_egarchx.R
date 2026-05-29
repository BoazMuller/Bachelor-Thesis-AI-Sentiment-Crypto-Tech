args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || "-h" %in% args) {
  cat(
    paste(
      "Usage:",
      "Rscript r/modeling/run_armax_egarchx.R [input_csv] [output_dir] [max_p] [max_q]",
      "",
      "Fits BTC and NASDAQ-100 ARMAX-EGARCHX models with EGARCH(1,1),",
      "Student-t innovations, BIC-selected ARMA mean lags, and lagged",
      "sentiment in both the mean and variance equations.",
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
  "results/models/armax_egarchx/egarch_input_returns_sentiment.csv"
)
output_dir <- ifelse(
  length(args) >= 2,
  args[[2]],
  "results/models/armax_egarchx"
)
max_p <- ifelse(length(args) >= 3, as.integer(args[[3]]), 3L)
max_q <- ifelse(length(args) >= 4, as.integer(args[[4]]), 3L)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- readr::read_csv(input_csv, show_col_types = FALSE) |>
  dplyr::mutate(
    date = as.Date(.data$date),
    lagged_raw_ais = dplyr::lag(.data$raw_ais)
  )

asset_specs <- list(
  bitcoin = "bitcoin_adj_close_log_return",
  ndx = "ndx_adj_close_log_return"
)

sentiment_specs <- list(
  benchmark_eais = "lagged_expectation_adjusted_ais",
  robustness_raw_ais = "lagged_raw_ais"
)

fit_rows <- list()
lag_rows <- list()
diagnostic_rows <- list()
volatility_rows <- list()

fit_model <- function(asset_name, return_column, specification_name, sentiment_column, p, q, model_data) {
  xreg <- as.matrix(model_data[[sentiment_column]])
  colnames(xreg) <- "sentiment_lag"
  spec <- rugarch::ugarchspec(
    variance.model = list(
      model = "eGARCH",
      garchOrder = c(1, 1),
      external.regressors = xreg
    ),
    mean.model = list(
      armaOrder = c(p, q),
      include.mean = TRUE,
      external.regressors = xreg
    ),
    distribution.model = "std"
  )
  rugarch::ugarchfit(
    spec = spec,
    data = model_data[[return_column]] * 100,
    solver = "hybrid",
    solver.control = list(trace = 0)
  )
}

fit_converged <- function(fit) {
  isTRUE(fit@fit$convergence == 0)
}

assert_fit_converged <- function(fit, asset_name, specification_name, p, q) {
  if (!fit_converged(fit)) {
    stop(
      "Selected ARMAX-EGARCHX fit did not converge for ",
      asset_name,
      " / ",
      specification_name,
      " with ARMA(",
      p,
      ",",
      q,
      "). Convergence code: ",
      fit@fit$convergence
    )
  }
}

extract_coefficients <- function(fit, asset_name, specification_name, p, q) {
  default <- as.data.frame(fit@fit$matcoef)
  names(default) <- trimws(names(default))
  default$term <- rownames(default)
  rownames(default) <- NULL
  default$standard_error_type <- "default"

  if (!is.null(fit@fit$robust.matcoef)) {
    robust <- as.data.frame(fit@fit$robust.matcoef)
    names(robust) <- trimws(names(robust))
    robust$term <- rownames(robust)
    rownames(robust) <- NULL
    robust$standard_error_type <- "robust"
    values <- dplyr::bind_rows(default, robust)
  } else {
    values <- default
  }

  values |>
    dplyr::rename(
      estimate = `Estimate`,
      standard_error = `Std. Error`,
      t_value = `t value`,
      p_value = `Pr(>|t|)`
    ) |>
    dplyr::mutate(
      asset = asset_name,
      specification = specification_name,
      selected_p = p,
      selected_q = q,
      .before = 1
    )
}

extract_diagnostics <- function(fit, asset_name, specification_name, p, q, nobs) {
  residuals_std <- as.numeric(rugarch::residuals(fit, standardize = TRUE))
  info <- rugarch::infocriteria(fit)
  info_names <- rownames(info)
  if (is.null(info_names)) {
    info_names <- names(info)
  }

  info_rows <- tibble::tibble(
    asset = asset_name,
    specification = specification_name,
    selected_p = p,
    selected_q = q,
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
      specification = specification_name,
      selected_p = p,
      selected_q = q,
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
    specification = specification_name,
    selected_p = p,
    selected_q = q,
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

extract_volatility <- function(fit, model_data, asset_name, specification_name) {
  tibble::tibble(
    date = model_data$date,
    asset = asset_name,
    specification = specification_name,
    conditional_volatility = as.numeric(rugarch::sigma(fit)) / 100,
    fitted_mean = as.numeric(rugarch::fitted(fit)) / 100,
    residual = as.numeric(rugarch::residuals(fit)) / 100,
    standardized_residual = as.numeric(rugarch::residuals(fit, standardize = TRUE))
  )
}

for (asset_name in names(asset_specs)) {
  return_column <- asset_specs[[asset_name]]
  for (specification_name in names(sentiment_specs)) {
    sentiment_column <- sentiment_specs[[specification_name]]
    model_data <- data |>
      dplyr::select(date, dplyr::all_of(return_column), dplyr::all_of(sentiment_column)) |>
      tidyr::drop_na()

    candidates <- list()
    for (p in 0:max_p) {
      for (q in 0:max_q) {
        fit <- tryCatch(
          fit_model(asset_name, return_column, specification_name, sentiment_column, p, q, model_data),
          error = function(error) error
        )
        if (inherits(fit, "error")) {
          lag_rows[[length(lag_rows) + 1]] <- tibble::tibble(
            asset = asset_name,
            specification = specification_name,
            p = p,
            q = q,
            bic = NA_real_,
            converged = FALSE,
            selected_by_bic = FALSE,
            nobs = nrow(model_data),
            note = paste(class(fit)[1], fit$message, sep = ": ")
          )
          next
        }

        converged <- fit_converged(fit)
        bic <- if (converged) as.numeric(rugarch::infocriteria(fit)["Bayes", 1]) else NA_real_
        if (converged && !is.na(bic)) {
          candidates[[length(candidates) + 1]] <- list(p = p, q = q, bic = bic, fit = fit)
        }
        lag_rows[[length(lag_rows) + 1]] <- tibble::tibble(
          asset = asset_name,
          specification = specification_name,
          p = p,
          q = q,
          bic = bic,
          converged = converged,
          selected_by_bic = FALSE,
          nobs = nrow(model_data),
          note = ifelse(converged, "", paste("Non-zero solver convergence code:", fit@fit$convergence))
        )
      }
    }

    valid_candidates <- candidates[!is.na(vapply(candidates, function(candidate) candidate$bic, numeric(1)))]
    if (length(valid_candidates) == 0) {
      stop("No converged candidate models for ", asset_name, " / ", specification_name)
    }

    selected_index <- which.min(vapply(valid_candidates, function(candidate) candidate$bic, numeric(1)))
    selected <- valid_candidates[[selected_index]]
    assert_fit_converged(selected$fit, asset_name, specification_name, selected$p, selected$q)

    lag_rows <- lapply(lag_rows, function(row) {
      row$selected_by_bic <- row$selected_by_bic | (
        row$asset == asset_name &&
        row$specification == specification_name &&
        row$p == selected$p &&
        row$q == selected$q
      )
      row
    })

    fit_rows[[length(fit_rows) + 1]] <- extract_coefficients(
      selected$fit,
      asset_name,
      specification_name,
      selected$p,
      selected$q
    )
    diagnostic_rows[[length(diagnostic_rows) + 1]] <- extract_diagnostics(
      selected$fit,
      asset_name,
      specification_name,
      selected$p,
      selected$q,
      nrow(model_data)
    )
    volatility_rows[[length(volatility_rows) + 1]] <- extract_volatility(
      selected$fit,
      model_data,
      asset_name,
      specification_name
    )
  }
}

readr::write_csv(dplyr::bind_rows(fit_rows), file.path(output_dir, "armax_egarchx_coefficients.csv"))
readr::write_csv(dplyr::bind_rows(diagnostic_rows), file.path(output_dir, "armax_egarchx_diagnostics.csv"))
readr::write_csv(dplyr::bind_rows(lag_rows), file.path(output_dir, "armax_egarchx_lag_selection.csv"))
readr::write_csv(dplyr::bind_rows(volatility_rows), file.path(output_dir, "armax_egarchx_conditional_volatility.csv"))

message("Wrote ARMAX-EGARCHX outputs to ", output_dir)
