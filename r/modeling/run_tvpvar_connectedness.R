args <- commandArgs(trailingOnly = TRUE)

if ("--help" %in% args || "-h" %in% args) {
  cat(
    paste(
      "Usage:",
      "Rscript r/modeling/run_tvpvar_connectedness.R <input_csv> <output_dir> [nlag] [nfore] [columns_csv] [rds_output_dir]",
      "",
      "Runs ConnectednessApproach TVP-VAR on a complete volatility panel.",
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

dates <- as.Date(raw_data$date)
model_data <- raw_data |>
  dplyr::select(-date) |>
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric))

if (anyNA(dates)) {
  stop("Input CSV contains dates that could not be parsed.")
}

if (anyNA(model_data)) {
  stop("Input CSV contains missing model values. Prepare a complete TVP-VAR panel first.")
}

zoo_data <- zoo::zoo(model_data, order.by = dates)

dca <- ConnectednessApproach::ConnectednessApproach(
  zoo_data,
  nlag = nlag,
  nfore = nfore,
  model = "TVP-VAR",
  connectedness = "Time",
  VAR_config = list(
    TVPVAR = list(
      kappa1 = 0.99,
      kappa2 = 0.99,
      prior = "BayesPrior",
      gamma = 0.01
    )
  ),
  Connectedness_config = list(
    TimeConnectedness = list(generalized = TRUE)
  )
)

saveRDS(dca, file.path(rds_output_dir, paste0("tvpvar_connectedness_h", nfore, ".rds")))

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

message("Wrote TVP-VAR connectedness outputs to ", output_dir)
