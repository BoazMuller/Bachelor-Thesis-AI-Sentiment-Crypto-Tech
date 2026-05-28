script_args <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_args[grepl("^--file=", script_args)][1])

if (is.na(script_file)) {
  stop("Could not determine setup script path.", call. = FALSE)
}

project_root <- normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
lockfile <- file.path(project_root, "renv.lock")

if (!file.exists(lockfile)) {
  stop("renv.lock was not found at the project root.", call. = FALSE)
}

if (!requireNamespace("renv", quietly = TRUE)) {
  stop(
    paste(
      "The renv package is required for R environment restoration.",
      "Install renv once with install.packages('renv'), then rerun:",
      "Rscript r/setup_renv.R"
    ),
    call. = FALSE
  )
}

renv::restore(project = project_root, lockfile = lockfile, prompt = FALSE)
