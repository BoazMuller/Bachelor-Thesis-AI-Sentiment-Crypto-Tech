required_packages <- c(
  "ConnectednessApproach",
  "dplyr",
  "readr",
  "tidyr",
  "xts",
  "zoo"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste(
      "Missing R packages:",
      paste(missing_packages, collapse = ", "),
      "\nRestore the project R environment with:",
      "\n  Rscript r/setup_renv.R",
      "\nThe model scripts do not install packages at run time."
    ),
    call. = FALSE
  )
}

invisible(required_packages)
