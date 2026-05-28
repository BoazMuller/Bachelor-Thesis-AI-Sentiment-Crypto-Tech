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
  install.packages(missing_packages)
}

invisible(lapply(required_packages, require, character.only = TRUE))
