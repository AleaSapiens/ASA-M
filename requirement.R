packages <- c(
  "mlr3verse",
  "mlr3",
  "mlr3pipelines",
  "mlr3tuningspaces",
  "mlr3filters",
  "mlr3benchmark",
  "data.table",
  "readxl",
  "tidyr",
  "writexl",
  "future",
  "pROC",
  "ggplot2",
  "ggprism",
  "ggtext",
  "rsample",
  "dplyr",
  "here",
  "progressr",
  "jsonlite",
  "boot",
  "dcurves",
  "rmda",
  "kernelshap",
  "shapviz",
  "DALEX",
  "DALEXtra",
  "patchwork",
  "shiny",
  "pak",
  "kknn",
  "lightgbm"
)
# Install the missing package
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}
invisible(lapply(packages, install_if_missing))
library(pak)
pak::pak("mlr-org/mlr3extralearners@*release")
install.packages('remotes')
remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.3/catboost-R-Windows-1.2.3.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))

