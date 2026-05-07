library(shiny)
library(data.table)
library(mlr3verse)
library(mlr3)
library(mlr3tuningspaces)
library(mlr3extralearners)
library(kernelshap)
library(rsconnect)
library(shapviz)
library(ggplot2)
library(DT)
library(jsonlite)
library(shinythemes)
library(shinyWidgets)
library(httr) 
#sessionInfo()
# R version 4.4.3 (2025-02-28 ucrt)
# Platform: x86_64-w64-mingw32/x64
# Running under: Windows 11 x64 (build 26200)
# 
# Matrix products: default
# 
# 
# locale:
# [1] LC_COLLATE=Chinese (Simplified)_China.utf8  LC_CTYPE=Chinese (Simplified)_China.utf8   
# [3] LC_MONETARY=Chinese (Simplified)_China.utf8 LC_NUMERIC=C                               
# [5] LC_TIME=Chinese (Simplified)_China.utf8    
# 
# time zone: Asia/Shanghai
# tzcode source: internal
# 
# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
# [1] httr_1.4.8              shinyWidgets_0.9.1      shinythemes_1.2.0       jsonlite_2.0.0          DT_0.34.0              
# [6] ggplot2_4.0.2           shapviz_0.10.3          rsconnect_1.7.0         kernelshap_0.9.1        mlr3extralearners_1.3.0
# [11] mlr3tuningspaces_0.6.0  mlr3tuning_1.6.0        paradox_1.0.1           mlr3verse_0.3.1         mlr3_1.3.0             
# [16] data.table_1.18.0       shiny_1.13.0           
# 
# loaded via a namespace (and not attached):
# [1] rlang_1.1.6          magrittr_2.0.4       otel_0.2.0           compiler_4.4.3       flexmix_2.3-20      
# [6] vctrs_0.6.5          pkgconfig_2.0.3      crayon_1.5.3         fastmap_1.2.0        backports_1.5.0     
# [11] fontawesome_0.5.3    promises_1.5.0       mlr3cluster_0.3.0    modeltools_0.2-24    cachem_1.1.0        
# [16] mlr3misc_0.21.0      later_1.4.5          survdistr_0.0.1      uuid_1.2-2           fpc_2.2-14          
# [21] mlr3measures_1.2.0   mlr3fselect_1.5.0    parallel_4.4.3       prabclus_2.3-5       cluster_2.1.8.1     
# [26] R6_2.6.1             bslib_0.10.0         RColorBrewer_1.1-3   parallelly_1.46.1    jquerylib_0.1.4     
# [31] diptest_0.77-2       xgboost_3.2.0.1      Rcpp_1.1.0           iterators_1.0.14     future.apply_1.20.2 
# [36] httpuv_1.6.16        Matrix_1.7-4         splines_4.4.3        nnet_7.3-20          tidyselect_1.2.1    
# [41] rstudioapi_0.18.0    yaml_2.3.12          mlr3viz_0.11.0       codetools_0.2-20     listenv_0.10.1      
# [46] lattice_0.22-7       tibble_3.3.0         withr_3.0.2          S7_0.2.1             future_1.70.0       
# [51] survival_3.8-3       mclust_6.1.2         kernlab_0.9-33       pillar_1.11.1        mlr3mbo_1.0.0       
# [56] mlr3filters_0.9.0    checkmate_2.3.4      foreach_1.5.2        stats4_4.4.3         generics_0.1.4      
# [61] bbotk_1.9.0          scales_1.4.0         globals_0.19.1       xtable_1.8-8         class_7.3-23        
# [66] glue_1.8.0           tools_4.4.3          mlr3inferr_0.2.1     mlr3pipelines_0.11.0 robustbase_0.99-7   
# [71] mlr3hyperband_1.1.0  grid_4.4.3           crosstalk_1.2.2      mlr3data_0.9.0       palmerpenguins_0.1.1
# [76] cli_3.6.5            spacefillr_0.4.0     doFuture_1.2.1       dplyr_1.1.4          gtable_0.3.6        
# [81] DEoptimR_1.1-4       sass_0.4.10          digest_0.6.39        mlr3cmprsk_0.0.1     lgr_0.5.2           
# [86] htmlwidgets_1.6.4    farver_2.1.2         memoise_2.0.1        htmltools_0.5.9      lifecycle_1.0.5     
# [91] mlr3learners_0.14.0  mime_0.13            MASS_7.3-65  

# Project Configuration
PROJECT_ROOT <- normalizePath(getwd())
MODEL_DIR <- file.path(PROJECT_ROOT, "models")

# Key Drivers Computation Function
compute_key_drivers <- function(learner, X_new, bg_X, feature_cols, target_levels) {
  n <- nrow(X_new)
  empty_drivers <- data.table(
    driver1 = rep(NA_character_, n),
    driver2 = rep(NA_character_, n),
    driver3 = rep(NA_character_, n)
  )
  
  tryCatch({
    bg_X_features <- bg_X[, feature_cols, with = FALSE]
    
    ps <- permshap(
      object = learner,
      X = X_new,
      bg_X = bg_X_features,
      predict_type = "prob"
    )
    
    ps_result <- shapviz(ps)
    sv_pos <- ps_result[["Low"]]
    shap_mat <- as.matrix(sv_pos$S)
    
    feature_names <- colnames(shap_mat)
    
    drivers_mat <- matrix(NA_character_, nrow = n, ncol = 3)
    
    for (i in seq_len(n)) {
      shap_values <- shap_mat[i, ]
      ord <- order(abs(shap_values), decreasing = TRUE)
      k <- min(3, length(ord))
      idx <- ord[seq_len(k)]
      shap_vals <- shap_values[idx]
      feature_names_sel <- feature_names[idx]
      
      descriptions <- character(k)
      
      for (j in seq_len(k)) {
        feat_name <- feature_names_sel[j]
        shap_val <- shap_vals[j]
        feat_val <- X_new[[feat_name]][i]
        
        if (feat_name == "Gender") {
          gender_label <- ifelse(as.character(feat_val) == "1", "Male", "Female")
          direction <- ifelse(shap_val >= 0, "↑", "↓")
          descriptions[j] <- paste0("Gender(", gender_label, ") ", direction)
        } else if (is.numeric(feat_val)) {
          direction <- ifelse(shap_val >= 0, "↑", "↓")
          descriptions[j] <- paste0(feat_name, "=", round(feat_val, 1), " ", direction)
        } else {
          direction <- ifelse(shap_val >= 0, "↑", "↓")
          descriptions[j] <- paste0(feat_name, " ", direction)
        }
      }
      
      drivers_mat[i, seq_len(k)] <- descriptions
    }
    
    return(data.table(
      driver1 = drivers_mat[, 1],
      driver2 = drivers_mat[, 2],
      driver3 = drivers_mat[, 3]
    ))
    
  }, error = function(e) {
    message("SHAP calculation error: ", e$message)
    return(empty_drivers)
  })
}


# 1. Single-user evaluation of an AI agent
generate_ai_interpretation <- function(risk_level, probability, key_drivers) {
  api_key <- Sys.getenv("DEEPSEEK_API_KEY")
  if (api_key == "") {
    api_key <- "sk-12345678" # <--- Please replace this with your actual API key
  }
  
  url <- "https://api.deepseek.com/chat/completions"
  
  driver1 <- if(!is.na(key_drivers$driver1[1])) key_drivers$driver1[1] else "None"
  driver2 <- if(!is.na(key_drivers$driver2[1])) key_drivers$driver2[1] else "None"
  driver3 <- if(!is.na(key_drivers$driver3[1])) key_drivers$driver3[1] else "None"
  
  user_prompt <- sprintf(
    "Based on the following prediction results, please provide a brief, professional clinical interpretation for the attending physician.
    The patient's current risk assessment for low muscle mass in Rheumatoid Arthritis (RA) is as follows:
    - Risk Level: %s
    - Predictive Probability: %.1f%%
    - Top 3 key drivers influencing this prediction (↑ indicates higher risk association, ↓ indicates lower risk association):
      1. %s
      2. %s
      3. %s
    
    Please summarize in 3-4 sentences:
    1. Inform the physician of the patient's risk status regarding low muscle mass.
    2. Briefly explain the clinical reasoning behind this risk based on the provided key indicators and their directions. Do NOT explain what machine learning or SHAP values are; explain directly from the perspective of clinical indicators. 
       If a key driver is gender, use only associative language (e.g., “is associated with higher/lower risk”). Do not describe it as actively raising or lowering risk in a causal manner.
    3. Provide a brief clinical recommendation or monitoring focus.
    
    CRITICAL INSTRUCTION: You must strictly use the term 'low muscle mass'. Do NOT use terms like 'sarcopenia', 'cachexia', or 'muscle wasting'.
    Tone: Professional, objective, and rigorous medical tone. 
    Language: Please reply entirely in English.",
    risk_level, probability * 100, driver1, driver2, driver3
  )
  
  body <- list(
    model = "deepseek-v4-pro",
    messages = list(
      list(role = "system", content = "You are the AI Clinical Assistant for the ASA-M system. You strictly follow clinical definitions and avoid over-interpretation."),
      list(role = "user", content = user_prompt)
    ),
    temperature = 0.2 
  )
  
  tryCatch({
    res <- POST(
      url, add_headers(Authorization = paste("Bearer", api_key)),
      content_type_json(), body = toJSON(body, auto_unbox = TRUE), timeout(300)
    )
    if (status_code(res) == 200) {
      parsed <- content(res, as = "parsed")
      return(parsed$choices[[1]]$message$content)
    } else {
      parsed_error <- content(res, as = "parsed")
      err_msg <- if(!is.null(parsed_error$error$message)) parsed_error$error$message else paste("HTTP", status_code(res))
      return(sprintf("AI interpretation service unavailable. Error: %s", err_msg))
    }
  }, error = function(e) {
    return(sprintf("Network error. Failed to generate AI interpretation. Error: %s", e$message))
  })
}

# 2. Batch Evaluation of an AI agent
generate_batch_ai_interpretation <- function(summary_stats) {
  api_key <- Sys.getenv("DEEPSEEK_API_KEY")
  if (api_key == "") {
    api_key <- "sk-12345678" # <--- Please replace this with your actual API key
  }
  
  url <- "https://api.deepseek.com/chat/completions"
  
  user_prompt <- sprintf(
    "We have conducted a batch risk assessment for low muscle mass in %d patients with Rheumatoid Arthritis (RA).
    
    Our system's predefined clinical risk stratification is:
    - Low Risk: Low probability of low muscle mass. Routine monitoring recommended.
    - Intermediate Risk: Moderate risk. Consider further assessment and lifestyle interventions.
    - High Risk: High probability of low muscle mass. Comprehensive assessment recommended.
    
    The batch prediction results for this RA cohort are:
    - Cohort Mean Predictive Probability: %.1f%%
    - Distribution:
      * High Risk: %d patients (%.1f%%)
      * Intermediate Risk: %d patients (%.1f%%)
      * Low Risk: %d patients (%.1f%%)
    
    Please provide a brief, professional clinical interpretation for the medical team (summarize in 3-4 sentences):
    1. Summarize the overall risk status of low muscle mass in this RA patient cohort.
    2. Provide a brief clinical recommendation or management focus for the cohort based on this specific distribution and our system's risk definitions.
    3. If a key driver is gender, use only associative language (e.g., “is associated with higher/lower risk”). Do not describe it as actively raising or lowering risk in a causal manner.
    
    CRITICAL INSTRUCTION: You must strictly use the term 'low muscle mass'. Do NOT use terms like 'sarcopenia', 'cachexia', or 'muscle wasting'.
    Tone: Professional, objective, and rigorous medical tone. 
    Language: Please reply entirely in English.",
    summary_stats$total_patients,
    summary_stats$mean_probability * 100,
    summary_stats$high_risk, (summary_stats$high_risk / summary_stats$total_patients) * 100,
    summary_stats$intermediate_risk, (summary_stats$intermediate_risk / summary_stats$total_patients) * 100,
    summary_stats$low_risk, (summary_stats$low_risk / summary_stats$total_patients) * 100
  )
  
  body <- list(
    model = "deepseek-v4-pro",
    messages = list(
      list(role = "system", content = "You are the AI Clinical Assistant for the ASA-M system. You strictly follow clinical definitions and avoid over-interpretation."),
      list(role = "user", content = user_prompt)
    ),
    temperature = 0.2
  )
  
  tryCatch({
    res <- POST(
      url, add_headers(Authorization = paste("Bearer", api_key)),
      content_type_json(), body = toJSON(body, auto_unbox = TRUE), timeout(300)
    )
    if (status_code(res) == 200) {
      parsed <- content(res, as = "parsed")
      return(parsed$choices[[1]]$message$content)
    } else {
      parsed_error <- content(res, as = "parsed")
      err_msg <- if(!is.null(parsed_error$error$message)) parsed_error$error$message else paste("HTTP", status_code(res))
      return(sprintf("Batch AI service unavailable. Error details: %s", err_msg))
    }
  }, error = function(e) {
    return(sprintf("Failed to connect to AI Advisor for batch summary. Error: %s", e$message))
  })
}

# ==================== UI INTERFACE ====================
ui <- navbarPage(
  title = div(
    tags$img(src = "https://cdn-icons-png.flaticon.com/512/2919/2919600.png",
             height = 30, style = "margin-right: 10px;"),
    "Augmented Synergy Advisor for RA Muscles: ASA-M"
  ),
  theme = shinytheme("flatly"),
  header = tags$head(
    tags$title("ASA-M: Augmented Synergy Advisor for RA Muscles"),
    tags$style(HTML("
      /* UI Custom Styles */
      .risk-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
      .low-risk { background-color: #eafaf1; border-left-color: #2ecc71; color: #27ae60; }
      .intermediate-risk { background-color: #fef9e7; border-left-color: #f1c40f; color: #d4ac0d; }
      .high-risk { background-color: #fdf2e9; border-left-color: #e67e22; color: #d35400; }
      .metric-card { background: #fff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 6px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
      .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
      .metric-label { font-size: 11px; text-transform: uppercase; color: #7f8c8d; margin-top: 5px; }
      .batch-summary-box { background: #fdfefe; border: 1px solid #d6dbdf; border-radius: 6px; padding: 15px; text-align: center; margin-bottom: 15px; }
      .summary-value { font-size: 22px; font-weight: bold; }
      .summary-label { font-size: 12px; color: #7f8c8d; }
      .risk-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; color: #fff; display: inline-block; }
      .badge-low { background-color: #2ecc71; }
      .badge-intermediate { background-color: #f1c40f; }
      .badge-high { background-color: #e67e22; }
      .driver-impact-up { color: #e74c3c; font-weight: bold; }
      .driver-impact-down { color: #2ecc71; font-weight: bold; }
    "))
  ),
  
  tabPanel(
    "Single Patient Assessment",
    icon = icon("user-md"),
    fluidRow(
      column(
        3,
        wellPanel(
          h4(icon("sliders-h"), "Clinical and Hematological Indicators", style = "color: #2c3e50;"),
          hr(style = "border-color: #bdc3c7;"),
          numericInputIcon("ANC", label = tags$span(icon("vial"), "ANC (10^9/L)"), value = 3.5, min = 0, max = 50, step = 0.1, icon = icon("temperature-low")),
          numericInputIcon("ALT", label = tags$span(icon("vial"), "ALT (U/L)"), value = 25, min = 0, max = 500, step = 1, icon = icon("chart-line")),
          numericInputIcon("AST", label = tags$span(icon("vial"), "AST (U/L)"), value = 22, min = 0, max = 500, step = 1, icon = icon("heartbeat")),
          numericInputIcon("Age", label = tags$span(icon("birthday-cake"), "Age (years)"), value = 65, min = 18, max = 120, step = 1, icon = icon("user-clock")),
          numericInputIcon("BMI", label = tags$span(icon("weight"), "BMI (kg/m²)"), value = 22.5, min = 10, max = 50, step = 0.1, icon = icon("balance-scale")),
          pickerInput("Gender", label = tags$span(icon("venus-mars"), "Gender"), choices = list("Male" = "1", "Female" = "2"), selected = "1"),
          br(),
          actionBttn("go_one", label = "Assess Risk", icon = icon("brain"), style = "gradient", color = "primary", block = TRUE, size = "lg"),
          hr()
        )
      ),
      column(
        9,
        uiOutput("prediction_result_ui"),
        
        conditionalPanel(
          condition = "output.show_drivers == true",
          hr(),
          h4(icon("robot"), "DeepSeek AI Clinical Assistant", style="color: #2980b9;"),
          div(
            class = "well",
            style = "background-color: #eaf2f8; border-left: 5px solid #2980b9; font-size: 15px; line-height: 1.6;",
            uiOutput("ai_agent_explanation_ui")
          ),
          br(),
          h4(icon("chart-bar"), "Key Drivers of Prediction"),
          div(
            class = "well key-drivers-table",
            DTOutput("key_drivers_table")
          ),
          br(),
          h5(icon("info-circle"), "Interpretation Guide"),
          div(
            class = "well",
            style = "background-color: #f8f9fa;",
            tags$ul(
              tags$li(tags$span(class = "driver-impact-up", "↑"), " indicates the feature increases low muscle mass risk"),
              tags$li(tags$span(class = "driver-impact-down", "↓"), " indicates the feature decreases low muscle mass risk"),
              tags$li("Driver 1: Most influential factor affecting prediction"),
              tags$li("Driver 2: Second most influential factor"),
              tags$li("Driver 3: Third most influential factor")
            )
          )
        )
      )
    )
  ),
  
  tabPanel(
    "Batch Assessment",
    icon = icon("users"),
    fluidRow(
      column(12, h3(icon("file-upload"), "Batch Patient Assessment"), p("Upload a CSV file containing multiple patients' clinical data for batch prediction."), hr())
    ),
    fluidRow(
      column(
        4,
        wellPanel(
          h4(icon("upload"), "Data Upload", style = "color: #2c3e50;"), hr(),
          div(
            class = "file-upload-box", style="background-color: #f8f9fa; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 8px; text-align: center;",
            h5(icon("cloud-upload-alt"), "Upload CSV File"),
            tags$ul(style="text-align: left; font-size: 13px;",
                    tags$li("ANC (10^9/L)"), tags$li("ALT (U/L)"), tags$li("AST (U/L)"),
                    tags$li("Gender (1=Male, 2=Female)"), tags$li("Age (years)"), tags$li("BMI (kg/m²)")
            ),
            fileInput("batch_file", label = NULL, accept = c(".csv", ".txt"), buttonLabel = "Browse...", placeholder = "No file selected"),
            downloadLink("download_template", "Download template: batch_template.csv")
          ),
          hr(),
          h5(icon("cog"), "Processing Options"),
          awesomeCheckbox("include_drivers", "Include Key Drivers Analysis", value = TRUE, status = "primary"),
          awesomeCheckbox("include_descriptive", "Include Descriptive Statistics", value = TRUE, status = "primary"),
          br(),
          actionBttn("go_batch", label = "Start Batch Prediction", icon = icon("play-circle"), style = "gradient", color = "success", block = TRUE, size = "lg"),
          br(),
          conditionalPanel(condition = "output.batch_result_available == true", downloadBttn("download_results", "Download Results", style = "gradient", color = "primary", block = TRUE))
        )
      ),
      column(
        8,
        uiOutput("batch_processing_ui"),
        conditionalPanel(
          condition = "output.batch_result_available == true",
          h4(icon("chart-pie"), "Batch Summary"),
          uiOutput("batch_summary_ui"),
          br(),
          h4(icon("robot"), "DeepSeek AI Batch Assistant", style = "color: #27ae60;"),
          div(
            class = "well",
            style = "background-color: #ebf5fb; border-left: 5px solid #27ae60; font-size: 15px; line-height: 1.6;",
            uiOutput("batch_ai_summary_ui")
          ),
          hr(),
          h4(icon("table"), "Detailed Predictions"),
          div(class = "well", style = "padding: 20px;", DTOutput("batch_results_table")),
          br(),
          conditionalPanel(
            condition = "input.include_descriptive == true",
            h4(icon("chart-bar"), "Descriptive Statistics"),
            uiOutput("descriptive_stats_ui")
          )
        )
      )
    )
  ),
  
  tabPanel(
    "About",
    icon = icon("info-circle"),
    fluidRow(
      column(
        8, offset = 2,
        div(
          class = "well", style = "padding: 30px;",
          h2(icon("hospital"), "ASA-M", style = "color: #2c3e50; text-align: center;"), hr(),
          
          h4("Augmented Synergy Advisor for RA Muscles (ASA-M)"),
          p(style = "font-size: 16px; line-height: 1.6;", 
            "The ASA-M system is an advanced machine learning-based tool designed specifically for patients with rheumatoid arthritis. It assesses the risk of developing low muscle mass by analyzing routinely available blood-based and clinical indicators. To further enhance clinical utility, the system integrates the DeepSeek Large Language Model (LLM) as an AI Clinical Assistant, which provides professional, automated interpretations based on the model's predictions and SHAP key driving factors."),
          
          br(),
          h4(icon("cogs"), "Technical Specifications"),
          tags$ul(
            style = "font-size: 15px; line-height: 1.8;",
            tags$li(strong("Machine Learning Model:"), " CatBoost with SHAP interpretability"),
            tags$li(strong("AI Clinical Assistant:"), " Powered by DeepSeek LLM for intelligent predictive summarization"),
            tags$li(strong("Features:"), " ANC, ALT, AST, Gender, Age, BMI"),
            tags$li(strong("Performance:"), " AUC = 0.8998, Accuracy = 81.52%")
          ),
          
          br(),
          h4(icon("shield-alt"), "Clinical Validation"),
          p(style = "font-size: 15px; line-height: 1.6;", 
            "The model has been validated on independent test datasets and demonstrates robust performance across diverse patient populations."),
          
          br(),
          h4(icon("handshake"), "Intended Use"),
          p(style = "font-size: 15px; line-height: 1.6;", 
            "This tool is intended for clinical research and screening purposes. It should not replace clinical judgment or comprehensive assessment.")
        )
      )
    )
  )
)

# ==================== SERVER LOGIC ====================
server <- function(input, output, session) {
  
  values <- reactiveValues(
    single_result = NULL, key_drivers = NULL, batch_results = NULL, batch_summary = NULL, ai_text = NULL, batch_ai_text = NULL
  )
  
  model_components <- reactive({
    required_files <- c("catboost_muscle_model.rds", "feature_cols.rds", "target_levels.rds", "threshold_asset.rds", "background_data.rds")
    missing_files <- required_files[!file.exists(file.path(MODEL_DIR, required_files))]
    if(length(missing_files) > 0) stop(paste("Missing required files:", paste(missing_files, collapse = ", ")))
    
    list(
      learner = readRDS(file.path(MODEL_DIR, "catboost_muscle_model.rds")),
      feature_cols = readRDS(file.path(MODEL_DIR, "feature_cols.rds")),
      target_levels = readRDS(file.path(MODEL_DIR, "target_levels.rds")),
      threshold_asset = readRDS(file.path(MODEL_DIR, "threshold_asset.rds")),
      background_data = readRDS(file.path(MODEL_DIR, "background_data.rds"))
    )
  })
  
  predict_single_patient <- function(params) {
    tryCatch({
      gender_processed <- ifelse(params$Gender %in% c("female", "f", "女", "woman", "2"), "2", "1")
      gender_factor <- factor(gender_processed, levels = c("1", "2"))
      
      input_data <- data.table(
        ANC = as.numeric(params$ANC), ALT = as.numeric(params$ALT), AST = as.numeric(params$AST),
        Gender = gender_factor, Age = as.numeric(params$Age), BMI = as.numeric(params$BMI),
        Muscle = factor(NA, levels = model_components()$target_levels)
      )
      
      task <- TaskClassif$new(id = "single_prediction", backend = input_data, target = "Muscle", positive = "Low")
      pred <- model_components()$learner$predict(task)
      prob_pos <- as.numeric(pred$prob[, "Low"])
      
      thresholds <- model_components()$threshold_asset$boot_ci_95
      risk_level <- if(prob_pos < thresholds[1]) "Low" else if(prob_pos < thresholds[2]) "Intermediate" else "High"
      
      X_new_for_shap <- input_data[, model_components()$feature_cols, with = FALSE]
      key_drivers_df <- compute_key_drivers(
        learner = model_components()$learner, X_new = X_new_for_shap,
        bg_X = model_components()$background_data, feature_cols = model_components()$feature_cols,
        target_levels = model_components()$target_levels
      )
      
      list(probability = round(prob_pos, 4), risk_level = risk_level, thresholds = thresholds, features = params, key_drivers = key_drivers_df)
    }, error = function(e) stop(sprintf("Prediction error: %s", e$message)))
  }
  
  predict_batch_patients <- function(data, include_drivers = TRUE) {
    tryCatch({
      required_cols <- c("ANC","ALT","AST","Gender","Age","BMI")
      missing_cols <- setdiff(required_cols, colnames(data))
      if(length(missing_cols) > 0) stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
      
      n_patients <- nrow(data)
      data_processed <- copy(data)
      if(is.character(data_processed$Gender)) {
        data_processed$Gender <- ifelse(tolower(data_processed$Gender) %in% c("female", "f", "女", "woman", "2"), "2", "1")
      }
      data_processed$Gender <- factor(data_processed$Gender, levels = c("1", "2"),labels = c("1", "2"))
      
      numeric_cols <- c("ANC","ALT","AST","Age","BMI")
      for(col in numeric_cols) data_processed[[col]] <- as.numeric(data_processed[[col]])
      data_processed$Muscle <- factor(NA, levels = model_components()$target_levels)
      
      progress <- shiny::Progress$new()
      progress$set(message = "Processing batch predictions", value = 0)
      on.exit(progress$close())
      
      results <- data.table(
        PatientID = if("PatientID" %in% colnames(data)) data$PatientID else paste0("Patient_", 1:n_patients),
        ANC = data_processed$ANC, ALT = data_processed$ALT, AST = data_processed$AST,
        Gender = data_processed$Gender, Age = data_processed$Age, BMI = data_processed$BMI,
        Probability = numeric(n_patients), RiskLevel = character(n_patients), stringsAsFactors = FALSE
      )
      if(include_drivers) {
        results$Driver1 <- character(n_patients); results$Driver2 <- character(n_patients); results$Driver3 <- character(n_patients)
      }
      
      thresholds <- model_components()$threshold_asset$boot_ci_95
      task_batch <- TaskClassif$new(id = "batch_prediction", backend = data_processed, target = "Muscle", positive = "Low")
      task_batch$select(model_components()$feature_cols)
      preds_batch <- model_components()$learner$predict(task_batch)
      probs <- as.numeric(preds_batch$prob[, "Low"])
      
      progress$set(message = "Calculating predictions", value = 0.3)
      results$Probability <- round(probs, 4)
      results$RiskLevel <- ifelse(probs < thresholds[1], "Low", ifelse(probs < thresholds[2], "Intermediate", "High"))
      
      if(include_drivers) {
        progress$set(message = "Calculating key drivers", value = 0.6)
        X_new_for_shap <- data_processed[, model_components()$feature_cols, with = FALSE]
        key_drivers_df <- compute_key_drivers(
          learner = model_components()$learner, X_new = X_new_for_shap,
          bg_X = model_components()$background_data, feature_cols = model_components()$feature_cols,
          target_levels = model_components()$target_levels
        )
        if(!is.null(key_drivers_df)) {
          results$Driver1 <- key_drivers_df$driver1
          results$Driver2 <- key_drivers_df$driver2
          results$Driver3 <- key_drivers_df$driver3
        }
      }
      
      progress$set(message = "Finalizing results", value = 0.9)
      summary_stats <- list(
        total_patients = n_patients, low_risk = sum(results$RiskLevel == "Low"),
        intermediate_risk = sum(results$RiskLevel == "Intermediate"), high_risk = sum(results$RiskLevel == "High"),
        mean_probability = mean(results$Probability), median_probability = median(results$Probability),
        sd_probability = sd(results$Probability), thresholds = thresholds
      )
      
      descriptive_stats <- if(n_patients > 0) {
        list(
          ANC = summary(data_processed$ANC), ALT = summary(data_processed$ALT), AST = summary(data_processed$AST),
          Gender_dist = table(data_processed$Gender), Age = summary(data_processed$Age), BMI = summary(data_processed$BMI)
        )
      } else { NULL }
      
      progress$set(value = 1)
      list(results = results, summary = summary_stats, descriptive = descriptive_stats)
      
    }, error = function(e) stop(sprintf("Batch prediction error: %s", e$message)))
  }
  
  observeEvent(input$go_one, {
    showModal(modalDialog(title = "Processing...", "Analyzing clinical parameters and consulting DeepSeek...", footer = NULL, easyClose = FALSE))
    
    result <- tryCatch({
      predict_single_patient(list(ANC = input$ANC, ALT = input$ALT, AST = input$AST, Gender = input$Gender, Age = input$Age, BMI = input$BMI))
    }, error = function(e) list(error = e$message))
    
    if(!is.null(result$error)) {
      removeModal()
      showNotification(paste("Error:", result$error), type = "error", duration = 5)
    } else {
      values$single_result <- result
      values$key_drivers <- result$key_drivers
      
      ai_response <- generate_ai_interpretation(result$risk_level, result$probability, result$key_drivers)
      values$ai_text <- ai_response
      removeModal()
    }
  })
  
  observeEvent(input$go_batch, {
    req(input$batch_file)
    showModal(modalDialog(
      title = "Processing Batch Data...",
      div(class = "text-center", h4("Analyzing multiple patient records"), p("This may take a moment depending on the number of patients."), br(), div(class = "spinner-border text-primary", role = "status", span(class = "sr-only", "Loading..."))),
      footer = NULL, easyClose = FALSE, size = "m"
    ))
    
    tryCatch({
      uploaded_data <- fread(input$batch_file$datapath, stringsAsFactors = FALSE)
      batch_result <- predict_batch_patients(data = uploaded_data, include_drivers = input$include_drivers)
      
      values$batch_results <- batch_result$results
      values$batch_summary <- batch_result$summary
      values$batch_descriptive <- batch_result$descriptive
      
      batch_ai_response <- generate_batch_ai_interpretation(batch_result$summary)
      values$batch_ai_text <- batch_ai_response
      
      removeModal()
      showNotification("Batch prediction completed successfully!", type = "message", duration = 5)
    }, error = function(e) {
      removeModal()
      showNotification(paste("Error:", e$message), type = "error", duration = 10)
    })
  })
  
  output$download_template <- downloadHandler(
    filename = function() { "batch_template.csv" },
    content = function(file) {
      template_data <- data.table(
        PatientID = c("P001", "P002", "P003"), ANC = c(3.5, 4.0, 2.8), ALT = c(25, 30, 40),
        AST = c(22, 24, 26), Gender = c("1", "2", "1"), Age = c(65, 70, 75), BMI = c(22.5, 24.0, 21.0)
      )
      fwrite(template_data, file)
    }
  )
  
  output$download_results <- downloadHandler(
    filename = function() { paste0("batch_predictions_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv") },
    content = function(file) { req(values$batch_results); fwrite(values$batch_results, file) }
  )
  
  output$key_drivers_table <- renderDT({
    req(values$key_drivers)
    driver_descriptions <- data.table(
      Rank = c("Driver 1", "Driver 2", "Driver 3"),
      Description = c(values$key_drivers$driver1[1], values$key_drivers$driver2[1], values$key_drivers$driver3[1])
    )
    driver_descriptions <- driver_descriptions[!is.na(Description)]
    if(nrow(driver_descriptions) == 0) return(NULL)
    
    styled_descriptions <- sapply(driver_descriptions$Description, function(desc) {
      if(grepl("↑", desc)) { gsub("↑", '<span class="driver-impact-up">↑</span>', desc) }
      else if(grepl("↓", desc)) { gsub("↓", '<span class="driver-impact-down">↓</span>', desc) }
      else { desc }
    })
    driver_descriptions$Description <- styled_descriptions
    
    datatable(driver_descriptions, options = list(paging = FALSE, searching = FALSE, info = FALSE, ordering = FALSE, dom = 't'), class = 'cell-border stripe', rownames = FALSE, colnames = c("Rank", "Key Driver (Feature & Impact Direction)"), escape = FALSE) %>%
      formatStyle('Rank', fontWeight = 'bold', backgroundColor = '#f8f9fa')
  })
  
  output$batch_processing_ui <- renderUI({
    if(is.null(input$batch_file)) {
      div(class = "text-center", style = "padding: 50px;", icon("users", "fa-5x", style = "color: #ecf0f1; margin-bottom: 20px;"), h4("Upload a CSV file to begin batch assessment"), p("Use the upload area on the left to select your patient data file"))
    }
  })
  
  output$batch_summary_ui <- renderUI({
    req(values$batch_summary)
    summary <- values$batch_summary
    tagList(
      fluidRow(
        column(3, div(class = "batch-summary-box", div(class = "summary-value", summary$total_patients), div(class = "summary-label", "Total Patients"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-low", summary$low_risk)), div(class = "summary-label", "Low Risk"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-intermediate", summary$intermediate_risk)), div(class = "summary-label", "Intermediate Risk"))),
        column(3, div(class = "batch-summary-box", div(class = "summary-value", span(class = "risk-badge badge-high", summary$high_risk)), div(class = "summary-label", "High Risk")))
      ),
      fluidRow(
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$mean_probability * 100, 1), "%")), div(class = "summary-label", "Mean Probability"))),
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$median_probability * 100, 1), "%")), div(class = "summary-label", "Median Probability"))),
        column(4, div(class = "batch-summary-box", div(class = "summary-value", paste0(round(summary$sd_probability * 100, 2), "%")), div(class = "summary-label", "Std Deviation")))
      )
    )
  })
  
  output$batch_results_table <- renderDT({
    req(values$batch_results)
    results_display <- copy(values$batch_results)
    results_display$Probability <- paste0(round(results_display$Probability * 100, 1), "%")
    results_display$Gender <- ifelse(results_display$Gender == "1", "Male", "Female")
    results_display$RiskLevel <- sapply(results_display$RiskLevel, function(risk) {
      badge_class <- switch(risk, "Low" = "badge-low", "Intermediate" = "badge-intermediate", "High" = "badge-high")
      paste0('<span class="risk-badge ', badge_class, '">', risk, '</span>')
    })
    
    datatable(results_display, options = list(pageLength = 10, scrollX = TRUE, dom = 'lfrtip', columnDefs = list(list(className = 'dt-center', targets = '_all'))), class = 'cell-border stripe hover', rownames = FALSE, escape = FALSE, selection = 'none') %>%
      formatStyle('RiskLevel', fontWeight = 'bold') %>% formatRound(columns = c('ANC','ALT','AST','Age','BMI'), digits = 1)
  })
  
  output$descriptive_stats_ui <- renderUI({
    req(values$batch_descriptive)
    fluidRow(column(6, h5("Continuous Variables Summary"), verbatimTextOutput("cont_summary")), column(6, h5("Categorical Variables Distribution"), verbatimTextOutput("cat_summary")))
  })
  
  output$cont_summary <- renderPrint({
    req(values$batch_descriptive)
    cat("ANC (10^9/L):\n"); print(values$batch_descriptive$ANC)
    cat("\nALT (U/L):\n"); print(values$batch_descriptive$ALT)
    cat("\nAST (U/L):\n"); print(values$batch_descriptive$AST)
    cat("\nAge (years):\n"); print(values$batch_descriptive$Age)
    cat("\nBMI (kg/m^2):\n"); print(values$batch_descriptive$BMI)
  })
  
  output$cat_summary <- renderPrint({
    req(values$batch_descriptive)
    cat("Gender Distribution:\n")
    gender_labels <- c("Male", "Female")
    gender_counts <- values$batch_descriptive$Gender_dist
    names(gender_counts) <- gender_labels[as.numeric(names(gender_counts))]
    print(gender_counts)
    cat("\nRisk Level Distribution:\n")
    if(!is.null(values$batch_results)) print(table(values$batch_results$RiskLevel))
  })
  
  output$batch_result_available <- reactive({ !is.null(values$batch_results) })
  outputOptions(output, "batch_result_available", suspendWhenHidden = FALSE)
  
  output$ai_agent_explanation_ui <- renderUI({
    req(values$ai_text)
    HTML(gsub("\n", "<br/>", values$ai_text))
  })
  
  output$batch_ai_summary_ui <- renderUI({
    req(values$batch_ai_text)
    HTML(gsub("\n", "<br/>", values$batch_ai_text))
  })
  
  output$prediction_result_ui <- renderUI({
    if(is.null(values$single_result)) {
      return(div(class = "text-center", style = "padding: 50px;", icon("stethoscope", "fa-5x", style = "color: #ecf0f1; margin-bottom: 20px;"), h4("Enter patient parameters to begin assessment"), p("Fill in the clinical values on the left and click 'Assess Risk'")))
    }
    
    result <- values$single_result
    risk_class <- switch(result$risk_level, "Low" = "low-risk", "Intermediate" = "intermediate-risk", "High" = "high-risk")
    risk_icon <- switch(result$risk_level, "Low" = icon("smile", class = "fa-3x", style = "color: #28a745;"), "Intermediate" = icon("meh", class = "fa-3x", style = "color: #ffc107;"), "High" = icon("frown", class = "fa-3x", style = "color: #dc3545;"))
    
    tagList(
      div(
        class = paste("risk-box", risk_class),
        fluidRow(
          column(2, class = "text-center", risk_icon),
          column(10, h3(paste(result$risk_level, "Risk"), style = "margin-top: 0;"), p(strong("Probability: "), paste0(round(result$probability * 100, 1), "%")))
        ),
        hr(),
        fluidRow(
          column(4, div(class = "metric-card", div(class = "metric-value", paste0(round(result$probability * 100, 1), "%")), div(class = "metric-label", "Low Muscle Mass Probability"))),
          column(4, div(class = "metric-card", div(class = "metric-value", round(result$thresholds[1], 3)), div(class = "metric-label", "Low Risk Threshold"))),
          column(4, div(class = "metric-card", div(class = "metric-value", round(result$thresholds[2], 3)), div(class = "metric-label", "High Risk Threshold")))
        ),
        hr(),
        
        h4(icon("chart-pie"), "Risk Interpretation"),
        tags$ul(
          tags$li(strong("Low Risk (<", round(result$thresholds[1], 3), "):"), " Low probability of low muscle mass. Routine monitoring recommended."),
          tags$li(strong("Intermediate Risk (", round(result$thresholds[1], 3), "-", round(result$thresholds[2], 3), "):"), " Moderate risk. Consider further assessment and lifestyle interventions."),
          tags$li(strong("High Risk (>", round(result$thresholds[2], 3), "):"), " High probability of low muscle mass. Comprehensive assessment recommended.")
        ),
        hr(),
        h4(icon("notes-medical"), "Input Parameters"),
        fluidRow(
          column(2, strong("ANC:"), br(), result$features$ANC, " ×10^9/L"),
          column(2, strong("ALT:"), br(), result$features$ALT, " U/L"),
          column(2, strong("AST:"), br(), result$features$AST, " U/L"),
          column(2, strong("Age:"), br(), result$features$Age, " years"),
          column(2, strong("BMI:"), br(), result$features$BMI, " kg/m²"),
          column(2, strong("Gender:"), br(), ifelse(result$features$Gender == "1", "Male", "Female"))
        )
      )
    )
  })
  
  output$show_drivers <- reactive({
    !is.null(values$single_result) && !is.null(values$key_drivers) && any(!is.na(values$key_drivers$driver1))
  })
  outputOptions(output, "show_drivers", suspendWhenHidden = FALSE)
}

shinyApp(ui = ui, server = server)