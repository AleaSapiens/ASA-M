# ASA-M: Augmented Synergy Advisor for RA Muscles

## Project Overview
ASA-M is a clinical decision support tool designed to assess the risk of low muscle mass (LMM) in patients with rheumatoid arthritis (RA). The system integrates a CatBoost predictive model with the DeepSeek Large Language Model (LLM) to provide automated clinical interpretations based on patient-specific data and SHAP (SHapley Additive exPlanations) values.

## Core Features
*   **Risk Prediction:** Accurate assessment of low muscle mass risk using clinical and laboratory indicators.
*   **AI Clinical Assistant:** Powered by DeepSeek LLM for intelligent summarization and clinical insights.
*   **Explainable AI:** Integration of SHAP to identify key drivers behind each prediction.
*   **Dual Assessment Modes:** Supports individual patient entry and high-throughput batch processing via CSV.

## Technical Specifications
*   **Algorithm:** CatBoost
*   **Interpretability:** SHAP Values
*   **LLM Integration:** DeepSeek (API-based interpretation layer)
*   **Input Features:** ANC, ALT, AST, Gender, Age, BMI
*   **Performance Metrics:** 
    *   AUC: 0.8998
    *   Accuracy: 81.52%

### Risk Classification
The system categorizes patients into three risk levels based on the predicted probability ($P$):
1. Low Risk: $P < 0.313$
2. Intermediate Risk: $0.313 <= P < 0.579$
3. High Risk: $P >= 0.579$

## Usage Instructions
### API Key Configuration
The system integrates the DeepSeek LLM to generate clinical interpretations. Please configure your API key using one of the following methods:
Recommended: Add DEEPSEEK_API_KEY="your_api_key_here" to your system environment variables. Run the following command in R to open your `.Renviron` file and securely store your API keys or environment constants:
> ```R
> usethis::edit_r_environ()
> ```
Manual: Open the script and replace the placeholder at the beginning of the generate_ai_interpretation and generate_batch_ai_interpretation functions: api_key <- "your_actual_key".

### Model Files Deployment
The prediction engine requires specific pre-trained components. Ensure a models/ directory exists in the project root with the following 5 essential serialized files:

catboost_muscle_model.rds: The core CatBoost classification model.

feature_cols.rds: Stores the required input feature names and their strict internal order (ANC, ALT, AST, Gender, Age, BMI).

target_levels.rds: Defines the classification labels to ensure logic consistency.

threshold_asset.rds: Contains the risk stratification thresholds calibrated with 95% confidence intervals.

background_data.rds: A reference dataset sourced from the training cohort, used to calculate SHAP values.

### Single Assessment
1. Input Gender (Male/Female), Age (years), BMI (kg/m²), ANC ($\times 10^{9}$/L), ALT (U/L), AST (U/L).
2. Click "Assess Risk" to generate a probability score and risk level (low, intermediate, high).
3. Review the AI-generated interpretation for clinical context.

### Batch Assessment
1. Upload a CSV file (include ANC, ALT, AST, Gender, Age, BMI) following the provided template and click "Start Batch Prediction".
2. Select "Include Key Drivers Analysis" for detailed SHAP-based analysis. Select "Include Descriptive Statistics" to add summary statistics to the output.
3. Export results in a comprehensive summary table.

## Clinical Application
*   **Target Population:** Patients with rheumatoid arthritis.
*   **Intended Use:** Designed for clinical research and auxiliary screening.
*   **Disclaimer:** This tool is not a replacement for professional clinical judgment or formal diagnostic procedures (such as DXA). Results should be interpreted by healthcare professionals.