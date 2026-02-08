# ASA-M: Augmented Synergy Advisor for RA Muscles

## Project Introduction

The ASA-M system is an advanced AI agent specifically designed for patients with rheumatoid arthritis. It assesses the risk of developing low muscle mass by analyzing routinely available blood-based and clinical indicators. This application employs a CatBoost model trained on comprehensive blood and clinical data to provide accurate and interpretable predictions of low muscle mass risk.

## Main Functions

* **Personalized Risk Assessment:** Obtain personalized risk prediction by inputting 6 parameters.
* **Interpretable Analysis:** Key driver analysis based on SHAP values.
* **Multi-level Risk Classification:** Three-tier risk assessment (low, medium, high).
* **Clinical Guidance:** Provide management recommendations based on risk level.
* **User-friendly Interface:** Intuitive parameter input and result display.

## Machine Learning Model

* **Algorithm:** CatBoost
* **Features:** ANC, ALT, AST, Gender, Age, BMI
* **Performance:** AUC = 0.8998, Accuracy = 81.52%
* **Threshold Optimization:** Bootstrap Confidence Interval Based on Youden's J Statistic

## Usage Instructions
### 1. Single Patient Assessment
Ideal for individual clinical assessments.
* Input Parameters: Enter the patient's clinical data:Demographics: Gender (Male/Female), Age (years), BMI (kg/m²). Lab Values: ANC ($\times 10^{9}$/L), ALT (U/L), AST (U/L).
* Assess: Click the "Assess Risk" button.
* Interpret Results:View the Risk Level (Low/Intermediate/High) and probability score. Review the Key Drivers table to understand which specific features contributed most to the prediction (e.g., "AST ↑" indicates elevated AST increased the risk).
### 2. Batch Assessment
Process multiple patient records simultaneously using a CSV file.
* Prepare Data:Download the CSV Template from the sidebar to ensure correct formatting. Your CSV file must contain the following columns: ANC, ALT, AST, Age, BMI, Gender. Optional: Include a Patient ID column for tracking.
* Data Formatting:Gender: Use 1 for Male and 2 for Female (text inputs like "F", "Female" are also supported).Numeric Values: Ensure no missing values in clinical columns.
* Upload & Analyze: Upload the file and check "Include Key Drivers" if detailed interpretability is needed.
* Assess: Click the "Start Batch Prediction" button.
* Export: View the summary dashboard and click "Download Results" to get the full prediction report.

## Clinical Application

### Target Population

* Patients with rheumatoid arthritis.
* Clinical research and screening scenarios.

### Usage Restrictions

* This tool is for clinical research and screening purposes only.
* It cannot replace a comprehensive medical assessment.
* Results should be combined with clinical judgment.
