# ASA-M: Augmented Synergy Advisor for RA Muscles

## Project Introduction

The ASA-M system is an advanced machine learning-based tool designed specifically for patients with rheumatoid arthritis. It assesses the risk of developing low muscle mass by analyzing routinely available blood-based and clinical indicators. This application employs a CatBoost model trained on comprehensive blood and clinical data to provide accurate and interpretable predictions of low muscle mass risk.

## Main Functions

* **Personalized Risk Assessment:** Obtain personalized risk prediction by inputting 6 parameters.
* **Interpretable Analysis:** Key driver analysis based on SHAP values.
* **Multi-level Risk Classification:** Three-tier risk assessment (low, medium, high).
* **Clinical Guidance:** Provide management recommendations based on risk level.
* **User-friendly Interface:** Intuitive parameter input and result display.

## Machine Learning Model

* **Algorithm:** CatBoost
* **Features:** ALT, Age, BMI, Gender, Hemoglobin, White Blood Cell Count
* **Performance:** AUC = 0.8001, Accuracy = 73.91%
* **Threshold Optimization:** Bootstrap Confidence Interval Based on Youden's J Statistic
* **External Validation:** Validated using NHANES 2011-2018 data.

## Usage Instructions

### 1. Patient Risk Assessment

1. Enter the patient's clinical parameters in the input panel:
    * ANC (U/L)
    * Age (years)
    * BMI (kg/m²)
    * Gender (male or female)
    * Hemoglobin (g/L)
    * White Blood Cell Count (×10⁹/L)
2. Click the **'Assess Risk'** button.
3. View the risk assessment results in the output panel.

### 2. Result Interpretation

The system will provide:

* **Risk Level:** Low Risk, Medium Risk, or High Risk.
* **Probability of Muscle Mass Reduction:** A probability score ranging from 0-100%.
* **Key Drivers:** The top 3 features influencing the prediction and their directions (positive/negative impact).
* **Clinical Recommendations:** Management guidance tailored to the identified risk level.

## Model Development

### Feature Selection

Based on clinical feasibility and predictive performance, a total of 6 features were finally selected:

1. ALT (Alanine Aminotransferase)
2. Age
3. BMI (Body Mass Index)
4. Gender
5. Hemoglobin
6. White Blood Cell Count

### Model Selection

* **Benchmark:** This study selected seven representative classifiers for benchmark testing: Neural Network (nnet), Categorical Boosting (CatBoost), Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes, and LightGBM. The overall dataset was divided into a training set and a test set in a 7:3 ratio. Model evaluation and selection were conducted through a nested resampling process, employing 10-fold outer cross-validation and 10-fold inner cross-validation. Prior to model training, all data features were preprocessed using a standardization pipeline. For models requiring hyperparameter tuning, this study utilized an automated tuner for hyperparameter optimization. The optimization strategy was random search, with 10 parameter combinations evaluated per batch. The AUC (classif.auc) from the inner 10-fold cross-validation served as the performance evaluation metric, and the search was terminated after evaluating 100 parameter combinations.

## Clinical Application

### Target Population

* Patients with rheumatoid arthritis.
* Clinical research and screening scenarios.

### Usage Restrictions

* This tool is for clinical research and screening purposes only.
* It cannot replace a comprehensive medical assessment.
* Results should be combined with clinical judgment.

## Contact Information

If you have any questions or suggestions, please contact us through the following ways:<br>
email: <17200623640@163.com>