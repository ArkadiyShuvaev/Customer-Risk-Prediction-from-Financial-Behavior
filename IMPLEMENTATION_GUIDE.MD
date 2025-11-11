## **📁 Complete Project File Structure**

```
credit-scoring-ml-project/
│
├── README.md                         # Project overview, setup instructions
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
│
├── data/
│   ├── raw/
│   │   ├── credit_score_raw.csv              # 100% full dataset
│   │   ├── credit_score_truncated_raw.csv    # The dataset with one row per customer (the initial dataset includes time-series to `credit_score` in the next month)
│   │   ├── train_full.csv                    # 80% of truncated (for development)
│   │   └── test_holdout.csv                  # 20% of truncated (LOCKED until end)
│   │
│   ├── interim/
│   │   ├── imputation_stats.json             # From train_full
│   │   ├── cleaning_config.json              # From train_full
│   │   ├── train_full_cleaned.csv
│   │   └── test_holdout_cleaned.csv          # (using train_full stats)
│   │
│   └── processed/
│       ├── train_full_processed.csv          # For K-Fold and final training
│       └── test_holdout_processed.csv        # For final evaluation ONLY
│
├── notebooks/
│   ├── 01_data_splitting_eda.ipynb           # Data splitting & exploration
│   ├── 02_data_cleaning_feature_engineering.ipynb    # Data cleaning & feature creation
│   └── 03_model_training_evaluation.ipynb    # Model building & assessment
│
├── src/                              # Reusable Python modules (optional)
│   ├── __init__.py
│   ├── data_processing.py           # Data cleaning functions
│   ├── feature_engineering.py       # Feature creation functions
│   └── model_utils.py               # Model training utilities
│
├── models/                           # Saved trained models
│   ├── baseline_logistic_regression.pkl
│   ├── random_forest_v1.pkl
│   ├── xgboost_v1.pkl
│   ├── lightgbm_v1.pkl
│   ├── best_model.pkl               # Final selected model
│   └── feature_scaler.pkl           # Saved scaler/transformer
│
├── outputs/
│   ├── figures/                     # Plots and visualizations
│   │   ├── eda/
│   │   │   ├── target_distribution.png
│   │   │   ├── correlation_heatmap.png
│   │   │   ├── age_distribution.png
│   │   │   └── income_boxplot.png
│   │   │
│   │   ├── feature_engineering/
│   │   │   ├── feature_importance.png
│   │   │   └── derived_features_dist.png
│   │   │
│   │   └── model_evaluation/
│   │       ├── confusion_matrix_rf.png
│   │       ├── roc_curves_comparison.png
│   │       ├── feature_importance_final.png
│   │       └── model_comparison_metrics.png
│   │
│   └── reports/                     # Analysis reports
│       ├── data_quality_summary.txt
│       ├── eda_insights.md
│       ├── feature_analysis.md
│       └── final_model_report.md
│
└── config/                          # Configuration files (optional)
    └── config.yaml                  # Project parameters and settings
```

---

## **🔄 Data Splitting Strategy: Hold-Out Test Set vs K-Fold Cross-Validation**

### **Two Types of Splits (They Serve Different Purposes):**

#### **1. Hold-Out Test Set (Final Evaluation)**
**Purpose:** Simulate completely unseen production data
**When:** Set aside at the very beginning, NEVER touch until final evaluation
**Size:** 15-20% of total data

#### **2. K-Fold Cross-Validation (Model Development)**
**Purpose:** Evaluate model performance during development, tune hyperparameters
**When:** During model training and hyperparameter tuning
**Applied to:** The training set only (NOT the hold-out test set)

---

### **📊 Visual Representation:**

```
Original Dataset (100%)
│
├─ Hold-Out Test Set (20%) ─────────────┐
│  LOCKED AWAY UNTIL FINAL EVALUATION   │ ← Used ONCE at the very end
│                                       │
└─ Training Set (80%) ──────────────────┤
   │                                    │
   └─ K-Fold Cross-Validation:          │
      │                                 │
      ├─ Fold 1: Train (64%) | Val (16%)│ ← Used during development
      ├─ Fold 2: Train (64%) | Val (16%)│
      ├─ Fold 3: Train (64%) | Val (16%)│
      ├─ Fold 4: Train (64%) | Val (16%)│
      └─ Fold 5: Train (64%) | Val (16%)│
```

---

## **📓 Notebook 1: Data Splitting & EDA**

**File Name:** `01_data_splitting_eda.ipynb`

### **Detailed Steps:**

#### **Section 1: Setup & Data Loading**
1. ✅ Import necessary libraries (pandas, numpy, matplotlib, seaborn, sklearn)
2. ✅ Set display options and random seed for reproducibility
3. ✅ Load raw data from `data/raw/credit_score_raw.csv`
4. ✅ Display first few rows and basic information

#### **Section 2: Hold-Out Test Set Split** ⭐ **CRITICAL**
**Purpose:** Create final evaluation set that is NEVER touched during development

1. ✅ Separate features (X) and target (y)
2. ✅ Perform stratified train-test split (80/20)
3. ✅ Use stratify parameter on Credit_Score to maintain class distribution
4. ✅ Set random_state for reproducibility
5. ✅ Save splits:
   - `data/raw/train_full.csv` (80% - use for everything)
   - `data/raw/test_holdout.csv` (20% - LOCKED until final evaluation)
6. ✅ Document: "test_holdout will NOT be used until final evaluation in Notebook 3"

### **Section 3-10: Standard EDA Steps** (on train_full ONLY)

#### **Section 3: Initial Data Assessment**
1. ✅ Check dataset shape (rows, columns)
2. ✅ Display data types for all columns
3. ✅ Generate summary statistics using describe()
4. ✅ Identify columns with missing values and their percentages
5. ✅ Check for duplicate rows (especially customer_id + month combinations)
6. ⚠️ Document initial findings in markdown cell

#### **Section 4: Missing Value Analysis**
1. ✅ Create visualization showing missing value patterns
2. ✅ missing value distribution by column
3. Determine missingness type (MCAR, MAR, MNAR)
4. Document strategy for each column:
   - **name:** Impute with "Unknown" or drop (not predictive)
   - **ssn:** Drop column (privacy concern, not useful for modeling)
   - **occupation:** Impute with mode or "Unknown"
   - **age:** Impute with median (check distribution first)
   - **annual_income:** Impute with median
   - **monthly_inhand_salary:** Impute with median or derive from annual_income
   - **num_of_loan:** Impute with 0 (missing = no loans)
   - **num_of_delayed_payment:** Impute with 0 (missing = no delays)

#### **Section 5: Outlier Detection & Treatment**
1. Identify outliers using boxplots for numerical features
2. Define acceptable ranges:
   - **age:** 18-80 years (flag values outside)
   - **num_bank_accounts:** 0-10 (cap extreme values)
   - **num_credit_card:** 0-15 (cap extreme values)
   - **interest_rate:** 0-36% (flag negatives)
   - **num_of_loan:** 0-15 (cap extreme values)
   - **delay_from_due_date:** 0-150 days (cap extreme values)
   - **num_of_delayed_payment:** 0-30 (cap extreme values)
3. Apply capping using percentile method (1st-99th percentile)
4. Replace negative values with 0 where logically applicable
5. Document outliers removed and transformation applied

#### **Section 6: Data Consistency Checks**
1. Validate annual_income vs monthly_inhand_salary relationship
2. Check if monthly_inhand_salary * 12 ≈ annual_income (±20% tolerance)
3. Flag inconsistent records for review
4. Standardize string columns (trim spaces, consistent case)
5. Verify month values are valid (January-August)

#### **Section 7: Duplicate Handling**
1. Identify exact duplicates
2. Check for duplicates based on customer_id + month
3. Decide on removal or aggregation strategy
4. Remove duplicates and document count

#### **Section 8: Target Variable Analysis**
1. Check target variable (Credit_Score) distribution
2. Count and percentage for each class (Good, Poor, Standard)
3. Create bar plot and pie chart for visualization
4. Check for class imbalance
5. Document if resampling strategies will be needed

#### **Section 9: Exploratory Data Analysis - Numerical Features**
1. Create histograms for all numerical features
2. Generate boxplots to visualize distributions and outliers
3. Check for skewness and kurtosis
4. Create correlation matrix heatmap
5. Analyze correlation with target variable (encode target numerically)
6. Identify highly correlated features (multicollinearity check)
7. Document key findings for each feature

#### **Section 10: Exploratory Data Analysis - Categorical Features**
1. Generate value counts for occupation
2. Create bar plots for categorical distributions
3. Analyze month distribution
4. Create cross-tabulation between categorical features and target
5. Generate stacked bar plots showing relationship with Credit_Score
6. Identify rare categories (frequency < 5%)
7. Document insights on predictive categorical features

#### **Section 11: Save Configuration & Insights**
1. Save imputation statistics to `data/interim/imputation_stats.json`
2. Save cleaning configuration to `data/interim/cleaning_config.json`
3. Export EDA visualizations to `outputs/figures/eda/`
4. Create summary markdown document with insights
5. Save to `outputs/reports/eda_insights.md`

---

## **📓 Notebook 2: Data Cleaning & Feature Engineering**

**File Name:** `02_data_cleaning_feature_engineering.ipynb`

### **Detailed Steps:**

#### **Section 1: Load Data & Statistics**
1.  Import necessary libraries
2.  Load both splits:
   - `data/raw/train_full.csv`
   - `data/raw/test_holdout.csv`
3.  Load statistics calculated from train_full:
   - `data/interim/imputation_stats.json`
   - `data/interim/cleaning_config.json`

#### **Section 2: Clean Training Data**
1.  Apply cleaning using statistics from train_full
2.  Handle missing values using train_full medians/modes
3.  Apply outlier capping using train_full percentiles
4.  Perform data consistency checks
5.  Save cleaned training data to `data/interim/train_full_cleaned.csv`

#### **Section 3: Clean Test Data** ⭐ **USING TRAINING STATISTICS**
1.  Apply SAME transformations to test_holdout using train_full statistics
2.  Use train_full imputation values for missing data
3.  Use train_full capping ranges for outliers
4.  Save cleaned test data to `data/interim/test_holdout_cleaned.csv`

#### **Section 4: Feature Engineering on Training Data**
1.  Create derived features (fit on train_full only):
   - avg_loan_amount, debt_ratio, payment_delay_ratio, etc.
2.  Perform binning operations (using train_full distributions)
3.  Fit encoders on training categorical data
4.  Create interaction features
5.  Fit scalers on training numerical features
6.  Save all transformers to `models/` directory
7.  Save processed training data to `data/processed/train_full_processed.csv`

#### **Section 5: Feature Engineering on Test Data** ⭐ **USING TRAINING TRANSFORMERS**
1.  Apply SAME derived features to test_holdout
2.  Transform test categorical data using fitted encoders
3.  Scale test numerical data using fitted scalers
4.  Save processed test data to `data/processed/test_holdout_processed.csv`

#### **Section 6: Final Validation**
1.  Verify no data leakage occurred
2.  Check feature consistency between train and test
3.  Create feature engineering report
4.  Save to `outputs/reports/feature_analysis.md`

---

## **📓 Notebook 3: Model Training & Evaluation**

**File Name:** `03_model_training_evaluation.ipynb`

### **Detailed Steps:**

#### **Section 1: Setup & Load Processed Data**
1.  Import necessary libraries (sklearn, xgboost, lightgbm, optuna)
2.  Import visualization libraries (matplotlib, seaborn, shap)
3.  Set random seed for reproducibility
4.  Load processed datasets:
   - `data/processed/train_full_processed.csv`
   - `data/processed/test_holdout_processed.csv`
5.  Separate features and target for both sets
6.  Verify data shapes and feature consistency

#### **Section 2: Model Development with K-Fold CV** ⭐ **THIS IS WHERE K-FOLD HAPPENS**
**Use K-Fold on train_full for model development**

1.  Define K-Fold strategy (Stratified K-Fold, k=5)
2.  Implement cross-validation for baseline models
3.  Calculate average CV performance metrics
4.  Use CV scores for model comparison and selection

#### **Section 3-6: Individual Model Training** (with K-Fold CV)
- Baseline Model - Logistic Regression
- Random Forest Classifier  
- XGBoost Classifier
- LightGBM Classifier

**For each model:**
1.  Use K-Fold CV for hyperparameter tuning
2.  Select best parameters based on CV performance
3.  Train final model on entire train_full
4.  Save model to `models/` directory

#### **Section 7: Final Evaluation on Hold-Out Test Set** ⭐ **FIRST TIME USING TEST**
**THIS IS THE ONLY TIME YOU USE test_holdout!**

1.  Load best performing model
2.  Predict on hold-out test set (FIRST TIME SEEING THIS DATA!)
3.  Calculate final test metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
4.  Generate confusion matrix and classification report
5.  Compare test performance with CV estimates
6.  Document final unbiased performance estimate

#### **Section 8: Model Interpretation & Insights**
1.  Perform SHAP analysis on final model
2.  Analyze feature importance and contributions
3.  Create interpretability visualizations
4.  Generate business insights and recommendations

#### **Section 9: Final Model Deployment Preparation**
1.  Save final model to `models/best_model.pkl`
2.  Create model card with performance characteristics
3.  Document deployment specifications
4.  Create comprehensive final report
5.  Save to `outputs/reports/final_model_report.md`

---

## **🎯 Key Workflow Summary:**

### **The Complete Data Flow:**

```
Step 1: Split data into train_full (80%) and test_holdout (20%)
        ↓
Step 2: Do ALL analysis on train_full only
        ↓
Step 3: Clean both sets using train_full statistics
        ↓
Step 4: Engineer features on both sets using train_full transformers
        ↓
Step 5: Use K-Fold CV on train_full for model development
        ↓
Step 6: Train final model on entire train_full
        ↓
Step 7: Evaluate final model on test_holdout (FIRST TIME!)
        ↓
Step 8: Report test_holdout performance as final result
```

### **What Happens in Each Notebook:**

**Notebook 1:**
- Split into train_full and test_holdout
- Analyze train_full only
- Never look at test_holdout

**Notebook 2:**
- Clean train_full (calculate statistics)
- Clean test_holdout (apply train_full statistics)
- Engineer features on both (fit on train_full, transform both)

**Notebook 3:**
- Use K-Fold CV on train_full (model development)
- Train final model on train_full
- Evaluate ONCE on test_holdout (final performance)

---

## **💡 Common Mistakes to Avoid:**

### **❌ WRONG: Using test_holdout in K-Fold**
```python
# DON'T DO THIS!
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_combined):  # X_combined includes test!
    # This causes data leakage!
```

### ** CORRECT: K-Fold on train_full only**
```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for train_idx, val_idx in kf.split(X_train_full, y_train_full):
    # Only splits train_full, test_holdout never involved
```

### **❌ WRONG: Multiple evaluations on test_holdout**
```python
# DON'T DO THIS!
model1.score(X_test_holdout, y_test_holdout)  # First evaluation
model2.score(X_test_holdout, y_test_holdout)  # Second evaluation - LEAKAGE!
# Each time you evaluate, you "learn" something about test set
```

### ** CORRECT: Use CV for comparison, test_holdout once**
```python
# Compare models using K-Fold CV on train_full
cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5)

# Train final model on full train_full
final_model.fit(X_train_full, y_train_full)

# Evaluate on test_holdout ONCE
final_score = final_model.score(X_test_holdout, y_test_holdout)
```

---

## **📝 File Naming Conventions**

### **General Principles:**
1. Use lowercase letters
2. Separate words with underscores (_)
3. Use descriptive names
4. Include version numbers when applicable
5. Add date stamps for reports (YYYY-MM-DD format)
6. Be consistent across project

### **Notebooks:**
- Format: `##_descriptive_name.ipynb`
- Examples:
  - `01_data_splitting_eda.ipynb`
  - `02_data_cleaning_feature_engineering.ipynb`
  - `03_model_training_evaluation.ipynb`

### **Data Files:**
- Raw: `credit_score_raw.csv`, `train_full.csv`, `test_holdout.csv`
- Cleaned: `train_full_cleaned.csv`, `test_holdout_cleaned.csv`
- Processed: `train_full_processed.csv`, `test_holdout_processed.csv`
- Configuration: `imputation_stats.json`, `cleaning_config.json`

### **Model Files:**
- Format: `model_name_version.pkl`
- Examples:
  - `logistic_regression_baseline.pkl`
  - `random_forest_v1.pkl`
  - `xgboost_tuned_v2.pkl`
  - `best_model.pkl`

### **Visualization Files:**
- Format: `descriptive_name_plot_type.png`
- Examples:
  - `target_distribution_barplot.png`
  - `correlation_heatmap.png`
  - `confusion_matrix_xgboost.png`

### **Report Files:**
- Format: `report_type_YYYY_MM_DD.md` or `.pdf`
- Examples:
  - `data_quality_report_2024_01_15.md`
  - `eda_insights_2024_01_16.md`
  - `final_model_report_2024_01_20.pdf`

---

## **📌 How to Name This Document for Your Project**

**Recommended File Names:**

1. **If storing in project root:**
   - `IMPLEMENTATION_GUIDE.md`  ← **Recommended**
   - `ML_PIPELINE_GUIDE.md`
   - `PROJECT_STRUCTURE.md`

2. **If storing in documentation folder:**
   - `docs/IMPLEMENTATION_GUIDE.md`
   - `docs/ml_pipeline_detailed_steps.md`

**My Recommendation:**
```
credit-scoring-ml-project/
├── README.md
├── IMPLEMENTATION_GUIDE.md  ← Save this document here
├── requirements.txt
└── ...
```

---

## ** Summary**

This updated 3-notebook structure provides:

1. **Clear Data Separation:** Hold-out test set for final evaluation only
2. **Proper K-Fold Usage:** Cross-validation on training data only
3. **No Data Leakage:** Strict separation between development and evaluation
4. **Reproducibility:** Numbered prefixes indicate execution order
5. **Best Practices:** Follows industry standards for ML project organization
6. **Professional Workflow:** Ensures unbiased performance estimates

The key insight is that **hold-out splitting and K-Fold cross-validation serve different purposes** and work together in a nested structure:

- **Hold-out test set (20%):** Final unbiased evaluation (use once)
- **K-Fold CV on training set (80%):** Model development and tuning (use many times)

This approach prevents both data leakage and overfitting, giving you reliable performance estimates for your credit scoring ML project! 🎯
