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
│   │   ├── credit_score_truncated_raw.csv    # One-row-per-customer snapshot
│   │   ├── train_full.csv                    # 80% of truncated (for development)
│   │   └── test_holdout.csv                  # 20% of truncated (LOCKED until end)
│   │
│   ├── interim/
│   │   ├── imputation_stats.json             # Calculated from train_full ONLY
│   │   ├── cleaning_config.json              # Calculated from train_full ONLY
│   │   ├── columns_to_drop.json              # Decided from train_full analysis
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

## **📊 Data Splitting Strategy: Hold-Out Test Set vs K-Fold Cross-Validation**

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
3. ✅ Load raw data from `data/raw/credit_score_truncated_raw.csv`
4. ✅ Display first few rows and basic information

---

#### **Section 2: Hold-Out Test Set Split** ⭐ **CRITICAL**
**Purpose:** Create final evaluation set that is NEVER touched during development

```python
# Pseudo-code example:
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('data/raw/credit_score_truncated_raw.csv')

# Separate features and target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Create hold-out test set (20%)
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X, y, 
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Recombine for easier handling
train_full = pd.concat([X_train_full, y_train_full], axis=1)
test_holdout = pd.concat([X_test_holdout, y_test_holdout], axis=1)

# Save splits
train_full.to_csv('data/raw/train_full.csv', index=False)
test_holdout.to_csv('data/raw/test_holdout.csv', index=False)

print(f"Training set: {train_full.shape}")
print(f"Test holdout set: {test_holdout.shape}")
```

**⚠️ IMPORTANT DOCUMENTATION:**
- Add markdown cell: "test_holdout will NOT be used until final evaluation in Notebook 3"
- Add markdown cell: "All analysis, statistics, and decisions will be made using train_full ONLY"

---

#### **Section 3: Initial Data Assessment** (on train_full ONLY)
**🔴 FROM THIS POINT FORWARD, WORK ONLY WITH train_full**

```python
# Load train_full for analysis
train = pd.read_csv('data/raw/train_full.csv')
```

1. ✅ Check dataset shape (rows, columns)
2. ✅ Display data types for all columns
3. ✅ Generate summary statistics using describe()
4. ✅ Identify columns with missing values and their percentages
5. ✅ Check for duplicate rows (especially customer_id + month combinations)
6. ⚠️ Document initial findings in markdown cell

---

#### **Section 4: Missing Value Analysis** (on train_full ONLY)

**4.1 Calculate Missing Value Statistics**
```python
# Calculate missing values from train_full
missing_stats = train.isnull().sum()
missing_percent = (train.isnull().sum() / len(train)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_stats,
    'Missing_Percent': missing_percent
}).sort_values('Missing_Percent', ascending=False)

print(missing_df[missing_df['Missing_Count'] > 0])
```

**4.2 Visualize Missing Patterns**
- Create bar chart showing missing percentage by column
- Visualize missing data patterns using missingno library (optional)
- Check if missingness correlates between columns

**4.3 Determine Missingness Type**
Based on patterns observed:
- **MCAR (Missing Completely at Random):** No pattern, random distribution
- **MAR (Missing at Random):** Missingness related to other observed variables
- **MNAR (Missing Not at Random):** Missingness related to the value itself

**4.4 Document Column-by-Column Strategy** ⭐ **DECISIONS MADE HERE**

Create detailed strategy table:

| Column | Missing % | Action | Reason |
|--------|-----------|--------|--------|
| name | X% | **DROP** | Not predictive, privacy concern |
| ssn | X% | **DROP** | Privacy concern, not useful |
| customer_id | 0% | **KEEP** | Identifier (drop before modeling) |
| month | X% | **KEEP + IMPUTE** | Temporal feature, impute with mode |
| occupation | X% | **IMPUTE with "Unknown"** | Categorical, create new category |
| age | X% | **IMPUTE with median** | Numerical, central tendency |
| annual_income | X% | **IMPUTE with median** | Numerical, important feature |
| monthly_inhand_salary | X% | **IMPUTE with median** | Can cross-validate with annual_income |
| num_bank_accounts | X% | **IMPUTE with median** | Count variable |
| num_credit_card | X% | **IMPUTE with median** | Count variable |
| interest_rate | X% | **IMPUTE with median** | Important financial metric |
| num_of_loan | X% | **IMPUTE with 0** | Missing likely means no loans |
| type_of_loan | X% | **IMPUTE with "No_Loan"** | Missing likely means no loans |
| delay_from_due_date | X% | **IMPUTE with 0** | Missing likely means no delays |
| num_of_delayed_payment | X% | **IMPUTE with 0** | Missing likely means no delays |
| changed_credit_limit | X% | **KEEP + IMPUTE** | Analyze further |
| num_credit_inquiries | X% | **IMPUTE with median** | Important credit behavior |
| credit_mix | X% | **IMPUTE with mode** | Categorical credit diversity |
| outstanding_debt | X% | **IMPUTE with median** | Important financial metric |
| credit_utilization_ratio | X% | **IMPUTE with median** | Key credit metric |
| credit_history_age | X% | **IMPUTE with median** | Important predictor |
| payment_of_min_amount | X% | **IMPUTE with mode** | Categorical behavior |
| total_emi_per_month | X% | **IMPUTE with median** | Financial obligation |
| amount_invested_monthly | X% | **IMPUTE with median** | Financial behavior |
| payment_behaviour | X% | **IMPUTE with mode** | Categorical pattern |
| monthly_balance | X% | **IMPUTE with median** | Financial snapshot |
| Credit_Score | 0% | **TARGET** | No missing values expected |

**4.5 Save Column Drop List**
```python
# Columns to drop (decided from train_full analysis)
columns_to_drop = ['name', 'ssn']

# Save configuration
import json
with open('data/interim/columns_to_drop.json', 'w') as f:
    json.dump(columns_to_drop, f)
```

**4.6 Calculate and Save Imputation Statistics** ⭐ **CRITICAL FOR CONSISTENCY**
```python
# Calculate imputation values from train_full ONLY
imputation_stats = {
    # Numerical features - use median
    'age': train['age'].median(),
    'annual_income': train['annual_income'].median(),
    'monthly_inhand_salary': train['monthly_inhand_salary'].median(),
    'num_bank_accounts': train['num_bank_accounts'].median(),
    'num_credit_card': train['num_credit_card'].median(),
    'interest_rate': train['interest_rate'].median(),
    'num_of_loan': 0,  # Missing = no loans
    'delay_from_due_date': 0,  # Missing = no delays
    'num_of_delayed_payment': 0,  # Missing = no delays
    'num_credit_inquiries': train['num_credit_inquiries'].median(),
    'outstanding_debt': train['outstanding_debt'].median(),
    'credit_utilization_ratio': train['credit_utilization_ratio'].median(),
    'credit_history_age': train['credit_history_age'].median(),
    'total_emi_per_month': train['total_emi_per_month'].median(),
    'amount_invested_monthly': train['amount_invested_monthly'].median(),
    'monthly_balance': train['monthly_balance'].median(),
    
    # Categorical features - use mode or default
    'month': train['month'].mode()[0],
    'occupation': 'Unknown',
    'type_of_loan': 'No_Loan',
    'credit_mix': train['credit_mix'].mode()[0],
    'payment_of_min_amount': train['payment_of_min_amount'].mode()[0],
    'payment_behaviour': train['payment_behaviour'].mode()[0]
}

# Save imputation statistics
with open('data/interim/imputation_stats.json', 'w') as f:
    json.dump(imputation_stats, f, indent=4)

print("✅ Imputation statistics saved (calculated from train_full only)")
```

---

#### **Section 5: Outlier Detection & Treatment** (on train_full ONLY)

**5.1 Identify Outliers Using Boxplots**
```python
# Numerical columns for outlier analysis
numerical_cols = ['age', 'num_bank_accounts', 'num_credit_card', 
                  'interest_rate', 'num_of_loan', 'delay_from_due_date',
                  'num_of_delayed_payment', 'annual_income']

# Create boxplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 4, idx % 4]
    train.boxplot(column=col, ax=ax)
    ax.set_title(f'{col} Distribution')
plt.tight_layout()
plt.savefig('outputs/figures/eda/outlier_boxplots.png')
```

**5.2 Define and Save Acceptable Ranges** ⭐ **FROM train_full**
```python
# Calculate percentiles from train_full for capping
cleaning_config = {
    'age_min': max(18, train['age'].quantile(0.01)),
    'age_max': min(80, train['age'].quantile(0.99)),
    
    'num_bank_accounts_max': train['num_bank_accounts'].quantile(0.99),
    'num_credit_card_max': train['num_credit_card'].quantile(0.99),
    
    'interest_rate_min': 0,
    'interest_rate_max': train['interest_rate'].quantile(0.99),
    
    'num_of_loan_max': train['num_of_loan'].quantile(0.99),
    'delay_from_due_date_max': train['delay_from_due_date'].quantile(0.99),
    'num_of_delayed_payment_max': train['num_of_delayed_payment'].quantile(0.99),
    
    'annual_income_min': train['annual_income'].quantile(0.01),
    'annual_income_max': train['annual_income'].quantile(0.99)
}

# Save cleaning configuration
with open('data/interim/cleaning_config.json', 'w') as f:
    json.dump(cleaning_config, f, indent=4)

print("✅ Cleaning configuration saved (calculated from train_full only)")
```

---

#### **Section 6: Data Consistency Checks** (on train_full ONLY)
1. ✅ Validate annual_income vs monthly_inhand_salary relationship
2. ✅ Check if monthly_inhand_salary * 12 ≈ annual_income (±20% tolerance)
3. ✅ Flag inconsistent records for review
4. ✅ Standardize string columns (trim spaces, consistent case)
5. ✅ Verify month values are valid (January-August)

---

#### **Section 7: Duplicate Handling** (on train_full ONLY)
```python
# Check for duplicates in train_full
duplicates = train.duplicated().sum()
print(f"Exact duplicates: {duplicates}")

# Check for duplicates by customer_id + month
customer_month_dups = train.duplicated(subset=['customer_id', 'month']).sum()
print(f"Customer+Month duplicates: {customer_month_dups}")

# Remove duplicates if any
train_dedup = train.drop_duplicates()
print(f"Rows removed: {len(train) - len(train_dedup)}")
```

---

#### **Section 8: Target Variable Analysis** (on train_full ONLY)
```python
# Target distribution in train_full
target_counts = train['Credit_Score'].value_counts()
target_percent = train['Credit_Score'].value_counts(normalize=True) * 100

print("\nCredit Score Distribution:")
print(target_counts)
print("\nPercentages:")
print(target_percent)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
target_counts.plot(kind='bar', ax=ax1)
ax1.set_title('Credit Score Distribution (Count)')
target_percent.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_title('Credit Score Distribution (%)')
plt.savefig('outputs/figures/eda/target_distribution.png')
```

---

#### **Section 9-10: Exploratory Data Analysis** (on train_full ONLY)
Continue with standard EDA procedures on train_full dataset

---

#### **Section 11: Save Configuration & Insights**
```python
# Summary of what was saved:
print("="*60)
print("CONFIGURATION FILES SAVED (from train_full analysis):")
print("="*60)
print("1. data/interim/columns_to_drop.json")
print("2. data/interim/imputation_stats.json")
print("3. data/interim/cleaning_config.json")
print("\nThese will be applied to BOTH train_full and test_holdout in Notebook 2")
print("="*60)
```

---

## **📓 Notebook 2: Data Cleaning & Feature Engineering**

**File Name:** `02_data_cleaning_feature_engineering.ipynb`

### **Detailed Steps:**

#### **Section 1: Load Data & Configuration Files**
```python
import pandas as pd
import json

# Load both datasets
train_full = pd.read_csv('data/raw/train_full.csv')
test_holdout = pd.read_csv('data/raw/test_holdout.csv')

# Load configuration (calculated from train_full in Notebook 1)
with open('data/interim/columns_to_drop.json', 'r') as f:
    columns_to_drop = json.load(f)

with open('data/interim/imputation_stats.json', 'r') as f:
    imputation_stats = json.load(f)

with open('data/interim/cleaning_config.json', 'r') as f:
    cleaning_config = json.load(f)

print("✅ Configuration loaded from train_full analysis")
```

---

#### **Section 2: Drop Columns** ⭐ **APPLY TO BOTH DATASETS**
```python
# Drop columns from BOTH datasets
print("Dropping columns:", columns_to_drop)

train_full = train_full.drop(columns=columns_to_drop)
test_holdout = test_holdout.drop(columns=columns_to_drop)

print(f"Train shape after drop: {train_full.shape}")
print(f"Test shape after drop: {test_holdout.shape}")
```

---

#### **Section 3: Apply Imputation** ⭐ **APPLY TO BOTH DATASETS**
```python
def apply_imputation(df, imputation_stats):
    """Apply imputation using pre-calculated statistics"""
    df_imputed = df.copy()
    
    for column, value in imputation_stats.items():
        if column in df_imputed.columns:
            df_imputed[column].fillna(value, inplace=True)
    
    return df_imputed

# Apply to train_full
train_full_imputed = apply_imputation(train_full, imputation_stats)

# Apply to test_holdout (using SAME statistics from train_full)
test_holdout_imputed = apply_imputation(test_holdout, imputation_stats)

print("✅ Imputation applied to both datasets using train_full statistics")
```

---

#### **Section 4: Apply Outlier Capping** ⭐ **APPLY TO BOTH DATASETS**
```python
def apply_outlier_capping(df, cleaning_config):
    """Apply outlier capping using pre-calculated ranges"""
    df_capped = df.copy()
    
    # Age capping
    if 'age' in df_capped.columns:
        df_capped['age'] = df_capped['age'].clip(
            cleaning_config['age_min'], 
            cleaning_config['age_max']
        )
    
    # Other numerical features
    if 'num_bank_accounts' in df_capped.columns:
        df_capped['num_bank_accounts'] = df_capped['num_bank_accounts'].clip(
            upper=cleaning_config['num_bank_accounts_max']
        )
    
    # ... repeat for other features
    
    return df_capped

# Apply to train_full
train_full_capped = apply_outlier_capping(train_full_imputed, cleaning_config)

# Apply to test_holdout (using SAME ranges from train_full)
test_holdout_capped = apply_outlier_capping(test_holdout_imputed, cleaning_config)

print("✅ Outlier capping applied to both datasets using train_full ranges")
```

---

#### **Section 5: Save Cleaned Data**
```python
# Save cleaned datasets
train_full_capped.to_csv('data/interim/train_full_cleaned.csv', index=False)
test_holdout_capped.to_csv('data/interim/test_holdout_cleaned.csv', index=False)

print("✅ Cleaned data saved")
print(f"   - data/interim/train_full_cleaned.csv")
print(f"   - data/interim/test_holdout_cleaned.csv")
```

---

#### **Section 6: Feature Engineering on train_full** ⭐ **FIT ON TRAIN_FULL**
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load cleaned train_full
train = pd.read_csv('data/interim/train_full_cleaned.csv')

# Create derived features
train['debt_ratio'] = train['num_of_loan'] / (train['num_bank_accounts'] + 1)
train['income_to_debt'] = train['annual_income'] / (train['outstanding_debt'] + 1)
# ... more feature engineering

# Separate features and target
X_train = train.drop('Credit_Score', axis=1)
y_train = train['Credit_Score']

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Fit scaler on train_full numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Save scaler
joblib.dump(scaler, 'models/feature_scaler.pkl')

# Fit label encoder on categorical features
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, 'models/label_encoders.pkl')

# Recombine with target
train_processed = pd.concat([X_train, y_train], axis=1)
train_processed.to_csv('data/processed/train_full_processed.csv', index=False)

print("✅ Feature engineering completed on train_full")
print("✅ Transformers saved (scaler, encoders)")
```

---

#### **Section 7: Feature Engineering on test_holdout** ⭐ **TRANSFORM USING TRAIN_FULL TRANSFORMERS**
```python
# Load cleaned test_holdout
test = pd.read_csv('data/interim/test_holdout_cleaned.csv')

# Create SAME derived features
test['debt_ratio'] = test['num_of_loan'] / (test['num_bank_accounts'] + 1)
test['income_to_debt'] = test['annual_income'] / (test['outstanding_debt'] + 1)
# ... same features as train

# Separate features and target
X_test = test.drop('Credit_Score', axis=1)
y_test = test['Credit_Score']

# Load fitted transformers
scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Transform numerical features using train_full scaler
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Transform categorical features using train_full encoders
for col in categorical_cols:
    le = encoders[col]
    X_test[col] = le.transform(X_test[col].astype(str))

# Recombine with target
test_processed = pd.concat([X_test, y_test], axis=1)
test_processed.to_csv('data/processed/test_holdout_processed.csv', index=False)

print("✅ Feature engineering completed on test_holdout using train_full transformers")
```

---

## **📓 Notebook 3: Model Training & Evaluation**

**File Name:** `03_model_training_evaluation.ipynb`

### **Detailed Steps:**

#### **Section 1: Load Processed Data**
```python
# Load processed datasets
train_full = pd.read_csv('data/processed/train_full_processed.csv')
test_holdout = pd.read_csv('data/processed/test_holdout_processed.csv')

# Separate features and target
X_train_full = train_full.drop('Credit_Score', axis=1)
y_train_full = train_full['Credit_Score']

X_test_holdout = test_holdout.drop('Credit_Score', axis=1)
y_test_holdout = test_holdout['Credit_Score']

print(f"Train: {X_train_full.shape}")
print(f"Test (hold-out): {X_test_holdout.shape}")
```

---

#### **Section 2: Model Development with K-Fold CV** ⭐ **ON train_full ONLY**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Define K-Fold (on train_full only!)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model
model = LogisticRegression(max_iter=1000)

# K-Fold Cross-Validation on train_full
cv_scores = cross_val_score(model, X_train_full, y_train_full, 
                            cv=kf, scoring='f1_weighted')

print(f"CV F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

#### **Section 3-6: Train Multiple Models** (with K-Fold CV on train_full)
- Repeat Section 2 pattern for:
  - Random Forest
  - XGBoost
  - LightGBM
- Use K-Fold CV for hyperparameter tuning
- Compare CV scores
- Select best model

---

#### **Section 7: Final Evaluation on test_holdout** ⭐ **FIRST AND ONLY USE**
```python
# Train final model on entire train_full
best_model = RandomForestClassifier(...)  # with best hyperparameters
best_model.fit(X_train_full, y_train_full)

# Predict on test_holdout (FIRST TIME!)
y_pred_test = best_model.predict(X_test_holdout)

# Calculate final metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report

test_accuracy = accuracy_score(y_test_holdout, y_pred_test)
test_f1 = f1_score(y_test_holdout, y_pred_test, average='weighted')

print("="*60)
print("FINAL TEST SET PERFORMANCE (Hold-Out Evaluation)")
print("="*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_holdout, y_pred_test))
```

---

## **🎯 Complete Workflow Summary**

```
Step 1: Split into train_full (80%) and test_holdout (20%)
        ↓
Step 2: Analyze train_full → Calculate statistics, decide on drops/imputation
        ↓
Step 3: Apply drops to BOTH datasets
        ↓
Step 4: Apply imputation to BOTH using train_full statistics
        ↓
Step 5: Apply outlier capping to BOTH using train_full ranges
        ↓
Step 6: Engineer features on train_full → Fit transformers
        ↓
Step 7: Apply same features to test_holdout using fitted transformers
        ↓
Step 8: Use K-Fold CV on train_full for model development
        ↓
Step 9: Train final model on entire train_full
        ↓
Step 10: Evaluate ONCE on test_holdout
        ↓
Step 11: Report test_holdout performance as final unbiased estimate
```

---

## **💡 Critical Points to Remember**

### **When to Drop Columns:**
- ✅ **Decision made in Notebook 1** (analyzing train_full only)
- ✅ **Applied in Notebook 2** (to both train_full and test_holdout)
- ❌ **Never drop different columns** from train vs test

### **When to Impute Data:**
- ✅ **Statistics calculated in Notebook 1** (from train_full only)
- ✅ **Imputation applied in Notebook 2** (to both datasets using train_full stats)
- ❌ **Never calculate separate statistics** for test_holdout

### **When to Apply Transformations:**
- ✅ **Fit transformers in Notebook 2** (on train_full only)
- ✅ **Transform both datasets in Notebook 2** (train_full and test_holdout)
- ❌ **Never fit transformers** on test_holdout

---

## **❌ Common Mistakes to Avoid**

### **Mistake 1: Using test_holdout in K-Fold**
```python
# ❌ WRONG: Including test_holdout in K-Fold
X_combined = pd.concat([X_train_full, X_test_holdout])
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_combined):
    # This causes data leakage!
```

```python
# ✅ CORRECT: K-Fold on train_full only
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X_train_full, y_train_full):
    # Only splits train_full, test_holdout never involved
```

---

### **Mistake 2: Calculating Separate Statistics for Test**
```python
# ❌ WRONG: Calculating imputation values from test_holdout
test_age_median = test_holdout['age'].median()
test_holdout['age'].fillna(test_age_median, inplace=True)
# This causes data leakage!
```

```python
# ✅ CORRECT: Use train_full statistics for both
train_age_median = train_full['age'].median()
train_full['age'].fillna(train_age_median, inplace=True)
test_holdout['age'].fillna(train_age_median, inplace=True)  # Same value!
```

---

### **Mistake 3: Multiple Evaluations on test_holdout**
```python
# ❌ WRONG: Evaluating multiple times
model1.score(X_test_holdout, y_test_holdout)  # First evaluation
model2.score(X_test_holdout, y_test_holdout)  # Second evaluation - LEAKAGE!
model3.score(X_test_holdout, y_test_holdout)  # Third evaluation - MORE LEAKAGE!
```

```python
# ✅ CORRECT: Use CV for comparison, test_holdout once
cv_score_1 = cross_val_score(model1, X_train_full, y_train_full, cv=5).mean()
cv_score_2 = cross_val_score(model2, X_train_full, y_train_full, cv=5).mean()

best_model = model1 if cv_score_1 > cv_score_2 else model2
best_model.fit(X_train_full, y_train_full)

# Evaluate on test_holdout ONCE
final_score = best_model.score(X_test_holdout, y_test_holdout)
```

---

### **Mistake 4: Fitting Scaler on Test Data**
```python
# ❌ WRONG: Fitting separate scalers
train_scaler = StandardScaler()
train_full[numerical_cols] = train_scaler.fit_transform(train_full[numerical_cols])

test_scaler = StandardScaler()  # Different scaler!
test_holdout[numerical_cols] = test_scaler.fit_transform(test_holdout[numerical_cols])
# This causes data leakage!
```

```python
# ✅ CORRECT: Fit on train, transform both
scaler = StandardScaler()
train_full[numerical_cols] = scaler.fit_transform(train_full[numerical_cols])
test_holdout[numerical_cols] = scaler.transform(test_holdout[numerical_cols])  # No fit!
```

---

### **Mistake 5: Dropping Different Columns**
```python
# ❌ WRONG: Different drops for train and test
train_full = train_full.drop(['name', 'ssn'], axis=1)
test_holdout = test_holdout.drop(['name'], axis=1)  # Forgot ssn!
# Features don't match!
```

```python
# ✅ CORRECT: Same drops for both
columns_to_drop = ['name', 'ssn']
train_full = train_full.drop(columns_to_drop, axis=1)
test_holdout = test_holdout.drop(columns_to_drop, axis=1)
```

---

## **📋 Checklist: Data Leakage Prevention**

### **Notebook 1: Data Splitting & EDA**
- [ ] Split data into train_full (80%) and test_holdout (20%) FIRST
- [ ] Perform ALL analysis on train_full ONLY
- [ ] Calculate statistics from train_full ONLY
- [ ] Save column drop list from train_full analysis
- [ ] Save imputation statistics from train_full
- [ ] Save cleaning configuration from train_full
- [ ] NEVER look at test_holdout data
- [ ] Document: "test_holdout locked until Notebook 3"

### **Notebook 2: Data Cleaning & Feature Engineering**
- [ ] Load both train_full and test_holdout
- [ ] Load configuration files from Notebook 1
- [ ] Drop SAME columns from both datasets
- [ ] Apply imputation to both using train_full statistics
- [ ] Apply outlier capping to both using train_full ranges
- [ ] Create derived features on train_full
- [ ] Fit transformers (scaler, encoders) on train_full ONLY
- [ ] Create SAME derived features on test_holdout
- [ ] Transform test_holdout using fitted transformers
- [ ] Verify feature consistency between train and test
- [ ] Save processed datasets

### **Notebook 3: Model Training & Evaluation**
- [ ] Load processed train_full and test_holdout
- [ ] Use K-Fold CV on train_full ONLY
- [ ] Compare models using CV scores
- [ ] Perform hyperparameter tuning with K-Fold on train_full
- [ ] Train final model on entire train_full
- [ ] Evaluate on test_holdout ONCE ONLY
- [ ] Report test_holdout performance as final result
- [ ] Document CV vs test performance comparison

---

## **📊 Complete Data Flow with File Names**

```
┌─────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 1: Data Splitting & EDA                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    credit_score_truncated_raw.csv (100%)
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
        train_full.csv (80%)    test_holdout.csv (20%)
                  │                       │
                  │                  [LOCKED AWAY]
                  │                       │
                  ▼                       │
        [Analyze train_full]              │
                  │                       │
                  ├─► columns_to_drop.json
                  ├─► imputation_stats.json
                  └─► cleaning_config.json
                              │
                              │
┌─────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 2: Data Cleaning & Feature Engineering                │
└─────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────────────────┐
                  ▼                                   ▼
        train_full.csv                      test_holdout.csv
                  │                                   │
                  │    Load Configuration Files       │
                  │◄───────────────────────────────┐  │
                  │                                │  │
                  ├─► columns_to_drop.json        │  │
                  ├─► imputation_stats.json       │  │
                  └─► cleaning_config.json        │  │
                  │                                │  │
                  ├─► [Drop Columns] ─────────────┼──┤
                  │                                │  │
                  ├─► [Apply Imputation] ─────────┼──┤
                  │    (using train_full stats)   │  │
                  │                                │  │
                  ├─► [Apply Outlier Capping] ────┼──┤
                  │    (using train_full ranges)  │  │
                  ▼                                ▼  │
    train_full_cleaned.csv          test_holdout_cleaned.csv
                  │                                   │
                  ├─► [Create Features] ─────────────┤
                  │                                   │
                  ├─► [Fit Transformers] ────────────┐
                  │   • StandardScaler               │
                  │   • LabelEncoders                │
                  │                                   │
                  ├─► feature_scaler.pkl             │
                  ├─► label_encoders.pkl             │
                  │                                   │
                  ├─► [Transform Train] ─────────────┤
                  │                                   │
                  │   [Transform Test] ◄──────────────┤
                  │   (using fitted transformers)     │
                  ▼                                   ▼
   train_full_processed.csv        test_holdout_processed.csv
                  │                                   │
                  │                           [STILL LOCKED]
                  │                                   │
┌─────────────────────────────────────────────────────────────────┐
│ NOTEBOOK 3: Model Training & Evaluation                        │
└─────────────────────────────────────────────────────────────────┘
                  │                                   │
                  ▼                                   │
    train_full_processed.csv                          │
                  │                                   │
                  ├─► [K-Fold CV]                     │
                  │   ├─ Fold 1 (64% + 16%)           │
                  │   ├─ Fold 2 (64% + 16%)           │
                  │   ├─ Fold 3 (64% + 16%)           │
                  │   ├─ Fold 4 (64% + 16%)           │
                  │   └─ Fold 5 (64% + 16%)           │
                  │                                   │
                  ├─► Average CV Score                │
                  │                                   │
                  ├─► [Hyperparameter Tuning]         │
                  │   (using K-Fold CV)               │
                  │                                   │
                  ├─► [Train Final Model]             │
                  │   (on entire train_full)          │
                  │                                   │
                  └─────────────────┐                 │
                                    │                 │
                                    ▼                 ▼
                            test_holdout_processed.csv
                                    │
                                    ├─► [Evaluate ONCE]
                                    │
                                    ▼
                        Final Unbiased Performance
                              Estimate
```

---

## **📝 Code Template: Complete Pipeline**

### **Notebook 1: Section 2 - Split and Save Configuration**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

# Load raw data
df = pd.read_csv('data/raw/credit_score_truncated_raw.csv')

# Perform stratified split
train_full, test_holdout = train_test_split(
    df, 
    test_size=0.20,
    stratify=df['Credit_Score'],
    random_state=42
)

# Save splits
train_full.to_csv('data/raw/train_full.csv', index=False)
test_holdout.to_csv('data/raw/test_holdout.csv', index=False)

print(f"✅ Data split complete")
print(f"   Train: {train_full.shape}")
print(f"   Test:  {test_holdout.shape}")

# ============================================================
# ALL SUBSEQUENT ANALYSIS DONE ON train_full ONLY
# ============================================================

# Analyze train_full and create configuration
train = train_full.copy()

# 1. Decide which columns to drop
columns_to_drop = ['name', 'ssn']  # Based on analysis

# 2. Calculate imputation statistics
imputation_stats = {
    'age': float(train['age'].median()),
    'annual_income': float(train['annual_income'].median()),
    'monthly_inhand_salary': float(train['monthly_inhand_salary'].median()),
    'num_bank_accounts': float(train['num_bank_accounts'].median()),
    'num_credit_card': float(train['num_credit_card'].median()),
    'interest_rate': float(train['interest_rate'].median()),
    'num_of_loan': 0,
    'delay_from_due_date': 0,
    'num_of_delayed_payment': 0,
    'num_credit_inquiries': float(train['num_credit_inquiries'].median()),
    'outstanding_debt': float(train['outstanding_debt'].median()),
    'credit_utilization_ratio': float(train['credit_utilization_ratio'].median()),
    'credit_history_age': float(train['credit_history_age'].median()),
    'total_emi_per_month': float(train['total_emi_per_month'].median()),
    'amount_invested_monthly': float(train['amount_invested_monthly'].median()),
    'monthly_balance': float(train['monthly_balance'].median()),
    'month': train['month'].mode()[0],
    'occupation': 'Unknown',
    'type_of_loan': 'No_Loan',
    'credit_mix': train['credit_mix'].mode()[0],
    'payment_of_min_amount': train['payment_of_min_amount'].mode()[0],
    'payment_behaviour': train['payment_behaviour'].mode()[0]
}

# 3. Calculate cleaning configuration
cleaning_config = {
    'age_min': float(max(18, train['age'].quantile(0.01))),
    'age_max': float(min(80, train['age'].quantile(0.99))),
    'num_bank_accounts_max': float(train['num_bank_accounts'].quantile(0.99)),
    'num_credit_card_max': float(train['num_credit_card'].quantile(0.99)),
    'interest_rate_min': 0.0,
    'interest_rate_max': float(train['interest_rate'].quantile(0.99)),
    'num_of_loan_max': float(train['num_of_loan'].quantile(0.99)),
    'delay_from_due_date_max': float(train['delay_from_due_date'].quantile(0.99)),
    'num_of_delayed_payment_max': float(train['num_of_delayed_payment'].quantile(0.99)),
    'annual_income_min': float(train['annual_income'].quantile(0.01)),
    'annual_income_max': float(train['annual_income'].quantile(0.99))
}

# Save all configuration files
with open('data/interim/columns_to_drop.json', 'w') as f:
    json.dump(columns_to_drop, f, indent=4)

with open('data/interim/imputation_stats.json', 'w') as f:
    json.dump(imputation_stats, f, indent=4)

with open('data/interim/cleaning_config.json', 'w') as f:
    json.dump(cleaning_config, f, indent=4)

print("\n✅ Configuration files saved:")
print("   - data/interim/columns_to_drop.json")
print("   - data/interim/imputation_stats.json")
print("   - data/interim/cleaning_config.json")
print("\nThese will be applied to BOTH train_full and test_holdout in Notebook 2")
```

---

### **Notebook 2: Complete Cleaning and Feature Engineering**

```python
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================
# SECTION 1: Load Data and Configuration
# ============================================================

# Load both datasets
train_full = pd.read_csv('data/raw/train_full.csv')
test_holdout = pd.read_csv('data/raw/test_holdout.csv')

# Load configuration (from train_full analysis)
with open('data/interim/columns_to_drop.json', 'r') as f:
    columns_to_drop = json.load(f)

with open('data/interim/imputation_stats.json', 'r') as f:
    imputation_stats = json.load(f)

with open('data/interim/cleaning_config.json', 'r') as f:
    cleaning_config = json.load(f)

print("✅ Data and configuration loaded")
print(f"   Train: {train_full.shape}")
print(f"   Test:  {test_holdout.shape}")

# ============================================================
# SECTION 2: Drop Columns (APPLY TO BOTH)
# ============================================================

print(f"\nDropping columns: {columns_to_drop}")
train_full = train_full.drop(columns=columns_to_drop)
test_holdout = test_holdout.drop(columns=columns_to_drop)

print(f"✅ Columns dropped from both datasets")
print(f"   Train: {train_full.shape}")
print(f"   Test:  {test_holdout.shape}")

# ============================================================
# SECTION 3: Apply Imputation (APPLY TO BOTH)
# ============================================================

def apply_imputation(df, imputation_stats):
    """Apply imputation using pre-calculated statistics from train_full"""
    df_imputed = df.copy()
    
    for column, value in imputation_stats.items():
        if column in df_imputed.columns:
            missing_before = df_imputed[column].isnull().sum()
            df_imputed[column].fillna(value, inplace=True)
            missing_after = df_imputed[column].isnull().sum()
            if missing_before > 0:
                print(f"   {column}: {missing_before} → {missing_after}")
    
    return df_imputed

print("\nApplying imputation to train_full:")
train_full = apply_imputation(train_full, imputation_stats)

print("\nApplying imputation to test_holdout (using train_full statistics):")
test_holdout = apply_imputation(test_holdout, imputation_stats)

print("✅ Imputation completed")

# ============================================================
# SECTION 4: Apply Outlier Capping (APPLY TO BOTH)
# ============================================================

def apply_outlier_capping(df, cleaning_config):
    """Apply outlier capping using pre-calculated ranges from train_full"""
    df_capped = df.copy()
    
    # Age
    if 'age' in df_capped.columns:
        df_capped['age'] = df_capped['age'].clip(
            cleaning_config['age_min'], 
            cleaning_config['age_max']
        )
    
    # Number of bank accounts
    if 'num_bank_accounts' in df_capped.columns:
        df_capped['num_bank_accounts'] = df_capped['num_bank_accounts'].clip(
            upper=cleaning_config['num_bank_accounts_max']
        )
    
    # Number of credit cards
    if 'num_credit_card' in df_capped.columns:
        df_capped['num_credit_card'] = df_capped['num_credit_card'].clip(
            upper=cleaning_config['num_credit_card_max']
        )
    
    # Interest rate
    if 'interest_rate' in df_capped.columns:
        df_capped['interest_rate'] = df_capped['interest_rate'].clip(
            cleaning_config['interest_rate_min'],
            cleaning_config['interest_rate_max']
        )
    
    # Number of loans
    if 'num_of_loan' in df_capped.columns:
        df_capped['num_of_loan'] = df_capped['num_of_loan'].clip(
            upper=cleaning_config['num_of_loan_max']
        )
    
    # Delay from due date
    if 'delay_from_due_date' in df_capped.columns:
        df_capped['delay_from_due_date'] = df_capped['delay_from_due_date'].clip(
            upper=cleaning_config['delay_from_due_date_max']
        )
    
    # Number of delayed payments
    if 'num_of_delayed_payment' in df_capped.columns:
        df_capped['num_of_delayed_payment'] = df_capped['num_of_delayed_payment'].clip(
            upper=cleaning_config['num_of_delayed_payment_max']
        )
    
    # Annual income
    if 'annual_income' in df_capped.columns:
        df_capped['annual_income'] = df_capped['annual_income'].clip(
            cleaning_config['annual_income_min'],
            cleaning_config['annual_income_max']
        )
    
    return df_capped

print("\nApplying outlier capping:")
train_full = apply_outlier_capping(train_full, cleaning_config)
test_holdout = apply_outlier_capping(test_holdout, cleaning_config)

print("✅ Outlier capping completed")

# ============================================================
# SECTION 5: Save Cleaned Data
# ============================================================

train_full.to_csv('data/interim/train_full_cleaned.csv', index=False)
test_holdout.to_csv('data/interim/test_holdout_cleaned.csv', index=False)

print("\n✅ Cleaned data saved:")
print("   - data/interim/train_full_cleaned.csv")
print("   - data/interim/test_holdout_cleaned.csv")

# ============================================================
# SECTION 6: Feature Engineering on train_full (FIT)
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING - TRAIN_FULL (FIT)")
print("="*60)

# Load cleaned train_full
train = pd.read_csv('data/interim/train_full_cleaned.csv')

# Separate features and target
X_train = train.drop('Credit_Score', axis=1)
y_train = train['Credit_Score']

# Drop customer_id if exists (identifier, not feature)
if 'customer_id' in X_train.columns:
    X_train = X_train.drop('customer_id', axis=1)

# Create derived features
print("\nCreating derived features on train_full...")
X_train['debt_ratio'] = X_train['num_of_loan'] / (X_train['num_bank_accounts'] + 1)
X_train['income_to_debt'] = X_train['annual_income'] / (X_train['outstanding_debt'] + 1)
X_train['credit_per_account'] = X_train['num_credit_card'] / (X_train['num_bank_accounts'] + 1)
X_train['monthly_debt_burden'] = X_train['total_emi_per_month'] / (X_train['monthly_inhand_salary'] + 1)
X_train['savings_rate'] = X_train['amount_invested_monthly'] / (X_train['monthly_inhand_salary'] + 1)

# Identify column types
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"   Numerical features: {len(numerical_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Fit scaler on train_full numerical features
print("\nFitting StandardScaler on train_full...")
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
joblib.dump(scaler, 'models/feature_scaler.pkl')
print("✅ Scaler fitted and saved")

# Fit encoders on train_full categorical features
print("\nFitting LabelEncoders on train_full...")
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    encoders[col] = le
    print(f"   {col}: {len(le.classes_)} classes")

joblib.dump(encoders, 'models/label_encoders.pkl')
print("✅ Encoders fitted and saved")

# Recombine and save
train_processed = pd.concat([X_train, y_train], axis=1)
train_processed.to_csv('data/processed/train_full_processed.csv', index=False)
print("\n✅ Processed train_full saved")

# ============================================================
# SECTION 7: Feature Engineering on test_holdout (TRANSFORM)
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING - TEST_HOLDOUT (TRANSFORM)")
print("="*60)

# Load cleaned test_holdout
test = pd.read_csv('data/interim/test_holdout_cleaned.csv')

# Separate features and target
X_test = test.drop('Credit_Score', axis=1)
y_test = test['Credit_Score']

# Drop customer_id if exists
if 'customer_id' in X_test.columns:
    X_test = X_test.drop('customer_id', axis=1)

# Create SAME derived features
print("\nCreating SAME derived features on test_holdout...")
X_test['debt_ratio'] = X_test['num_of_loan'] / (X_test['num_bank_accounts'] + 1)
X_test['income_to_debt'] = X_test['annual_income'] / (X_test['outstanding_debt'] + 1)
X_test['credit_per_account'] = X_test['num_credit_card'] / (X_test['num_bank_accounts'] + 1)
X_test['monthly_debt_burden'] = X_test['total_emi_per_month'] / (X_test['monthly_inhand_salary'] + 1)
X_test['savings_rate'] = X_test['amount_invested_monthly'] / (X_test['monthly_inhand_salary'] + 1)

# Load fitted transformers
print("\nLoading fitted transformers...")
scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Transform numerical features
print("\nTransforming numerical features (using train_full scaler)...")
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("✅ Numerical features transformed")

# Transform categorical features
print("\nTransforming categorical features (using train_full encoders)...")
for col in categorical_cols:
    le = encoders[col]
    # Handle unseen categories
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    print(f"   {col}: transformed")

print("✅ Categorical features transformed")

# Recombine and save
test_processed = pd.concat([X_test, y_test], axis=1)
test_processed.to_csv('data/processed/test_holdout_processed.csv', index=False)
print("\n✅ Processed test_holdout saved")

# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "="*60)
print("FINAL VERIFICATION")
print("="*60)

# Verify shapes match
train_proc = pd.read_csv('data/processed/train_full_processed.csv')
test_proc = pd.read_csv('data/processed/test_holdout_processed.csv')

print(f"Train processed: {train_proc.shape}")
print(f"Test processed:  {test_proc.shape}")

# Verify same columns (except target)
train_cols = set(train_proc.columns) - {'Credit_Score'}
test_cols = set(test_proc.columns) - {'Credit_Score'}

if train_cols == test_cols:
    print("✅ Feature columns match perfectly!")
else:
    print("❌ WARNING: Feature mismatch detected!")
    print(f"   Only in train: {train_cols - test_cols}")
    print(f"   Only in test: {test_cols - train_cols}")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)
print("Ready for Model Training in Notebook 3!")
```

---

### **Notebook 3: Model Training with K-Fold CV**

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# SECTION 1: Load Processed Data
# ============================================================

print("="*60)
print("LOADING PROCESSED DATA")
print("="*60)

# Load processed datasets
train_full = pd.read_csv('data/processed/train_full_processed.csv')
test_holdout = pd.read_csv('data/processed/test_holdout_processed.csv')

# Separate features and target
X_train_full = train_full.drop('Credit_Score', axis=1)
y_train_full = train_full['Credit_Score']

X_test_holdout = test_holdout.drop('Credit_Score', axis=1)
y_test_holdout = test_holdout['Credit_Score']

print(f"✅ Data loaded")
print(f"   Train features: {X_train_full.shape}")
print(f"   Train target: {y_train_full.shape}")
print(f"   Test features: {X_test_holdout.shape}")
print(f"   Test target: {y_test_holdout.shape}")

print(f"\nTarget distribution in train_full:")
print(y_train_full.value_counts())

# ============================================================
# SECTION 2: Baseline Model with K-Fold CV
# ============================================================

print("\n" + "="*60)
print("BASELINE MODEL - LOGISTIC REGRESSION")
print("="*60)

# Define K-Fold strategy (on train_full only!)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model
baseline_model = LogisticRegression(max_iter=1000, random_state=42)

# K-Fold Cross-Validation on train_full
print("\nPerforming 5-Fold Cross-Validation on train_full...")
cv_scores = cross_val_score(baseline_model, X_train_full, y_train_full, 
                             cv=kf, scoring='f1_weighted', n_jobs=-1)

print(f"\nCV F1-Scores per fold: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full train_full for saving
baseline_model.fit(X_train_full, y_train_full)
joblib.dump(baseline_model, 'models/baseline_logistic_regression.pkl')
print("✅ Baseline model saved")

# ============================================================
# SECTION 3: Random Forest with K-Fold CV
# ============================================================

print("\n" + "="*60)
print("RANDOM FOREST CLASSIFIER")
print("="*60)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# K-Fold Cross-Validation
print("\nPerforming 5-Fold Cross-Validation on train_full...")
cv_scores_rf = cross_val_score(rf_model, X_train_full, y_train_full, 
                                cv=kf, scoring='f1_weighted', n_jobs=-1)

print(f"\nCV F1-Scores per fold: {cv_scores_rf}")
print(f"Mean CV F1-Score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Train on full train_full
rf_model.fit(X_train_full, y_train_full)
joblib.dump(rf_model, 'models/random_forest_v1.pkl')
print("✅ Random Forest model saved")

# ============================================================
# SECTION 4: Hyperparameter Tuning with GridSearchCV
# ============================================================

print("\n" + "="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print(f"\nParameter grid: {param_grid}")

# GridSearchCV with K-Fold on train_full
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=kf,  # 5-fold CV on train_full
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

print("\nPerforming GridSearchCV with 5-Fold CV on train_full...")
grid_search.fit(X_train_full, y_train_full)

print(f"\n✅ GridSearchCV complete")
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")

# Best model is automatically trained on full train_full
best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, 'models/random_forest_tuned.pkl')
print("✅ Tuned Random Forest model saved")

# ============================================================
# SECTION 5: Model Comparison (using CV scores)
# ============================================================

print("\n" + "="*60)
print("MODEL COMPARISON (Cross-Validation Scores)")
print("="*60)

models_cv_scores = {
    'Logistic Regression': cv_scores.mean(),
    'Random Forest': cv_scores_rf.mean(),
    'Random Forest (Tuned)': grid_search.best_score_
}

print("\nModel Performance on train_full (5-Fold CV):")
for model_name, score in models_cv_scores.items():
    print(f"   {model_name}: {score:.4f}")

# Select best model based on CV
best_model_name = max(models_cv_scores, key=models_cv_scores.get)
print(f"\n✅ Best model (based on CV): {best_model_name}")

# ============================================================
# SECTION 6: FINAL EVALUATION ON HOLD-OUT TEST SET
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION ON HOLD-OUT TEST SET")
print("="*60)
print("⚠️  THIS IS THE FIRST AND ONLY TIME WE USE test_holdout!")
print("="*60)

# Load best model (or use the one already in memory)
final_model = best_rf_model  # or load from disk

# Predict on hold-out test set (FIRST TIME!)
print("\nPredicting on test_holdout...")
y_pred_test = final_model.predict(X_test_holdout)

# Calculate final test metrics
test_accuracy = accuracy_score(y_test_holdout, y_pred_test)
test_f1 = f1_score(y_test_holdout, y_pred_test, average='weighted')

print("\n" + "="*60)
print("FINAL TEST SET PERFORMANCE")
print("="*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test_holdout, y_pred_test))

# Confusion Matrix
cm = confusion_matrix(y_test_holdout, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Final Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('outputs/figures/model_evaluation/confusion_matrix_final.png')
plt.close()
print("✅ Confusion matrix saved")

# ============================================================
# SECTION 7: Compare CV vs Test Performance
# ============================================================

print("\n" + "="*60)
print("CROSS-VALIDATION vs TEST SET PERFORMANCE")
print("="*60)

print(f"\nBest Model: {best_model_name}")
print(f"   CV F1-Score (train_full):  {models_cv_scores[best_model_name]:.4f}")
print(f"   Test F1-Score (holdout):   {test_f1:.4f}")
print(f"   Difference:                {abs(models_cv_scores[best_model_name] - test_f1):.4f}")

if abs(models_cv_scores[best_model_name] - test_f1) < 0.05:
    print("\n✅ Good generalization! CV and test scores are similar.")
elif test_f1 < models_cv_scores[best_model_name]:
    print("\n⚠️  Model may be slightly overfitting. Test score lower than CV.")
else:
    print("\n✅ Model generalizes well! Test score better than CV.")

# ============================================================
# SECTION 8: Save Final Model and Report
# ============================================================

# Save best model as final
joblib.dump(final_model, 'models/best_model.pkl')
print("\n✅ Final model saved as: models/best_model.pkl")

# Create final report
report = f"""
# Credit Scoring Model - Final Report

## Model Information
- **Best Model**: {best_model_name}
- **Training Data**: {X_train_full.shape[0]} samples
- **Test Data**: {X_test_holdout.shape[0]} samples
- **Number of Features**: {X_train_full.shape[1]}

## Performance Metrics

### Cross-Validation (5-Fold on train_full)
- **Mean F1-Score**: {models_cv_scores[best_model_name]:.4f}

### Hold-Out Test Set (Final Evaluation)
- **Accuracy**: {test_accuracy:.4f}
- **F1-Score**: {test_f1:.4f}

## Classification Report
```
{classification_report(y_test_holdout, y_pred_test)}
```

## Model Generalization
- CV vs Test Difference: {abs(models_cv_scores[best_model_name] - test_f1):.4f}
- Generalization Assessment: {'Good' if abs(models_cv_scores[best_model_name] - test_f1) < 0.05 else 'Fair'}

## Notes
- All preprocessing steps used statistics calculated from train_full only
- No data leakage detected
- Test set was completely held out until final evaluation
- K-Fold CV was performed only on train_full for model development
"""

with open('outputs/reports/final_model_report.md', 'w') as f:
    f.write(report)

print("✅ Final report saved: outputs/reports/final_model_report.md")

print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*60)
```

---

## **🎯 Summary: The Complete Workflow**

### **The Golden Rules**

1. **Split First**: Create train_full (80%) and test_holdout (20%) immediately
2. **Analyze Train Only**: All EDA, statistics, and decisions from train_full
3. **Save Configuration**: Store all statistics and decisions as JSON files
4. **Apply to Both**: Use train_full statistics for both datasets
5. **Fit on Train**: All transformers fitted on train_full only
6. **Transform Both**: Apply fitted transformers to both datasets
7. **K-Fold on Train**: Use K-Fold CV on train_full for model development
8. **Test Once**: Evaluate on test_holdout exactly once at the end

---

## **📊 Key Decision Points**

### **When to Drop Columns (Notebook 1)**
- Analyze train_full for relevance and data quality
- Save list to `columns_to_drop.json`
- Apply to both datasets in Notebook 2

### **When to Impute (Notebook 1 + 2)**
- Calculate statistics from train_full only (Notebook 1)
- Save to `imputation_stats.json`
- Apply to both datasets in Notebook 2

### **When to Cap Outliers (Notebook 1 + 2)**
- Calculate percentiles from train_full only (Notebook 1)
- Save to `cleaning_config.json`
- Apply to both datasets in Notebook 2

### **When to Scale/Encode (Notebook 2)**
- Fit transformers on train_full
- Save fitted objects (`scaler.pkl`, `encoders.pkl`)
- Transform both datasets using fitted objects

### **When to Use K-Fold (Notebook 3)**
- Use only on train_full for model development
- Never include test_holdout in K-Fold
- Use K-Fold for hyperparameter tuning

### **When to Use test_holdout (Notebook 3)**
- Load only at the very end
- Predict and evaluate exactly once
- Report as final unbiased estimate

---

## **✅ Final Checklist**

### **Before Starting:**
- [ ] Understand train/test split vs K-Fold CV
- [ ] Know what data leakage is and how to avoid it
- [ ] Have clear project structure set up

### **Notebook 1 Complete When:**
- [ ] Data split into train_full and test_holdout
- [ ] All analysis done on train_full only
- [ ] columns_to_drop.json saved
- [ ] imputation_stats.json saved
- [ ] cleaning_config.json saved
- [ ] EDA visualizations created
- [ ] EDA insights documented

### **Notebook 2 Complete When:**
- [ ] Columns dropped from both datasets
- [ ] Imputation applied to both using train_full stats
- [ ] Outliers capped in both using train_full ranges
- [ ] Features engineered on train_full (fit)
- [ ] Same features applied to test_holdout (transform)
- [ ] Transformers saved (scaler, encoders)
- [ ] Processed datasets saved
- [ ] No data leakage confirmed

### **Notebook 3 Complete When:**
- [ ] K-Fold CV performed on train_full only
- [ ] Models compared using CV scores
- [ ] Best model selected based on CV
- [ ] Final model trained on entire train_full
- [ ] test_holdout evaluated exactly once
- [ ] Final report generated
- [ ] Best model saved

---

## **🎓 Learning Outcomes**

By following this guide, you will:

1. ✅ Understand the difference between hold-out splitting and K-Fold CV
2. ✅ Prevent data leakage in all preprocessing steps
3. ✅ Apply proper train-test separation
4. ✅ Use K-Fold CV correctly for model development
5. ✅ Generate unbiased performance estimates
6. ✅ Follow ML best practices and industry standards
7. ✅ Create reproducible and production-ready ML pipelines

---

## **📚 Additional Resources**

### **Recommended Reading:**
- Hands-On Machine Learning with Scikit-Learn (Géron)
- The Elements of Statistical Learning (Hastie et al.)
- Feature Engineering for Machine Learning (Zheng & Casari)

### **Key Concepts to Study:**
- Data leakage and how to prevent it
- Cross-validation strategies
- Bias-variance tradeoff
- Feature engineering best practices
- Model evaluation metrics

---

## **📝 Document Version**

**Version**: 2.0  
**Last Updated**: 2025-01-11  
**Changes from v1.0**:
- Clarified when to drop columns vs when to impute
- Added complete code templates for all notebooks
- Emphasized train_full statistics for both datasets
- Added detailed data flow diagrams
- Included comprehensive checklists

---

**END OF IMPLEMENTATION GUIDE**