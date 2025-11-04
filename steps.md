1. Data Loading & Cleaning
2. EDA (Exploratory Data Analysis)
3. Feature Engineering
4. Baseline (Logistic Regression)
5. Tree-based Models (RF, XGB, LGBM)
6. Evaluation & Comparison

### **1. Data Loading & Cleaning Plan**

**Objective:** Remove inconsistencies, missing values, and outliers to make the dataset reliable.

**Steps:**

1. **Missing Values**

   * `name`, `ssn`, `occupation`, `age`, `annual_income`, `monthly_inhand_salary`, `num_of_loan`, `num_of_delayed_payment` have `<NA>` or `nan`.
   * Strategy:

     * For categorical (`name`, `occupation`): impute with `"Unknown"` or mode.
     * For numerical (`age`, `annual_income`): impute with median or mean if distribution is normal; else use median.
     * For `num_of_loan`, `num_of_delayed_payment`: impute 0 if missing implies no loans/payments.

2. **Outliers**

   * `age` has unrealistic values: 95, 99, 100.
   * `num_bank_accounts`, `num_credit_card`, `interest_rate`, `num_of_loan`, `delay_from_due_date`, `num_of_delayed_payment` have extreme values (negative or very high numbers).
   * Strategy:

     * Define acceptable ranges or use percentile capping (e.g., 1st–99th percentile).
     * Replace negative values with 0 where applicable.

3. **Duplicates**

   * Check for duplicates in `customer_id` + `month` combination.
   * Remove duplicates or aggregate.

4. **Data Types**

   * Ensure proper types:

     * Categorical: `name`, `occupation`, `month`, `ssn`.
     * Numerical: `age`, `annual_income`, `monthly_inhand_salary`, `num_bank_accounts`, `num_credit_card`, `interest_rate`, `num_of_loan`, `delay_from_due_date`, `num_of_delayed_payment`.

5. **Consistency**

   * Standardize strings: trim spaces, consistent capitalization.
   * Check for consistency between `annual_income` and `monthly_inhand_salary`.



### **2. EDA (Exploratory Data Analysis)**

Goal: understand data distributions, spot issues, detect relationships with target (`Credit_Score`).

**Include:**

1. **Basic stats**

   * `df.info()`, `df.describe()`, missing values, data types.
2. **Target distribution**

   * Count of `Good / Poor / Standard`.
   * Bar plot or pie chart.
3. **Numerical features**

   * Histograms, boxplots.
   * Check for outliers, skewness, zeros.
   * Correlation with target (e.g., `Age`, `Annual_Income`).
4. **Categorical features**

   * Value counts, bar plots.
   * Check rare categories.
   * Relation with target (`cross-tab` or stacked bar plots).
5. **Relationships**

   * Scatter plots for numerical pairs.
   * Heatmap for correlations.
   * Pairplots for small subsets.
6. **Missing / inconsistent data**

   * Nulls, duplicates, impossible values (negative salary, etc.)
7. **Feature-target insights**

   * Which features seem predictive.
   * Flags for feature engineering (e.g., high skew features may need transformation).

### **3. Feature Engineering Plan**

**Objective:** Transform raw data into meaningful features for modeling.

**Steps:**

1. **Derived Features**

   * `avg_loan_amount = annual_income / num_of_loan` (handle zeros).
   * `debt_ratio = num_of_loan / num_bank_accounts`.
   * `payment_delay_ratio = num_of_delayed_payment / num_of_loan` (if loans > 0).
   * `monthly_to_annual_ratio = monthly_inhand_salary * 12 / annual_income`.

2. **Binning**

   * `age_group` = categorize age (e.g., `<25`, `25–35`, `36–50`, `50+`).
   * `income_bracket` = low, medium, high based on percentiles.

3. **Categorical Encoding**

   * `occupation`, `month` → One-Hot Encoding.
   * `ssn` → drop or hash if privacy required.

4. **Interaction Features**

   * Combine features that may interact: e.g., `occupation * income_bracket`.
   * `num_credit_card / num_bank_accounts` → financial exposure metric.

5. **Normalization/Scaling**

   * Scale numerical features (`annual_income`, `monthly_inhand_salary`, `num_of_loan`, `num_of_delayed_payment`) using Min-Max or StandardScaler.

6. **Temporal Features**

   * Convert `month` into `month_number` (January=1 … August=8).
   * Calculate tenure if relevant (not in current dataset, but can be added if historical info exists).
