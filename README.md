## Problem Statement
You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

## Task
Given a person’s credit-related information, build a machine learning model that can classify the credit score.

## Dataset
The project uses the following Kaggle dataset: [Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)

## Data files (local workspace)
- data/raw/credit_score_raw.csv
  - Full original dataset. Contains multiple rows per customer across months (time-series rows).
- data/raw/credit_score_truncated_raw.csv
  - One-row-per-customer snapshot (latest observation per customer). Created by keeping each customer's most recent month to simplify modeling for prototyping (see notebooks/00_truncate_data.ipynb).
- data/raw/train_full.csv
  - 80% split of the snapshot dataset for development experiments.
- data/raw/test_holdout.csv
  - 20% locked holdout from the snapshot dataset. Kept untouched until final evaluation.

Notes on creating a latest-observation snapshot
- Motivation: The original dataset contains multiple observations per customer over time. Predicting next-month credit_score properly requires time-series / sequence-aware methods and temporal validation. For fast prototyping, we create a cross-sectional snapshot by keeping only the most recent month per customer.
- Consequences: This removes temporal dynamics and prevents sequence-based modeling. Appropriate for initial model building and benchmarking, but for production or true next-month forecasting retain the time dimension and use time-aware pipelines and validation.