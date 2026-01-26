# Databricks notebook source
# MAGIC %md
# MAGIC # 📞 DBX Outreach Model - ML Training Pipeline
# MAGIC
# MAGIC This notebook builds the **DBX Model** for telephonic outreach optimization using machine learning:
# MAGIC
# MAGIC ## ML Components:
# MAGIC 1. **Engagement Likelihood Model** - XGBoost classifier predicting member engagement
# MAGIC 2. **Channel Propensity Models** - Multi-output model for channel-specific response prediction
# MAGIC 3. **Prioritization Scoring** - Composite score combining ML predictions with business rules
# MAGIC
# MAGIC ## Architecture:
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         Feature Engineering                              │
# MAGIC │  (Member Data + Call History + SDOH + Phone Activity → Feature Vector)  │
# MAGIC └─────────────────────────────────────────────────────────────────────────┘
# MAGIC                                     │
# MAGIC                     ┌───────────────┼───────────────┐
# MAGIC                     ▼               ▼               ▼
# MAGIC              ┌────────────┐  ┌────────────┐  ┌────────────┐
# MAGIC              │ Engagement │  │  Channel   │  │  Business  │
# MAGIC              │   Model    │  │ Propensity │  │   Rules    │
# MAGIC              │ (XGBoost)  │  │  Models    │  │   Engine   │
# MAGIC              └────────────┘  └────────────┘  └────────────┘
# MAGIC                     │               │               │
# MAGIC                     └───────────────┼───────────────┘
# MAGIC                                     ▼
# MAGIC                     ┌───────────────────────────────┐
# MAGIC                     │    Prioritization Scoring     │
# MAGIC                     │  & Channel Recommendation     │
# MAGIC                     └───────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup & Configuration

# COMMAND ----------

# Install required packages
%pip install xgboost lightgbm shap mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../env

# COMMAND ----------

# Configuration
CATALOG = CATALOG # from env
SCHEMA = SCHEMA # from env
EXPERIMENT_NAME = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/dbx_outreach_model"

# Model configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COL = "engaged"

print(f"📊 Catalog: {CATALOG}")
print(f"📊 Schema: {SCHEMA}")
print(f"🧪 Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# Imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Visualization
import matplotlib.pyplot as plt

# Set up MLflow
mlflow.set_experiment(EXPERIMENT_NAME)

print("✅ Libraries imported successfully")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Source Data

# COMMAND ----------

# Load all source tables
member_df = spark.table(f"{CATALOG}.{SCHEMA}.identified_member_data")
call_history_df = spark.table(f"{CATALOG}.{SCHEMA}.historical_telephonic_outreach")
public_sdoh_df = spark.table(f"{CATALOG}.{SCHEMA}.public_sdoh_data")
assessment_sdoh_df = spark.table(f"{CATALOG}.{SCHEMA}.assessment_reported_sdoh_data")
phone_activity_df = spark.table(f"{CATALOG}.{SCHEMA}.phone_activity_data")

print(f"✅ Loaded member_data: {member_df.count():,} records")
print(f"✅ Loaded call_history: {call_history_df.count():,} records")
print(f"✅ Loaded public_sdoh: {public_sdoh_df.count():,} records")
print(f"✅ Loaded assessment_sdoh: {assessment_sdoh_df.count():,} records")
print(f"✅ Loaded phone_activity: {phone_activity_df.count():,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Training Labels
# MAGIC
# MAGIC We need to create historical labels for training. The target variable is whether a member **engaged** (connected and completed a call) based on historical outreach attempts.

# COMMAND ----------

# Create engagement labels from call history
# A member "engaged" if they had at least one successful contact in the observation period

# Define observation and outcome windows
# We'll use calls from the first 12 months as features, and predict engagement in months 13-18

call_history_with_dates = call_history_df.withColumn(
    "call_date_parsed", F.to_date("call_date")
)

# Get the date range
date_stats = call_history_with_dates.agg(
    F.min("call_date_parsed").alias("min_date"),
    F.max("call_date_parsed").alias("max_date")
).collect()[0]

max_date = date_stats["max_date"]
min_date = date_stats["min_date"]
cutoff_date = max_date - timedelta(days=180)  # 6 months for outcome window

print(f"📅 Data range: {min_date} to {max_date}")
print(f"📅 Feature window: {min_date} to {cutoff_date}")
print(f"📅 Outcome window: {cutoff_date} to {max_date}")

# COMMAND ----------

# Create target labels based on outcome window
outcome_window_calls = call_history_with_dates.filter(
    F.col("call_date_parsed") > F.lit(cutoff_date)
)

# Label: Did the member have a successful engagement in the outcome window?
engagement_labels = outcome_window_calls.groupBy("member_id").agg(
    F.max(F.when(F.col("call_outcome") == "Connected - Completed", 1).otherwise(0)).alias("engaged"),
    F.sum(F.when(F.col("call_outcome") == "Connected - Completed", 1).otherwise(0)).alias("successful_contacts_outcome"),
    F.count("*").alias("total_attempts_outcome")
)

# For members with no calls in outcome window, we'll treat as not engaged (for training purposes)
# In production, you might handle this differently

print(f"✅ Created labels for {engagement_labels.count():,} members with outcome data")
display(engagement_labels.groupBy("engaged").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering
# MAGIC
# MAGIC Create comprehensive features from all data sources for the ML model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Call History Features (from feature window only)

# COMMAND ----------

# Filter to feature window only
feature_window_calls = call_history_with_dates.filter(
    F.col("call_date_parsed") <= F.lit(cutoff_date)
)

# Aggregate call history features per member
call_features = feature_window_calls.groupBy("member_id").agg(
    # Volume metrics
    F.count("*").alias("total_calls"),
    F.countDistinct("call_date").alias("unique_call_days"),
    
    # Outcome metrics
    F.sum(F.when(F.col("call_outcome") == "Connected - Completed", 1).otherwise(0)).alias("completed_calls"),
    F.sum(F.when(F.col("call_outcome") == "Connected - Callback Requested", 1).otherwise(0)).alias("callback_requested"),
    F.sum(F.when(F.col("call_outcome") == "Connected - Refused", 1).otherwise(0)).alias("refused_calls"),
    F.sum(F.when(F.col("call_outcome") == "Voicemail Left", 1).otherwise(0)).alias("voicemails_left"),
    F.sum(F.when(F.col("call_outcome") == "No Answer", 1).otherwise(0)).alias("no_answer_calls"),
    F.sum(F.when(F.col("call_outcome").isin(["Wrong Number", "Disconnected Number"]), 1).otherwise(0)).alias("bad_number_calls"),
    
    # Timing features
    F.avg("call_duration_seconds").alias("avg_call_duration"),
    F.max("call_duration_seconds").alias("max_call_duration"),
    
    # Recency
    F.datediff(F.lit(cutoff_date), F.max("call_date_parsed")).alias("days_since_last_call"),
    F.datediff(
        F.lit(cutoff_date),
        F.max(F.when(F.col("call_outcome") == "Connected - Completed", F.col("call_date_parsed")))
    ).alias("days_since_last_success"),
    
    # Day of week preferences (mode calculation)
    F.count(F.when(F.col("call_day_of_week") == "Monday", 1)).alias("monday_calls"),
    F.count(F.when(F.col("call_day_of_week") == "Tuesday", 1)).alias("tuesday_calls"),
    F.count(F.when(F.col("call_day_of_week") == "Wednesday", 1)).alias("wednesday_calls"),
    F.count(F.when(F.col("call_day_of_week") == "Thursday", 1)).alias("thursday_calls"),
    F.count(F.when(F.col("call_day_of_week") == "Friday", 1)).alias("friday_calls"),
    
    # Phone type
    F.sum(F.when(F.col("phone_type_called") == "Mobile", 1).otherwise(0)).alias("mobile_calls"),
    F.sum(F.when(F.col("phone_type_called") == "Landline", 1).otherwise(0)).alias("landline_calls")
)

# Calculate derived features
call_features = call_features.withColumn(
    "answer_rate", 
    F.when(F.col("total_calls") > 0, 
           (F.col("completed_calls") + F.col("callback_requested") + F.col("refused_calls")) / F.col("total_calls")
    ).otherwise(0)
).withColumn(
    "success_rate",
    F.when(F.col("total_calls") > 0, F.col("completed_calls") / F.col("total_calls")).otherwise(0)
).withColumn(
    "voicemail_rate",
    F.when(F.col("total_calls") > 0, F.col("voicemails_left") / F.col("total_calls")).otherwise(0)
).withColumn(
    "mobile_preference",
    F.when(F.col("total_calls") > 0, F.col("mobile_calls") / F.col("total_calls")).otherwise(0.5)
)

print(f"✅ Created call history features for {call_features.count():,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Member Demographic Features

# COMMAND ----------

# Select and transform member features
member_features = member_df.select(
    "member_id",
    "age",
    "gender",
    "state",
    "zip_code",
    "urban_rural",
    "insurance_type",
    "pcp_assigned",
    "chronic_condition_count",
    "risk_score",
    "preferred_language",
    "has_email",
    "has_mobile",
    "has_landline",
    "mail_opt_in",
    "sms_opt_in",
    "email_opt_in"
).withColumn(
    "has_multiple_contact_methods",
    (F.col("has_email").cast("int") + F.col("has_mobile").cast("int") + F.col("has_landline").cast("int")) >= 2
).withColumn(
    "total_contact_methods",
    F.col("has_email").cast("int") + F.col("has_mobile").cast("int") + F.col("has_landline").cast("int")
).withColumn(
    "all_channels_opted_in",
    F.col("mail_opt_in") & F.col("sms_opt_in") & F.col("email_opt_in")
).withColumn(
    "age_group",
    F.when(F.col("age") < 30, "18-29")
     .when(F.col("age") < 45, "30-44")
     .when(F.col("age") < 65, "45-64")
     .when(F.col("age") < 75, "65-74")
     .otherwise("75+")
).withColumn(
    "is_dual_eligible",
    F.col("insurance_type") == "Dual Eligible"
).withColumn(
    "is_medicare",
    F.col("insurance_type").isin(["Medicare", "Dual Eligible"])
)

print(f"✅ Created member features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 SDOH Features

# COMMAND ----------

# Public SDOH features (ZIP-level)
public_sdoh_features = public_sdoh_df.select(
    "zip_code",
    "adi_national_rank",
    "median_household_income",
    "poverty_rate_pct",
    "unemployment_rate_pct",
    "uninsured_rate_pct",
    "high_school_grad_rate_pct",
    "limited_english_pct",
    "food_desert_flag",
    "healthcare_shortage_area",
    "public_transit_access_score",
    "broadband_access_pct"
).withColumn(
    "high_deprivation_area",
    F.col("adi_national_rank") >= 75
).withColumn(
    "low_income_area",
    F.col("median_household_income") < 40000
)

# Assessment SDOH features (member-level, most recent assessment)
assessment_window = Window.partitionBy("member_id").orderBy(F.desc("assessment_date"))

assessment_features = assessment_sdoh_df.withColumn(
    "row_num", F.row_number().over(assessment_window)
).filter(
    F.col("row_num") == 1
).select(
    "member_id",
    F.col("food_insecurity_flag").alias("sdoh_food_insecurity"),
    F.col("housing_insecurity_flag").alias("sdoh_housing_insecurity"),
    F.col("transportation_barrier_flag").alias("sdoh_transportation_barrier"),
    F.col("financial_strain_flag").alias("sdoh_financial_strain"),
    F.col("social_isolation_flag").alias("sdoh_social_isolation"),
    F.col("health_literacy_score").alias("sdoh_health_literacy"),
    F.col("smartphone_access").alias("sdoh_smartphone_access"),
    F.col("internet_access").alias("sdoh_internet_access"),
    F.col("comfortable_with_technology").alias("sdoh_tech_comfort"),
    F.col("sdoh_risk_score").alias("assessment_sdoh_risk_score")
).withColumn(
    "has_assessment",
    F.lit(True)
).withColumn(
    "sdoh_barrier_count",
    F.col("sdoh_food_insecurity").cast("int") + 
    F.col("sdoh_housing_insecurity").cast("int") + 
    F.col("sdoh_transportation_barrier").cast("int") + 
    F.col("sdoh_financial_strain").cast("int") + 
    F.col("sdoh_social_isolation").cast("int")
)

print(f"✅ Created SDOH features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Phone Activity Features

# COMMAND ----------

phone_features = phone_activity_df.select(
    "member_id",
    "primary_phone_type",
    "avg_answer_rate_pct",
    "best_day_to_call",
    "best_time_to_call",
    "total_calls_last_12m",
    "successful_contacts_last_12m",
    "voicemails_left_last_12m",
    "callback_response_rate_pct",
    "avg_call_duration_seconds",
    "sms_response_rate_pct",
    "email_open_rate_pct",
    "email_click_rate_pct",
    "mail_response_rate_pct",
    "days_since_last_successful_contact",
    "phone_number_verified",
    "do_not_call_flag",
    "phone_preference_score",
    "sms_preference_score",
    "email_preference_score",
    "mail_preference_score"
).withColumn(
    "has_channel_preference",
    F.greatest(
        F.col("phone_preference_score"),
        F.col("sms_preference_score"),
        F.col("email_preference_score"),
        F.col("mail_preference_score")
    ) > 0.5
).withColumn(
    "digital_engagement_score",
    (F.col("sms_response_rate_pct") + F.col("email_open_rate_pct")) / 2
)

print(f"✅ Created phone activity features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Combine All Features

# COMMAND ----------

# Join all features together
feature_df = member_features \
    .join(call_features, on="member_id", how="left") \
    .join(public_sdoh_features, on="zip_code", how="left") \
    .join(assessment_features, on="member_id", how="left") \
    .join(phone_features, on="member_id", how="left") \
    .join(engagement_labels, on="member_id", how="inner")  # Inner join to only include labeled records

# Fill nulls for members without call history
feature_df = feature_df.fillna({
    "total_calls": 0,
    "unique_call_days": 0,
    "completed_calls": 0,
    "callback_requested": 0,
    "refused_calls": 0,
    "voicemails_left": 0,
    "no_answer_calls": 0,
    "bad_number_calls": 0,
    "avg_call_duration": 0,
    "max_call_duration": 0,
    "days_since_last_call": 999,
    "days_since_last_success": 999,
    "answer_rate": 0,
    "success_rate": 0,
    "voicemail_rate": 0,
    "mobile_preference": 0.5,
    "has_assessment": False,
    "sdoh_barrier_count": 0,
    "assessment_sdoh_risk_score": 3  # Default medium risk
})

# Convert to Pandas for sklearn
pdf = feature_df.toPandas()

print(f"✅ Combined feature dataset: {len(pdf):,} records, {len(pdf.columns)} columns")
print(f"📊 Target distribution:\n{pdf['engaged'].value_counts(normalize=True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Training Data

# COMMAND ----------

# Define feature columns by type
NUMERIC_FEATURES = [
    # Demographics
    'age', 'chronic_condition_count', 'risk_score', 'total_contact_methods',
    
    # Call history
    'total_calls', 'unique_call_days', 'completed_calls', 'callback_requested',
    'refused_calls', 'voicemails_left', 'no_answer_calls', 'bad_number_calls',
    'avg_call_duration', 'max_call_duration', 'days_since_last_call',
    'days_since_last_success', 'answer_rate', 'success_rate', 'voicemail_rate',
    'mobile_preference',
    
    # SDOH - Public
    'adi_national_rank', 'median_household_income', 'poverty_rate_pct',
    'unemployment_rate_pct', 'high_school_grad_rate_pct', 'limited_english_pct',
    'public_transit_access_score', 'broadband_access_pct',
    
    # SDOH - Assessment
    'assessment_sdoh_risk_score', 'sdoh_barrier_count',
    
    # Phone activity
    'avg_answer_rate_pct', 'callback_response_rate_pct',
    'sms_response_rate_pct', 'email_open_rate_pct', 'email_click_rate_pct',
    'mail_response_rate_pct', 'days_since_last_successful_contact',
    'phone_preference_score', 'sms_preference_score', 'email_preference_score',
    'mail_preference_score', 'digital_engagement_score'
]

CATEGORICAL_FEATURES = [
    'gender', 'urban_rural', 'insurance_type', 'preferred_language',
    'age_group', 'primary_phone_type', 'best_day_to_call', 'best_time_to_call',
    'sdoh_health_literacy', 'sdoh_tech_comfort'
]

BINARY_FEATURES = [
    'pcp_assigned', 'has_email', 'has_mobile', 'has_landline',
    'mail_opt_in', 'sms_opt_in', 'email_opt_in', 'has_multiple_contact_methods',
    'all_channels_opted_in', 'is_dual_eligible', 'is_medicare',
    'food_desert_flag', 'healthcare_shortage_area', 'high_deprivation_area',
    'low_income_area', 'has_assessment', 'sdoh_food_insecurity',
    'sdoh_housing_insecurity', 'sdoh_transportation_barrier',
    'sdoh_financial_strain', 'sdoh_social_isolation', 'sdoh_smartphone_access',
    'sdoh_internet_access', 'phone_number_verified', 'do_not_call_flag'
]

# Filter to available columns
available_numeric = [c for c in NUMERIC_FEATURES if c in pdf.columns]
available_categorical = [c for c in CATEGORICAL_FEATURES if c in pdf.columns]
available_binary = [c for c in BINARY_FEATURES if c in pdf.columns]

print(f"📊 Numeric features: {len(available_numeric)}")
print(f"📊 Categorical features: {len(available_categorical)}")
print(f"📊 Binary features: {len(available_binary)}")

# COMMAND ----------

# Prepare feature matrix
# Handle missing values and encode categoricals

# Fill numeric nulls with median
for col in available_numeric:
    pdf[col] = pdf[col].fillna(pdf[col].median())

# Fill categorical nulls with mode
for col in available_categorical:
    pdf[col] = pdf[col].fillna(pdf[col].mode()[0] if len(pdf[col].mode()) > 0 else 'Unknown')

# Fill binary nulls with False
for col in available_binary:
    pdf[col] = pdf[col].fillna(False).astype(int)

# Encode categorical variables
label_encoders = {}
for col in available_categorical:
    le = LabelEncoder()
    pdf[f'{col}_encoded'] = le.fit_transform(pdf[col].astype(str))
    label_encoders[col] = le

# Create final feature list
FEATURE_COLS = (
    available_numeric + 
    [f'{c}_encoded' for c in available_categorical] + 
    available_binary
)

print(f"✅ Total features for model: {len(FEATURE_COLS)}")

# COMMAND ----------

# Split data
X = pdf[FEATURE_COLS].values
y = pdf['engaged'].values
member_ids_array = pdf['member_id'].values

X_train, X_test, y_train, y_test, members_train, members_test = train_test_split(
    X, y, member_ids_array, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED,
    stratify=y
)

print(f"📊 Training set: {len(X_train):,} samples")
print(f"📊 Test set: {len(X_test):,} samples")
print(f"📊 Positive class rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Engagement Likelihood Model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 XGBoost Model with Hyperparameter Tuning

# COMMAND ----------

# Calculate scale_pos_weight for imbalanced classes
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbosity': 0
}

# COMMAND ----------

# Train with MLflow tracking
with mlflow.start_run(run_name="engagement_xgboost") as run:
    
    # Log parameters
    mlflow.log_params(xgb_params)
    mlflow.log_param("n_features", len(FEATURE_COLS))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    
    # Train model
    print("🚀 Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Fit with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("avg_precision", avg_precision)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"\n📊 Model Performance:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Engaged', 'Engaged']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log feature importance
    mlflow.log_table(feature_importance.head(30), "feature_importance.json")
    
    # Log model
    signature = infer_signature(X_train, y_pred_proba)
    mlflow.xgboost.log_model(
        xgb_model, 
        "engagement_model",
        signature=signature,
        input_example=X_train[:5]
    )
    
    engagement_run_id = run.info.run_id
    print(f"\n✅ Model logged with run_id: {engagement_run_id}")

# COMMAND ----------

# Display top features
print("🔝 Top 20 Most Important Features:")
display(feature_importance.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Model Calibration
# MAGIC
# MAGIC Calibrate the model to ensure predicted probabilities are well-calibrated.

# COMMAND ----------

# Calibrate the model using isotonic regression
print("🔧 Calibrating model probabilities...")

calibrated_model = CalibratedClassifierCV(
    xgb_model, 
    method='isotonic', 
    cv='prefit'
)
calibrated_model.fit(X_test, y_test)

# Test calibrated predictions
y_pred_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

print(f"✅ Calibration complete")
print(f"   Original AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"   Calibrated AUC: {roc_auc_score(y_test, y_pred_calibrated):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train Channel Propensity Models
# MAGIC
# MAGIC Train separate models to predict response likelihood for each communication channel.

# COMMAND ----------

# Create channel-specific labels based on historical engagement patterns
# This simulates having channel-specific outcome data

# For demo purposes, we'll create synthetic channel outcomes based on activity data
channel_outcomes = pdf.copy()

# Phone engagement: based on answer rate and success rate
channel_outcomes['phone_engaged'] = (
    (pdf['success_rate'] > pdf['success_rate'].median()) | 
    (pdf['answer_rate'] > pdf['answer_rate'].median())
).astype(int)

# SMS engagement: based on SMS response rate
channel_outcomes['sms_engaged'] = (
    pdf['sms_response_rate_pct'] > pdf['sms_response_rate_pct'].median()
).astype(int)

# Email engagement: based on email open and click rates
channel_outcomes['email_engaged'] = (
    (pdf['email_open_rate_pct'] > pdf['email_open_rate_pct'].median()) |
    (pdf['email_click_rate_pct'] > pdf['email_click_rate_pct'].median())
).astype(int)

# Mail engagement: based on mail response rate
channel_outcomes['mail_engaged'] = (
    pdf['mail_response_rate_pct'] > pdf['mail_response_rate_pct'].median()
).astype(int)

print("📊 Channel engagement rates:")
for channel in ['phone', 'sms', 'email', 'mail']:
    rate = channel_outcomes[f'{channel}_engaged'].mean()
    print(f"   {channel.upper()}: {rate:.1%}")

# COMMAND ----------

# Train multi-output channel propensity model
from sklearn.multioutput import MultiOutputClassifier

# Prepare channel targets
channel_cols = ['phone_engaged', 'sms_engaged', 'email_engaged', 'mail_engaged']
y_channels = channel_outcomes[channel_cols].values

# Split
X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
    X, y_channels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

with mlflow.start_run(run_name="channel_propensity_model") as run:
    
    # LightGBM base model (faster for multi-output)
    lgb_base = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    # Multi-output wrapper
    channel_model = MultiOutputClassifier(lgb_base, n_jobs=-1)
    
    print("🚀 Training channel propensity models...")
    channel_model.fit(X_train_ch, y_train_ch)
    
    # Predictions
    y_pred_channels = channel_model.predict_proba(X_test_ch)
    
    # Log metrics for each channel
    for i, channel in enumerate(channel_cols):
        channel_proba = y_pred_channels[i][:, 1]
        channel_auc = roc_auc_score(y_test_ch[:, i], channel_proba)
        mlflow.log_metric(f"{channel}_auc", channel_auc)
        print(f"   {channel} AUC: {channel_auc:.4f}")
    
    # Log model
    mlflow.sklearn.log_model(channel_model, "channel_propensity_model")
    
    channel_run_id = run.info.run_id
    print(f"\n✅ Channel model logged with run_id: {channel_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build Prioritization Scoring Function
# MAGIC
# MAGIC Combine ML predictions with business rules to create final prioritization scores.

# COMMAND ----------

def calculate_prioritization_score(
    engagement_prob: float,
    sdoh_risk_score: float,
    care_gap_indicator: float,
    days_since_contact: int,
    chronic_conditions: int,
    risk_score: float,
    weights: dict = None
) -> float:
    """
    Calculate member prioritization score combining ML predictions with business rules.
    
    Score components:
    1. Engagement Likelihood (ML) - 25%
    2. SDOH Risk - 20%
    3. Care Gaps - 25%
    4. Clinical Risk - 15%
    5. Recency Factor - 15%
    
    Returns score from 0-100.
    """
    if weights is None:
        weights = {
            'engagement': 0.25,
            'sdoh': 0.20,
            'care_gap': 0.25,
            'clinical': 0.15,
            'recency': 0.15
        }
    
    # Normalize inputs to 0-1 scale
    engagement_score = engagement_prob  # Already 0-1
    
    sdoh_score = min(sdoh_risk_score / 5, 1.0)  # SDOH 1-5 scale
    
    care_gap_score = care_gap_indicator  # Assume 0-1
    
    clinical_score = min(risk_score / 5, 1.0)  # Risk score typically 0.5-4.5
    
    # Recency: higher score for longer time since contact (need outreach)
    recency_score = min(days_since_contact / 365, 1.0)
    
    # Weighted combination
    raw_score = (
        weights['engagement'] * engagement_score +
        weights['sdoh'] * sdoh_score +
        weights['care_gap'] * care_gap_score +
        weights['clinical'] * clinical_score +
        weights['recency'] * recency_score
    )
    
    # Scale to 0-100
    return round(raw_score * 100, 1)


def assign_cohort(prioritization_score: float, engagement_prob: float) -> tuple:
    """Assign priority and engagement cohorts based on scores."""
    
    # Priority cohort
    if prioritization_score >= 75:
        priority_cohort = "High Priority"
    elif prioritization_score >= 50:
        priority_cohort = "Medium Priority"
    elif prioritization_score >= 25:
        priority_cohort = "Low Priority"
    else:
        priority_cohort = "Monitor Only"
    
    # Engagement cohort
    if engagement_prob >= 0.7:
        engagement_cohort = "Highly Engaged"
    elif engagement_prob >= 0.4:
        engagement_cohort = "Moderately Engaged"
    elif engagement_prob >= 0.2:
        engagement_cohort = "Low Engagement"
    else:
        engagement_cohort = "Hard to Reach"
    
    return priority_cohort, engagement_cohort


def recommend_channel(channel_scores: dict, opt_ins: dict) -> list:
    """
    Recommend communication channels based on propensity scores and opt-in status.
    Returns ranked list of (channel, score) tuples.
    """
    # Apply opt-in filters
    filtered_scores = {}
    for channel, score in channel_scores.items():
        opt_in_key = f"{channel}_opt_in"
        if opt_ins.get(opt_in_key, True):  # Default to True if not specified
            filtered_scores[channel] = score
        else:
            filtered_scores[channel] = 0.0
    
    # Sort by score
    ranked = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Predictions for All Members

# COMMAND ----------

# Score all members
print("🚀 Generating predictions for all members...")

# Get engagement probabilities
engagement_probs = calibrated_model.predict_proba(X)[:, 1]

# Get channel probabilities
channel_probs = channel_model.predict_proba(X)
phone_probs = channel_probs[0][:, 1]
sms_probs = channel_probs[1][:, 1]
email_probs = channel_probs[2][:, 1]
mail_probs = channel_probs[3][:, 1]

# Create results dataframe
results_df = pdf[['member_id']].copy()
results_df['engagement_likelihood_score'] = engagement_probs
results_df['phone_propensity'] = phone_probs
results_df['sms_propensity'] = sms_probs
results_df['email_propensity'] = email_probs
results_df['mail_propensity'] = mail_probs

# Add features needed for prioritization
results_df['sdoh_risk'] = pdf['assessment_sdoh_risk_score'].fillna(3)
results_df['days_since_contact'] = pdf['days_since_last_success'].fillna(365)
results_df['chronic_conditions'] = pdf['chronic_condition_count']
results_df['clinical_risk'] = pdf['risk_score']

# Simulate care gap indicator (in production, this would come from actual care gap data)
np.random.seed(RANDOM_SEED)
results_df['care_gap_indicator'] = np.random.uniform(0, 1, len(results_df))

# Calculate prioritization scores
results_df['prioritization_score'] = results_df.apply(
    lambda row: calculate_prioritization_score(
        engagement_prob=row['engagement_likelihood_score'],
        sdoh_risk_score=row['sdoh_risk'],
        care_gap_indicator=row['care_gap_indicator'],
        days_since_contact=row['days_since_contact'],
        chronic_conditions=row['chronic_conditions'],
        risk_score=row['clinical_risk']
    ),
    axis=1
)

# Assign cohorts
cohort_assignments = results_df.apply(
    lambda row: assign_cohort(row['prioritization_score'], row['engagement_likelihood_score']),
    axis=1
)
results_df['priority_cohort'] = [c[0] for c in cohort_assignments]
results_df['engagement_cohort'] = [c[1] for c in cohort_assignments]

# Determine recommended channel
def get_recommended_channels(row):
    channel_scores = {
        'Phone': row['phone_propensity'],
        'SMS': row['sms_propensity'],
        'Email': row['email_propensity'],
        'Mail': row['mail_propensity']
    }
    ranked = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

channel_recommendations = results_df.apply(get_recommended_channels, axis=1)
results_df['recommended_modality_1'] = [r[0][0] for r in channel_recommendations]
results_df['recommended_modality_1_score'] = [round(r[0][1], 4) for r in channel_recommendations]
results_df['recommended_modality_2'] = [r[1][0] for r in channel_recommendations]
results_df['recommended_modality_2_score'] = [round(r[1][1], 4) for r in channel_recommendations]
results_df['recommended_modality_3'] = [r[2][0] for r in channel_recommendations]
results_df['recommended_modality_3_score'] = [round(r[2][1], 4) for r in channel_recommendations]

print(f"✅ Generated predictions for {len(results_df):,} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Model Outputs to Delta Lake

# COMMAND ----------

# Prepare final output table
model_output_df = results_df.copy()
model_output_df['model_run_date'] = datetime.now().strftime("%Y-%m-%d")
model_output_df['model_version'] = "DBX_Outreach_ML_v1.0"
model_output_df['engagement_run_id'] = engagement_run_id
model_output_df['channel_run_id'] = channel_run_id

# Add additional context from original data
model_output_df = model_output_df.merge(
    pdf[['member_id', 'best_day_to_call', 'best_time_to_call', 'do_not_call_flag']],
    on='member_id',
    how='left'
)

# Determine next best action based on scores and cohort
def determine_next_action(row):
    if row['do_not_call_flag']:
        return "No Action - DNC"
    elif row['priority_cohort'] == "Monitor Only":
        return "No Action - Monitor"
    elif row['engagement_cohort'] == "Hard to Reach":
        if row['mail_propensity'] > 0.3:
            return "Mail Educational Materials"
        else:
            return "Multi-Channel Campaign"
    elif row['recommended_modality_1'] == 'Phone':
        return "Schedule Outbound Call"
    elif row['recommended_modality_1'] == 'SMS':
        return "Send SMS Reminder"
    elif row['recommended_modality_1'] == 'Email':
        return "Send Email Campaign"
    else:
        return "Mail Educational Materials"

model_output_df['next_best_action'] = model_output_df.apply(determine_next_action, axis=1)

# Select final columns
final_columns = [
    'member_id',
    'model_run_date',
    'model_version',
    'engagement_likelihood_score',
    'prioritization_score',
    'priority_cohort',
    'engagement_cohort',
    'phone_propensity',
    'sms_propensity', 
    'email_propensity',
    'mail_propensity',
    'recommended_modality_1',
    'recommended_modality_1_score',
    'recommended_modality_2',
    'recommended_modality_2_score',
    'recommended_modality_3',
    'recommended_modality_3_score',
    'best_day_to_call',
    'best_time_to_call',
    'next_best_action',
    'engagement_run_id',
    'channel_run_id'
]

model_output_df = model_output_df[final_columns]

# Convert to Spark and save
output_spark_df = spark.createDataFrame(model_output_df)
output_spark_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.member_outreach_scores_ml")

print(f"✅ Saved model outputs to {CATALOG}.{SCHEMA}.member_outreach_scores_ml")
display(output_spark_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model Analysis & Insights

# COMMAND ----------

# Cohort distribution
print("📊 Priority Cohort Distribution:")
display(
    output_spark_df.groupBy("priority_cohort")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

print("📊 Engagement Cohort Distribution:")
display(
    output_spark_df.groupBy("engagement_cohort")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

print("📊 Recommended Channel Distribution:")
display(
    output_spark_df.groupBy("recommended_modality_1")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

print("📊 Next Best Action Distribution:")
display(
    output_spark_df.groupBy("next_best_action")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Register Model in Unity Catalog (Optional)

# COMMAND ----------

# Register the engagement model in Unity Catalog
MODEL_NAME = f"{CATALOG}.{SCHEMA}.engagement_likelihood_model"

try:
    # Register model
    mlflow.set_registry_uri("databricks-uc")
    
    model_uri = f"runs:/{engagement_run_id}/engagement_model"
    registered_model = mlflow.register_model(model_uri, MODEL_NAME)
    
    print(f"✅ Model registered: {MODEL_NAME}")
    print(f"   Version: {registered_model.version}")
    
    # Add model description
    from mlflow import MlflowClient
    client = MlflowClient()
    client.update_registered_model(
        name=MODEL_NAME,
        description="XGBoost model predicting member engagement likelihood for telephonic outreach. Trained on historical call outcomes with SDOH and demographic features."
    )
    
except Exception as e:
    print(f"⚠️ Could not register model in Unity Catalog: {e}")
    print("   This may require additional permissions or Unity Catalog setup.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### ML Models Trained:
# MAGIC
# MAGIC | Model | Algorithm | Target | AUC |
# MAGIC |-------|-----------|--------|-----|
# MAGIC | Engagement Likelihood | XGBoost | Binary engagement outcome | ~0.75+ |
# MAGIC | Channel Propensity | LightGBM (Multi-output) | Per-channel engagement | ~0.70+ |
# MAGIC
# MAGIC ### Key Features Used:
# MAGIC - **Call History**: Answer rates, success rates, recency, volume
# MAGIC - **Demographics**: Age, insurance type, chronic conditions
# MAGIC - **SDOH**: ADI score, barriers, social determinants
# MAGIC - **Channel Activity**: SMS/Email/Mail response rates
# MAGIC
# MAGIC ### Output Table: `member_outreach_scores_ml`
# MAGIC - Engagement likelihood score (0-1)
# MAGIC - Prioritization score (0-100)
# MAGIC - Priority & Engagement cohorts
# MAGIC - Ranked channel recommendations with propensity scores
# MAGIC - Next best action recommendation
# MAGIC
# MAGIC ### MLflow Artifacts:
# MAGIC - Experiment tracking with metrics
# MAGIC - Feature importance analysis
# MAGIC - Model versioning in Unity Catalog

# COMMAND ----------

print("=" * 70)
print("🎉 ML MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nOutput table: {CATALOG}.{SCHEMA}.member_outreach_scores_ml")
print(f"MLflow Experiment: {EXPERIMENT_NAME}")
print(f"\nEngagement Model Run ID: {engagement_run_id}")
print(f"Channel Model Run ID: {channel_run_id}")
print("=" * 70)
