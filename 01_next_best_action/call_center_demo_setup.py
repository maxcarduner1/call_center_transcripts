# Databricks notebook source
# MAGIC %md
# MAGIC # 📞 Call Center Telephonic Outreach - Demo Data Setup
# MAGIC
# MAGIC This notebook generates sample Delta Lake tables in `maxf_demos.call_center` schema for demonstrating the **DBX Model** that:
# MAGIC - Defines cohorts of members by likelihood to engage from telephonic outreach
# MAGIC - Provides a prioritization score
# MAGIC - Recommends the next best communication modality (email, text, mail)
# MAGIC
# MAGIC ## Data Sources Created:
# MAGIC 1. **Historical Telephonic Outreach** - Past call attempts and outcomes
# MAGIC 2. **Public SDOH Data** - Social Determinants of Health from public sources
# MAGIC 3. **Identified Member Data** - 18 months of member information
# MAGIC 4. **Assessment Reported SDOH Data** - Self-reported SDOH from assessments
# MAGIC 5. **Phone Activity Data** - Member phone engagement patterns
# MAGIC 6. **Member Outreach Scores** - Final model output with prioritization scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Catalog and Schema

# COMMAND ----------

# MAGIC %run ../env

# COMMAND ----------

# DBTITLE 1,Untitled
# configure env file to set appropriate catalog/schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries and Define Helper Functions

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import random
from datetime import datetime, timedelta
import uuid

# Set seed for reproducibility
random.seed(42)

# Helper function to generate random dates
def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

# Generate member IDs
NUM_MEMBERS = 10000
member_ids = [f"MBR{str(i).zfill(8)}" for i in range(1, NUM_MEMBERS + 1)]

print(f"✅ Will generate data for {NUM_MEMBERS} members")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Identified Member Data (18 months)
# MAGIC Core member demographic and enrollment information

# COMMAND ----------

# Member demographic data
first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
               "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
               "Maria", "Jose", "Juan", "Ana", "Luis", "Rosa", "Carlos", "Carmen", "Miguel", "Laura"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]

# Urban/Rural classification
urban_rural = ["Urban", "Suburban", "Rural"]

# Insurance types
insurance_types = ["Medicare", "Medicaid", "Dual Eligible", "Commercial", "Exchange"]

# Generate member data
member_data = []
end_date = datetime(2025, 5, 31)
start_date_18m = end_date - timedelta(days=548)  # ~18 months

for member_id in member_ids:
    age = random.choices(
        range(18, 95),
        weights=[1 if a < 30 else 2 if a < 50 else 4 if a < 65 else 6 if a < 75 else 4 if a < 85 else 2 for a in range(18, 95)]
    )[0]
    
    member_data.append({
        "member_id": member_id,
        "first_name": random.choice(first_names),
        "last_name": random.choice(last_names),
        "date_of_birth": (datetime.now() - timedelta(days=age*365 + random.randint(0, 364))).strftime("%Y-%m-%d"),
        "age": age,
        "gender": random.choices(["M", "F", "U"], weights=[48, 50, 2])[0],
        "state": random.choice(states),
        "zip_code": str(random.randint(10000, 99999)),
        "urban_rural": random.choices(urban_rural, weights=[50, 35, 15])[0],
        "insurance_type": random.choices(insurance_types, weights=[30, 25, 15, 20, 10])[0],
        "enrollment_date": random_date(start_date_18m, end_date).strftime("%Y-%m-%d"),
        "pcp_assigned": random.choices([True, False], weights=[75, 25])[0],
        "chronic_condition_count": random.choices([0, 1, 2, 3, 4, 5], weights=[15, 25, 25, 20, 10, 5])[0],
        "risk_score": round(random.uniform(0.5, 4.5), 2),
        "preferred_language": random.choices(
            ["English", "Spanish", "Chinese", "Vietnamese", "Korean", "Other"],
            weights=[70, 15, 5, 3, 2, 5]
        )[0],
        "has_email": random.choices([True, False], weights=[65, 35])[0],
        "has_mobile": random.choices([True, False], weights=[80, 20])[0],
        "has_landline": random.choices([True, False], weights=[45, 55])[0],
        "mail_opt_in": random.choices([True, False], weights=[70, 30])[0],
        "sms_opt_in": random.choices([True, False], weights=[55, 45])[0],
        "email_opt_in": random.choices([True, False], weights=[50, 50])[0]
    })

# Create DataFrame and save as Delta table
member_df = spark.createDataFrame(member_data)
member_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.identified_member_data")

print(f"✅ Created identified_member_data table with {member_df.count()} records")
display(member_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Historical Telephonic Outreach Data
# MAGIC Past 18 months of call attempts and outcomes

# COMMAND ----------

# Call outcomes with realistic distributions
call_outcomes = [
    "Connected - Completed",
    "Connected - Callback Requested", 
    "Connected - Refused",
    "Voicemail Left",
    "No Answer",
    "Busy Signal",
    "Wrong Number",
    "Disconnected Number",
    "Do Not Call"
]

# Call purposes
call_purposes = [
    "Annual Wellness Visit Scheduling",
    "Care Gap Closure",
    "Medication Adherence",
    "Post-Discharge Follow-up",
    "Chronic Care Management",
    "Health Risk Assessment",
    "Benefits Education",
    "Transportation Assistance",
    "Social Services Referral"
]

# Generate call history
call_data = []
call_id = 1

for member_id in member_ids:
    # Each member has 0-15 historical calls
    num_calls = random.choices(range(0, 16), weights=[10, 15, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 1, 0.5, 0.5])[0]
    
    for _ in range(num_calls):
        call_date = random_date(start_date_18m, end_date)
        call_time_hour = random.choices(range(8, 20), weights=[5, 10, 12, 10, 8, 6, 5, 8, 12, 10, 8, 6])[0]
        
        outcome = random.choices(
            call_outcomes,
            weights=[15, 8, 5, 25, 30, 5, 4, 5, 3]
        )[0]
        
        # Call duration based on outcome
        if "Connected" in outcome:
            duration_seconds = random.randint(120, 900)
        elif outcome == "Voicemail Left":
            duration_seconds = random.randint(30, 90)
        else:
            duration_seconds = random.randint(5, 30)
        
        call_data.append({
            "call_id": f"CALL{str(call_id).zfill(10)}",
            "member_id": member_id,
            "call_date": call_date.strftime("%Y-%m-%d"),
            "call_time": f"{call_time_hour:02d}:{random.randint(0,59):02d}:00",
            "call_day_of_week": call_date.strftime("%A"),
            "call_purpose": random.choice(call_purposes),
            "call_outcome": outcome,
            "call_duration_seconds": duration_seconds,
            "call_attempts_same_day": random.choices([1, 2, 3], weights=[70, 25, 5])[0],
            "agent_id": f"AGT{random.randint(100, 500)}",
            "phone_type_called": random.choices(["Mobile", "Landline"], weights=[65, 35])[0],
            "callback_scheduled": outcome == "Connected - Callback Requested",
            "notes": None
        })
        call_id += 1



# COMMAND ----------

call_data[0]

# COMMAND ----------

# Define explicit schema to handle None values
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType

call_schema = StructType([
    StructField("call_id", StringType(), False),
    StructField("member_id", StringType(), False),
    StructField("call_date", StringType(), False),
    StructField("call_time", StringType(), False),
    StructField("call_day_of_week", StringType(), False),
    StructField("call_purpose", StringType(), False),
    StructField("call_outcome", StringType(), False),
    StructField("call_duration_seconds", IntegerType(), False),
    StructField("call_attempts_same_day", IntegerType(), False),
    StructField("agent_id", StringType(), False),
    StructField("phone_type_called", StringType(), False),
    StructField("callback_scheduled", BooleanType(), False),
    StructField("notes", StringType(), True)
])

call_df = spark.createDataFrame(call_data, schema=call_schema)
call_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.historical_telephonic_outreach")

print(f"✅ Created historical_telephonic_outreach table with {call_df.count()} records")
display(call_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Public SDOH Data
# MAGIC Social Determinants of Health data from public sources (Census, ACS, etc.) - ZIP code level

# COMMAND ----------

# Get unique ZIP codes from member data
zip_codes = list(set([m["zip_code"] for m in member_data]))

public_sdoh_data = []

for zip_code in zip_codes:
    # Generate realistic SDOH metrics at ZIP code level
    adi_score = random.randint(1, 100)  # Area Deprivation Index (1=least deprived, 100=most)
    
    public_sdoh_data.append({
        "zip_code": zip_code,
        "fips_code": f"{random.randint(1, 56):02d}{random.randint(1, 999):03d}",
        "county_name": f"County_{random.randint(1, 500)}",
        "adi_national_rank": adi_score,
        "adi_state_rank": random.randint(1, 10),
        "median_household_income": random.randint(25000, 150000),
        "poverty_rate_pct": round(random.uniform(3, 35), 1),
        "unemployment_rate_pct": round(random.uniform(2, 15), 1),
        "uninsured_rate_pct": round(random.uniform(3, 25), 1),
        "high_school_grad_rate_pct": round(random.uniform(65, 98), 1),
        "college_grad_rate_pct": round(random.uniform(10, 60), 1),
        "limited_english_pct": round(random.uniform(1, 30), 1),
        "food_desert_flag": random.choices([True, False], weights=[20, 80])[0],
        "healthcare_shortage_area": random.choices([True, False], weights=[25, 75])[0],
        "public_transit_access_score": random.randint(1, 10),
        "broadband_access_pct": round(random.uniform(50, 99), 1),
        "violent_crime_rate_per_1000": round(random.uniform(0.5, 15), 2),
        "air_quality_index": random.randint(20, 150),
        "walkability_score": random.randint(1, 100),
        "data_year": 2024,
        "data_source": random.choice(["ACS 5-Year", "Census 2020", "CDC SVI", "USDA Food Atlas"])
    })

public_sdoh_df = spark.createDataFrame(public_sdoh_data)
public_sdoh_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.public_sdoh_data")

print(f"✅ Created public_sdoh_data table with {public_sdoh_df.count()} records")
display(public_sdoh_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Assessment Reported SDOH Data
# MAGIC Self-reported Social Determinants of Health from member assessments

# COMMAND ----------

# Not all members complete assessments
assessed_members = random.sample(member_ids, int(NUM_MEMBERS * 0.45))  # 45% completion rate

assessment_data = []

for member_id in assessed_members:
    assessment_date = random_date(start_date_18m, end_date)
    
    # Generate correlated SDOH responses
    has_hardship = random.random() < 0.35  # 35% chance of some hardship
    
    assessment_data.append({
        "assessment_id": f"ASMT{str(uuid.uuid4())[:8].upper()}",
        "member_id": member_id,
        "assessment_date": assessment_date.strftime("%Y-%m-%d"),
        "assessment_type": random.choices(
            ["HRA", "SDOH Screening", "Annual Wellness", "Care Management Intake"],
            weights=[30, 40, 20, 10]
        )[0],
        "assessment_method": random.choices(
            ["Phone", "In-Person", "Digital/Online", "Mail"],
            weights=[35, 25, 30, 10]
        )[0],
        
        # Food Security
        "food_insecurity_flag": random.choices([True, False], weights=[25 if has_hardship else 8, 75 if not has_hardship else 92])[0],
        "food_insecurity_frequency": random.choices(
            ["Never", "Sometimes", "Often", "Always"],
            weights=[60, 25, 10, 5] if not has_hardship else [30, 35, 25, 10]
        )[0],
        
        # Housing
        "housing_insecurity_flag": random.choices([True, False], weights=[20 if has_hardship else 5, 80 if not has_hardship else 95])[0],
        "housing_type": random.choices(
            ["Own", "Rent", "With Family/Friends", "Assisted Living", "Homeless/Shelter"],
            weights=[40, 35, 15, 8, 2]
        )[0],
        "housing_concerns": random.choices(
            ["None", "Affordability", "Safety", "Accessibility", "Utilities", "Multiple"],
            weights=[50, 20, 10, 10, 5, 5] if not has_hardship else [20, 30, 15, 15, 10, 10]
        )[0],
        
        # Transportation
        "transportation_barrier_flag": random.choices([True, False], weights=[30 if has_hardship else 10, 70 if not has_hardship else 90])[0],
        "transportation_access": random.choices(
            ["Personal Vehicle", "Public Transit", "Ride Share", "Family/Friends", "Medical Transport", "Limited Access"],
            weights=[50, 15, 10, 15, 5, 5]
        )[0],
        "missed_appointments_due_to_transport": random.choices([0, 1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 3, 2])[0],
        
        # Financial
        "financial_strain_flag": random.choices([True, False], weights=[35 if has_hardship else 12, 65 if not has_hardship else 88])[0],
        "difficulty_paying_bills": random.choices(
            ["Never", "Rarely", "Sometimes", "Often", "Always"],
            weights=[40, 25, 20, 10, 5] if not has_hardship else [15, 20, 30, 25, 10]
        )[0],
        "medication_cost_barrier": random.choices([True, False], weights=[25 if has_hardship else 8, 75 if not has_hardship else 92])[0],
        
        # Social Support
        "social_isolation_flag": random.choices([True, False], weights=[25, 75])[0],
        "social_support_score": random.randint(1, 10),
        "lives_alone": random.choices([True, False], weights=[30, 70])[0],
        "caregiver_status": random.choices(
            ["Not a Caregiver", "Caregiver - Child", "Caregiver - Adult", "Caregiver - Both"],
            weights=[60, 20, 15, 5]
        )[0],
        
        # Health Literacy
        "health_literacy_score": random.choices(
            ["Adequate", "Limited", "Very Limited"],
            weights=[55, 30, 15]
        )[0],
        "needs_interpreter": random.choices([True, False], weights=[12, 88])[0],
        
        # Technology Access
        "smartphone_access": random.choices([True, False], weights=[75, 25])[0],
        "internet_access": random.choices([True, False], weights=[80, 20])[0],
        "comfortable_with_technology": random.choices(
            ["Very", "Somewhat", "Not Very", "Not at All"],
            weights=[30, 35, 25, 10]
        )[0],
        
        # Safety
        "safety_concerns_flag": random.choices([True, False], weights=[8, 92])[0],
        
        # Overall SDOH Risk Score (calculated)
        "sdoh_risk_score": random.randint(1, 5) if not has_hardship else random.randint(3, 5),
        "referrals_made": random.choices([True, False], weights=[40 if has_hardship else 15, 60 if not has_hardship else 85])[0]
    })

assessment_df = spark.createDataFrame(assessment_data)
assessment_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.assessment_reported_sdoh_data")

print(f"✅ Created assessment_reported_sdoh_data table with {assessment_df.count()} records")
display(assessment_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Phone Activity Data
# MAGIC Member phone engagement patterns and communication preferences

# COMMAND ----------

phone_activity_data = []

for member_id in member_ids:
    member_info = next(m for m in member_data if m["member_id"] == member_id)
    
    # Generate phone activity metrics
    has_mobile = member_info["has_mobile"]
    has_landline = member_info["has_landline"]
    
    # Engagement propensity based on various factors
    base_engagement = random.uniform(0.2, 0.9)
    
    phone_activity_data.append({
        "member_id": member_id,
        "primary_phone_type": "Mobile" if has_mobile else ("Landline" if has_landline else "None"),
        "secondary_phone_type": "Landline" if has_mobile and has_landline else ("Mobile" if has_landline and not has_mobile else None),
        
        # Call answer patterns
        "avg_answer_rate_pct": round(random.uniform(10, 80), 1),
        "best_day_to_call": random.choices(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            weights=[12, 14, 15, 14, 13, 18, 14]
        )[0],
        "best_time_to_call": random.choices(
            ["Morning (8-11)", "Midday (11-14)", "Afternoon (14-17)", "Evening (17-20)"],
            weights=[25, 20, 25, 30]
        )[0],
        "worst_time_to_call": random.choices(
            ["Morning (8-11)", "Midday (11-14)", "Afternoon (14-17)", "Evening (17-20)"],
            weights=[30, 25, 25, 20]
        )[0],
        
        # Historical engagement
        "total_calls_last_12m": random.randint(0, 20),
        "successful_contacts_last_12m": random.randint(0, 10),
        "voicemails_left_last_12m": random.randint(0, 15),
        "callback_response_rate_pct": round(random.uniform(0, 60), 1),
        "avg_call_duration_seconds": random.randint(60, 600),
        
        # SMS engagement
        "sms_response_rate_pct": round(random.uniform(20, 90), 1) if member_info["sms_opt_in"] else 0,
        "sms_clicks_last_12m": random.randint(0, 30) if member_info["sms_opt_in"] else 0,
        "sms_opt_out_flag": not member_info["sms_opt_in"],
        
        # Email engagement  
        "email_open_rate_pct": round(random.uniform(10, 70), 1) if member_info["email_opt_in"] else 0,
        "email_click_rate_pct": round(random.uniform(2, 30), 1) if member_info["email_opt_in"] else 0,
        "emails_opened_last_12m": random.randint(0, 50) if member_info["email_opt_in"] else 0,
        "email_unsubscribed_flag": not member_info["email_opt_in"],
        
        # Mail engagement
        "mail_response_rate_pct": round(random.uniform(5, 40), 1) if member_info["mail_opt_in"] else 0,
        "mail_returned_undeliverable": random.choices([True, False], weights=[5, 95])[0],
        
        # Contact recency
        "days_since_last_successful_contact": random.randint(0, 365),
        "days_since_last_attempt": random.randint(0, 180),
        
        # Phone number quality indicators
        "phone_number_verified": random.choices([True, False], weights=[70, 30])[0],
        "phone_number_age_days": random.randint(30, 1800),
        "suspected_wrong_number": random.choices([True, False], weights=[5, 95])[0],
        "do_not_call_flag": random.choices([True, False], weights=[3, 97])[0],
        
        # Calculated preference score
        "phone_preference_score": round(base_engagement * random.uniform(0.8, 1.2), 3),
        "sms_preference_score": round(base_engagement * random.uniform(0.5, 1.5), 3) if has_mobile else 0,
        "email_preference_score": round(base_engagement * random.uniform(0.4, 1.3), 3) if member_info["has_email"] else 0,
        "mail_preference_score": round(base_engagement * random.uniform(0.3, 1.0), 3),
        
        "last_updated": end_date.strftime("%Y-%m-%d")
    })

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType
schema = StructType([
    StructField('member_id', StringType(), True),
    StructField('primary_phone_type', StringType(), True),
    StructField('secondary_phone_type', StringType(), True),
    StructField('avg_answer_rate_pct', FloatType(), True),
    StructField('best_day_to_call', StringType(), True),
    StructField('best_time_to_call', StringType(), True),
    StructField('worst_time_to_call', StringType(), True),
    StructField('total_calls_last_12m', IntegerType(), True),
    StructField('successful_contacts_last_12m', IntegerType(), True),
    StructField('voicemails_left_last_12m', IntegerType(), True),
    StructField('callback_response_rate_pct', FloatType(), True),
    StructField('avg_call_duration_seconds', IntegerType(), True),
    StructField('sms_response_rate_pct', FloatType(), True),
    StructField('sms_clicks_last_12m', IntegerType(), True),
    StructField('sms_opt_out_flag', BooleanType(), True),
    StructField('email_open_rate_pct', FloatType(), True),
    StructField('email_click_rate_pct', FloatType(), True),
    StructField('emails_opened_last_12m', IntegerType(), True),
    StructField('email_unsubscribed_flag', BooleanType(), True),
    StructField('mail_response_rate_pct', FloatType(), True),
    StructField('mail_returned_undeliverable', BooleanType(), True),
    StructField('days_since_last_successful_contact', IntegerType(), True),
    StructField('days_since_last_attempt', IntegerType(), True),
    StructField('phone_number_verified', BooleanType(), True),
    StructField('phone_number_age_days', IntegerType(), True),
    StructField('suspected_wrong_number', BooleanType(), True),
    StructField('do_not_call_flag', BooleanType(), True),
    StructField('phone_preference_score', FloatType(), True),
    StructField('sms_preference_score', FloatType(), True),
    StructField('email_preference_score', FloatType(), True),
    StructField('mail_preference_score', FloatType(), True),
    StructField('last_updated', StringType(), True)
])

phone_activity_df = spark.createDataFrame(phone_activity_data, schema=schema)

phone_activity_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.phone_activity_data")

print(f"✅ Created phone_activity_data table with {phone_activity_df.count()} records")
display(phone_activity_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Member Outreach Scores (DBX Model Output)
# MAGIC Final model output with prioritization scores and recommended communication modality

# COMMAND ----------

# Generate model output scores
outreach_scores_data = []

for member_id in member_ids:
    member_info = next(m for m in member_data if m["member_id"] == member_id)
    phone_info = next(p for p in phone_activity_data if p["member_id"] == member_id)
    
    # Get assessment info if available
    assessment_info = next((a for a in assessment_data if a["member_id"] == member_id), None)
    
    # Calculate engagement likelihood based on multiple factors
    base_score = random.uniform(0.1, 0.9)
    
    # Adjust based on factors
    if phone_info["do_not_call_flag"]:
        base_score *= 0.1
    if phone_info["phone_number_verified"]:
        base_score *= 1.2
    if member_info["age"] > 75:
        base_score *= 0.9  # Slightly lower for very elderly
    if assessment_info and assessment_info.get("transportation_barrier_flag"):
        base_score *= 1.1  # Higher priority for those with barriers
    
    # Cap at 1.0
    engagement_likelihood = min(round(base_score, 4), 1.0)
    
    # Determine best communication modality based on preferences and engagement
    modality_scores = {
        "Phone": phone_info["phone_preference_score"],
        "SMS": phone_info["sms_preference_score"] if not phone_info["sms_opt_out_flag"] else 0.0,
        "Email": phone_info["email_preference_score"] if not phone_info["email_unsubscribed_flag"] else 0.0,
        "Mail": phone_info["mail_preference_score"]
    }
    
    # Sort by score
    sorted_modalities = sorted(modality_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate prioritization score (1-100)
    # Higher score = higher priority for outreach
    priority_factors = []
    
    # Care gap factor
    care_gap_score = random.uniform(0, 40)
    priority_factors.append(care_gap_score)
    
    # SDOH risk factor
    if assessment_info:
        sdoh_score = float(assessment_info["sdoh_risk_score"] * 8)  # 0-40 points
    else:
        sdoh_score = random.uniform(10, 25)  # Unknown = medium priority
    priority_factors.append(sdoh_score)
    
    # Engagement likelihood factor
    engagement_score = engagement_likelihood * 20  # 0-20 points
    priority_factors.append(engagement_score)
    
    prioritization_score = min(round(sum(priority_factors), 1), 100)
    
    # Assign cohort based on prioritization score
    if prioritization_score >= 75:
        cohort = "High Priority"
    elif prioritization_score >= 50:
        cohort = "Medium Priority"
    elif prioritization_score >= 25:
        cohort = "Low Priority"
    else:
        cohort = "Monitor Only"
    
    # Engagement cohort based on likelihood
    if engagement_likelihood >= 0.7:
        engagement_cohort = "Highly Engaged"
    elif engagement_likelihood >= 0.4:
        engagement_cohort = "Moderately Engaged"
    elif engagement_likelihood >= 0.2:
        engagement_cohort = "Low Engagement"
    else:
        engagement_cohort = "Hard to Reach"
    
    outreach_scores_data.append({
        "member_id": member_id,
        "model_run_date": end_date.strftime("%Y-%m-%d"),
        "model_version": "DBX_Outreach_v2.3",
        
        # Primary Scores
        "engagement_likelihood_score": engagement_likelihood,
        "prioritization_score": prioritization_score,
        
        # Cohort Assignments
        "priority_cohort": cohort,
        "engagement_cohort": engagement_cohort,
        
        # Recommended Communication
        "recommended_modality_1": sorted_modalities[0][0],
        "recommended_modality_1_score": float(round(sorted_modalities[0][1], 4)),
        "recommended_modality_2": sorted_modalities[1][0],
        "recommended_modality_2_score": float(round(sorted_modalities[1][1], 4)),
        "recommended_modality_3": sorted_modalities[2][0],
        "recommended_modality_3_score": float(round(sorted_modalities[2][1], 4)),
        
        # Best Contact Time
        "recommended_call_day": phone_info["best_day_to_call"],
        "recommended_call_time": phone_info["best_time_to_call"],
        
        # Component Scores (for explainability)
        "care_gap_score": round(care_gap_score, 2),
        "sdoh_risk_score": round(sdoh_score, 2),
        "engagement_history_score": round(engagement_score, 2),
        
        # Risk Flags
        "high_sdoh_risk_flag": sdoh_score >= 30,
        "care_gap_flag": care_gap_score >= 25,
        "declining_engagement_flag": random.choices([True, False], weights=[15, 85])[0],
        "recent_life_event_flag": random.choices([True, False], weights=[10, 90])[0],
        
        # Exclusion Flags
        "do_not_contact_flag": phone_info["do_not_call_flag"],
        "deceased_flag": False,
        "termed_flag": random.choices([True, False], weights=[2, 98])[0],
        
        # Recommended Actions
        "recommended_outreach_purpose": random.choice(call_purposes),
        "recommended_agent_skill_level": random.choices(
            ["Standard", "Senior", "Specialist", "Bilingual"],
            weights=[50, 25, 15, 10]
        )[0],
        
        # Model Confidence
        "prediction_confidence": round(random.uniform(0.6, 0.95), 3),
        
        # Next Best Action
        "next_best_action": random.choices([
            "Schedule Outbound Call",
            "Send SMS Reminder",
            "Send Email Campaign",
            "Mail Educational Materials",
            "No Action - Monitor",
            "Refer to Care Management"
        ], weights=[35, 20, 15, 10, 15, 5])[0]
    })

outreach_scores_df = spark.createDataFrame(outreach_scores_data)
outreach_scores_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.member_outreach_scores")

print(f"✅ Created member_outreach_scores table with {outreach_scores_df.count()} records")
display(outreach_scores_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary View - Outreach Campaign List
# MAGIC A denormalized view combining key fields for easy campaign execution

# COMMAND ----------

# Create a view joining key tables for campaign execution
spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.outreach_campaign_list AS
SELECT 
    m.member_id,
    m.first_name,
    m.last_name,
    m.age,
    m.gender,
    m.state,
    m.zip_code,
    m.preferred_language,
    m.insurance_type,
    m.chronic_condition_count,
    m.risk_score as clinical_risk_score,
    m.has_email,
    m.has_mobile,
    m.sms_opt_in,
    m.email_opt_in,
    
    o.prioritization_score,
    o.engagement_likelihood_score,
    o.priority_cohort,
    o.engagement_cohort,
    o.recommended_modality_1,
    o.recommended_modality_2,
    o.recommended_call_day,
    o.recommended_call_time,
    o.recommended_outreach_purpose,
    o.next_best_action,
    o.high_sdoh_risk_flag,
    o.care_gap_flag,
    o.do_not_contact_flag,
    
    p.avg_answer_rate_pct,
    p.days_since_last_successful_contact,
    p.best_day_to_call,
    p.best_time_to_call,
    
    s.adi_national_rank as area_deprivation_index,
    s.median_household_income as zip_median_income,
    s.food_desert_flag as zip_food_desert,
    s.healthcare_shortage_area as zip_healthcare_shortage
    
FROM {CATALOG}.{SCHEMA}.identified_member_data m
JOIN {CATALOG}.{SCHEMA}.member_outreach_scores o ON m.member_id = o.member_id
JOIN {CATALOG}.{SCHEMA}.phone_activity_data p ON m.member_id = p.member_id
LEFT JOIN {CATALOG}.{SCHEMA}.public_sdoh_data s ON m.zip_code = s.zip_code
WHERE o.do_not_contact_flag = FALSE
  AND o.deceased_flag = FALSE
  AND o.termed_flag = FALSE
""")

campaign_df = spark.table(f"{CATALOG}.{SCHEMA}.outreach_campaign_list")
print(f"✅ Created outreach_campaign_list view with {campaign_df.count()} actionable records")
display(campaign_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary: Tables Created
# MAGIC
# MAGIC | Table Name | Description | Record Count |
# MAGIC |------------|-------------|--------------|
# MAGIC | `identified_member_data` | Core member demographics and enrollment | 10,000 |
# MAGIC | `historical_telephonic_outreach` | 18 months of call history | ~50,000 |
# MAGIC | `public_sdoh_data` | ZIP-level SDOH from public sources | ~9,000 |
# MAGIC | `assessment_reported_sdoh_data` | Member self-reported SDOH | ~4,500 |
# MAGIC | `phone_activity_data` | Phone engagement patterns | 10,000 |
# MAGIC | `member_outreach_scores` | DBX Model output with scores | 10,000 |
# MAGIC | `outreach_campaign_list` (View) | Denormalized campaign execution view | ~9,500 |
