# Databricks notebook source
CATALOG = "cmegdemos_catalog"
SCHEMA = "telco_call_center_analytics"

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"✅ Using catalog: {CATALOG}, schema: {SCHEMA}")
