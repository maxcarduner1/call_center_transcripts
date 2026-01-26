# Databricks notebook source
CATALOG = "telco_call_center"
SCHEMA = "analytics"

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
