# Databricks notebook source
# MAGIC %pip install --upgrade -qqqq databricks-sdk
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,CONFIGS
local_service_principal_name = 'local-service-principal-mc'

group_name = 'mc-call-center-vibing'

LAKEBASE_INSTANCE_NAME = 'telco-call-center'

LAKEBASE_DB_NAME = 'public'

app_name = "call-center-analytics-app"


# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
user_email = w.current_user.me().user_name
username = user_email.split('@')[0].replace('.', '_')
DATABRICKS_HOST = w.config.host

# COMMAND ----------

# DBTITLE 1,create the app
import apps

app = apps.get_or_create_app(wc=w, name=app_name)
print(app_name)

# COMMAND ----------

# DBTITLE 1,create service principal for local dev
# create service principal & secret for service principal
import service_principals
from databricks.sdk.service.oauth2 import ServicePrincipalSecretsProxyAPI

sp = service_principals.get_or_create_service_principal(w, local_service_principal_name)

secret = w.service_principal_secrets_proxy.create(service_principal_id=sp.id)

DATABRICKS_CLIENT_ID = sp.application_id
DATABRICKS_CLIENT_SECRET = secret.secret

# COMMAND ----------

# DBTITLE 1,Create a group to authenticate to lakebase
import groups

group = groups.create_group_if_not_exists(wc=w, group_name=group_name)

# COMMAND ----------

# DBTITLE 1,Get or Create Lakebase Instance
import lakebase
import time

print("fetching instance")
instance = lakebase.get_or_create_lakebase_instance(w, LAKEBASE_INSTANCE_NAME) # fetches existing instance if already created. On fevm, will create.
time.sleep(30) # in case the instance is still starting up

# COMMAND ----------

# DBTITLE 1,Create our own db within Lakebase Instance
default_database_name = 'databricks_postgres'
target_database_name = LAKEBASE_DB_NAME
instance = lakebase.get_or_create_lakebase_instance(w, LAKEBASE_INSTANCE_NAME)

conn = lakebase.create_lakebase_connection(wc=w, instance=instance, db_name=default_database_name, user=user_email)
conn.set_isolation_level(0)
with conn.cursor() as cursor:
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{target_database_name}'")
    if not cursor.fetchone():
        cursor.execute(f'CREATE DATABASE "{target_database_name}"')
    else:
        print('database already existed, skipping')
    cursor.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC now sync table with lakebase from UC (transcripts)

# COMMAND ----------

# For your env:
print(f"""
      COPY/PASTE THESE INTO YOUR .env FILE ON YOUR MACHINE!
    {LAKEBASE_INSTANCE_NAME=}\n
    {LAKEBASE_DB_NAME=}\n
    {DATABRICKS_CLIENT_ID=}\n
    {DATABRICKS_CLIENT_SECRET=}\n
    {DATABRICKS_HOST=}
""")

# COMMAND ----------

displayHTML(f'<a style="font-size: 30px" href="/settings/workspace/identity-and-access/groups/{group.id}">Click this link, and add "{local_service_principal_name}" and your App\'s service principal ({app.service_principal_client_id}) to the group!</a>')
print(app.service_principal_client_id + " <--- your app's service principal for easy copying")
print(sp.application_id + " <--- your local development service principal for easy copying")

# COMMAND ----------

#also add group to lakebase permissions

# COMMAND ----------

# THIS IS A TEST TO SEE IF YOU SUCCESSFULLY ADDED YOUR LOCAL SERVICE PRINCIPAL TO THE ABOVE GROUP PROPERLY

from databricks.sdk import WorkspaceClient
w2 = WorkspaceClient(
    host = w.config.host,
    client_id = DATABRICKS_CLIENT_ID,
    client_secret = DATABRICKS_CLIENT_SECRET
)

conn = lakebase.create_lakebase_connection(wc=w2, instance=instance, db_name=LAKEBASE_DB_NAME, user=group_name)

with conn.cursor() as cursor:
    cursor.execute(f"""
        SELECT * from public.analytics.transcriptions_scored_lbp_sync;
    """)
    res = cursor.fetchall()
    print(res)
conn.close()

# COMMAND ----------

# DBTITLE 1,create the table we need in the public.analtyics schema for human eval


print("creating connection")
conn = lakebase.create_lakebase_connection(wc=w, instance=instance, db_name=LAKEBASE_DB_NAME, user=user_email)
conn.set_isolation_level(0) 
with conn.cursor() as cursor:
    print("creating table...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS public.analytics.human_evaluations (
            evaluation_id SERIAL PRIMARY KEY,
            call_id TEXT NOT NULL,
            evaluator_name TEXT,
            evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            scorecard_overrides JSONB,
            total_score_override INTEGER,
            feedback_text TEXT,
            UNIQUE(call_id)
        );
    """)
    cursor.execute(f"""GRANT ALL PRIVILEGES ON DATABASE {LAKEBASE_DB_NAME} TO PUBLIC""")
    cursor.execute(f"SELECT 1 FROM pg_roles WHERE rolname = '{group.display_name}'")
    role_exists = cursor.fetchone()
    if not role_exists:
        cursor.execute(f"SELECT databricks_create_role('{group.display_name}','GROUP');")
    cursor.execute(f"""GRANT ALL PRIVILEGES ON DATABASE {LAKEBASE_DB_NAME} TO \"{group.display_name}\"""")
    cursor.execute(f"""GRANT ALL PRIVILEGES ON SCHEMA public TO \"{group.display_name}\"""")
    cursor.execute(f"""GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO \"{group.display_name}\"""")
conn.close()
