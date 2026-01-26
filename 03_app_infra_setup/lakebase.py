
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database
from databricks.sdk.service.database import DatabaseInstance
import uuid
import psycopg2
import logging
logger = logging.getLogger(__name__)

def get_lakebase_instance(wc: WorkspaceClient, instance_name: str):
    try:
        logger.info(f"Checking if Lakebase instance '{instance_name}' exists...")
        existing_instance = wc.database.get_database_instance(name=instance_name)
        logger.info(
            f"Instance '{instance_name}' already exists. Returning existing instance."
        )
        if not existing_instance:
            raise Exception(f"Error: Instance '{instance_name}' not found.")
        return existing_instance
    except Exception as e:
        logger.error(f"Error checking instance existence: {str(e)}")
        raise e

# COMMAND ----------

def get_or_create_lakebase_instance(
    wc: WorkspaceClient,
    instance_name: str,
    capacity: str = "CU_1",
    node_count: int = 1,
    enable_readable_secondaries: bool = False,
    retention_window_in_days: int = 7,
) -> DatabaseInstance:
    """
    Get or create a Lakebase instance.

    Args:
        instance_name: Name of the instance to create/get
        capacity: Capacity of the instance (default: CU_1)
        node_count: Number of nodes (default: 1)
        enable_readable_secondaries: Whether to enable readable secondaries (default: False)
        retention_window_in_days: Retention window in days (default: 7)

    Returns:
        DatabaseInstance object
    """


    # Check if instance already exists
    try:
        existing_instance = get_lakebase_instance(wc, instance_name)
        return existing_instance
    except Exception as e:
        pass

    # Instance doesn't exist, create it
    logger.info(f"Creating new Lakebase instance '{instance_name}'...")
    instance = DatabaseInstance(
        name=instance_name,
        capacity=capacity,
        node_count=node_count,
        enable_readable_secondaries=enable_readable_secondaries,
        retention_window_in_days=retention_window_in_days,
    )

    instance_create = wc.database.create_database_instance_and_wait(instance)
    if not instance_create:
        raise Exception(f"Error: Instance '{instance_name}' not created.")
    return instance

def get_lakebase_token(wc: WorkspaceClient, instance_name: str):
    try:
        cred = wc.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[instance_name])
        return cred.token
    except Exception as e:
        logger.error(f"Error generating Lakebase token: {str(e)}")
        raise e

def create_lakebase_connection(wc: WorkspaceClient, instance: DatabaseInstance, db_name: str, user: str):
    cred = get_lakebase_token(wc, instance.name)
    conn = psycopg2.connect(
        host=instance.read_write_dns,
        dbname=db_name,
        user=user,
        password=cred,
        sslmode="require"
    )
    return conn

def grant_connect_to_database(wc: WorkspaceClient, instance: DatabaseInstance, db_name: str, user: str):
    conn = create_lakebase_connection(wc, instance, db_name, user)
    with conn.cursor() as cur:
        cur.execute(f"GRANT CONNECT ON DATABASE {db_name} TO {user}")
        conn.commit()
    conn.close()
    return True