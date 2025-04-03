import psycopg2
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest

# -----------------------------
# PostgreSQL Connection Parameters
# -----------------------------
conn_params = {
    'dbname': 'AI2',
    'user': 'postgres',
    'password': 'aak101010',
    'host': 'localhost',
    'port': '5432'
}

# Table to back up. (You can change this name if you like.)
TABLE_NAME = "my_table"
BACKUP_FILE = f"{TABLE_NAME}_backup.csv"

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create SQLAlchemy engine for reading data
engine = create_engine(f'postgresql://{conn_params["user"]}:{conn_params["password"]}@{conn_params["host"]}:{conn_params["port"]}/{conn_params["dbname"]}')

# -----------------------------
# Create Source Table if It Doesn't Exist
# -----------------------------
def create_source_table_if_not_exists():
    """
    Checks if the source table exists and, if not, creates it with a sample schema and data.
    """
    create_query = f"""
    CREATE TABLE IF NOT EXISTS "{TABLE_NAME}" (
        id SERIAL PRIMARY KEY,
        data TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    sample_data_query = f"""
    INSERT INTO "{TABLE_NAME}" (data) 
    SELECT 'Sample data ' || generate_series(1, 5)
    WHERE NOT EXISTS (SELECT 1 FROM "{TABLE_NAME}");
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(create_query)
            conn.commit()
            cur.execute(sample_data_query)
            conn.commit()
    logging.info(f'Ensured source table "{TABLE_NAME}" exists with sample data.')

# -----------------------------
# Create or Recreate backup_metrics Table
# -----------------------------
def create_or_recreate_metrics_table():
    """
    Drops (optional) and recreates the backup_metrics table with the correct schema.
    """
    drop_query = "DROP TABLE IF EXISTS backup_metrics;"
    create_query = """
    CREATE TABLE backup_metrics (
        id SERIAL PRIMARY KEY,
        backup_time TIMESTAMP DEFAULT NOW(),
        duration FLOAT,
        status VARCHAR(20)
    );
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(drop_query)
            conn.commit()
            cur.execute(create_query)
            conn.commit()
    logging.info("Created backup_metrics table with proper schema.")

# -----------------------------
# Backup Data and Record Metrics
# -----------------------------
def backup_data():
    start_time = time.time()
    try:
        # Read the table data using SQLAlchemy engine.
        df = pd.read_sql(f'SELECT * FROM "{TABLE_NAME}";', engine)
        df.to_csv(BACKUP_FILE, index=False)
        duration = time.time() - start_time
        logging.info(f"Backup completed in {duration:.2f} seconds, saved as {BACKUP_FILE}")
        record_backup_metric(duration, 'SUCCESS')
        return duration
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Backup failed: {e}")
        record_backup_metric(duration, 'FAILED')
        return None

def record_backup_metric(duration, status):
    insert_query = "INSERT INTO backup_metrics (duration, status) VALUES (%s, %s);"
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(insert_query, (duration, status))
            conn.commit()
    logging.info(f"Recorded backup metric: duration={duration:.2f}, status={status}")

# -----------------------------
# Fetch Historical Backup Durations
# -----------------------------
def fetch_backup_durations():
    query = "SELECT duration FROM backup_metrics;"
    with psycopg2.connect(**conn_params) as conn:
        df = pd.read_sql(query, conn)
    if df.empty:
        logging.warning("No backup duration data found.")
        return np.array([])
    return df['duration'].values

# -----------------------------
# AI: Detect Anomaly in Backup Duration
# -----------------------------
def detect_anomaly(durations):
    if len(durations) == 0:
        logging.warning("Not enough data for anomaly detection.")
        return False
    durations = durations.reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(durations)
    predictions = model.predict(durations)
    # Check if the latest backup duration is flagged as an anomaly (-1)
    is_anomaly = predictions[-1] == -1
    logging.info("Anomaly detection result: %s", "Anomaly detected" if is_anomaly else "No anomaly detected")
    return is_anomaly

# -----------------------------
# Restore Data from Backup
# -----------------------------
def restore_data():
    try:
        df = pd.read_csv(BACKUP_FILE)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Remove existing data in the source table (optional) before restoration
                cur.execute(f'DELETE FROM "{TABLE_NAME}";')
                conn.commit()
                # Insert each row from backup CSV into the source table.
                for _, row in df.iterrows():
                    columns = ', '.join(df.columns)
                    placeholders = ', '.join(['%s'] * len(row))
                    query = f'INSERT INTO "{TABLE_NAME}" ({columns}) VALUES ({placeholders});'
                    cur.execute(query, tuple(row))
                conn.commit()
        logging.info(f"Data restored to table {TABLE_NAME} from {BACKUP_FILE}")
    except Exception as e:
        logging.error(f"Restore failed: {e}")

# -----------------------------
# Main Process
# -----------------------------
if __name__ == "__main__":
    # Ensure the source table and metrics table exist.
    create_source_table_if_not_exists()
    create_or_recreate_metrics_table()

    # Perform the backup process.
    backup_duration = backup_data()
    if backup_duration is not None:
        durations = fetch_backup_durations()
        # If anomaly is detected (i.e. backup took unusually long), trigger restore.
        if detect_anomaly(np.array(durations)):
            logging.warning("Anomaly in backup duration detected. Initiating recovery process...")
            restore_data()
        else:
            logging.info("Backup duration within normal range. No recovery needed.")
