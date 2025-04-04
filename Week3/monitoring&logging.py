import psycopg2
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# PostgreSQL connection details
db_params = {
    'dbname': 'Monitoring&Logging',  
    'user': 'postgres',
    'password': 'aak101010',
    'host': 'localhost',
    'port': '5432'
}

error_counts = []  # For anomaly detection

def create_logs_table():
    """
    Create the migration_logs table if it doesn't exist.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migration_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level VARCHAR(10),
                message TEXT,
                action_taken TEXT
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        logging.info("Table 'migration_logs' created or already exists.")
    except Exception as e:
        logging.error("Error creating table: %s", e)

def ai_log_categorization(message):
    """
    AI-enhanced log categorization based on message content.
    """
    if "skip" in message.lower():
        return "WARNING"
    elif "error" in message.lower() or "fail" in message.lower():
        return "ERROR"
    else:
        return "INFO"

def detect_anomaly(new_errors):
    """
    Detect anomalies if the number of new errors spikes.
    """
    error_counts.append(new_errors)
    if len(error_counts) > 10:
        avg_errors = np.mean(error_counts)
        if new_errors > avg_errors * 1.5:
            logging.warning("Anomaly detected: Sudden spike in errors!")

def store_log_in_db(level, message, action_taken):
    """
    Store a log entry into the migration_logs table.
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        sql = "INSERT INTO migration_logs (level, message, action_taken) VALUES (%s, %s, %s);"
        cur.execute(sql, (level, message, action_taken))
        conn.commit()
        cur.close()
        conn.close()
        logging.info("Log stored in database.")
    except Exception as e:
        logging.error("Error storing log: %s", e)

def log_message(message, action_taken="None"):
    """
    Log a message using AI-enhanced categorization and store it in the database.
    """
    category = ai_log_categorization(message)
    if category == "INFO":
        logging.info(message)
    elif category == "WARNING":
        logging.warning(message)
    elif category == "ERROR":
        logging.error(message)
    store_log_in_db(category, message, action_taken)

def simulate_logging():
    """
    Simulate logging events for monitoring.
    """
    create_logs_table()  # Ensure table exists before logging

    # Simulated log events
    log_message("System startup: All systems nominal.", "Initialization complete")
    log_message("User login successful.", "Authentication passed")
    log_message("Skipping non-critical update for record ID 5.", "Update skipped")
    log_message("Error connecting to service endpoint.", "Connection retry")
    log_message("Failed to load module 'XYZ'.", "Module load failure")

    # Simulate anomaly detection by adding a new error count
    detect_anomaly(new_errors=3)

if __name__ == "__main__":
    simulate_logging()
