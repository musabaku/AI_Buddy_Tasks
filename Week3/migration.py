import psycopg2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ai_transform(content):
    """
    Simulate an AI transformation on the data:
      - Convert content to uppercase.
      - Simulate a sentiment analysis: if content length > 50, label as 'Positive', else 'Neutral'.
    Replace this function with your actual AI integration if needed.
    """
    transformed_content = content.upper()
    sentiment = 'Positive' if len(content) > 50 else 'Neutral'
    return transformed_content, sentiment

def ensure_tables(source_cur, target_cur, source_conn, target_conn):
    """
    Create the source and target tables if they do not exist.
    """
    source_table_sql = """
    CREATE TABLE IF NOT EXISTS source_table (
        id SERIAL PRIMARY KEY,
        content TEXT
    );
    """
    target_table_sql = """
    CREATE TABLE IF NOT EXISTS target_table (
        id INTEGER PRIMARY KEY,
        content TEXT,
        ai_sentiment TEXT
    );
    """
    try:
        source_cur.execute(source_table_sql)
        source_conn.commit()
        logging.info("Ensured source_table exists.")
    except Exception as e:
        logging.error("Error creating source_table: %s", e)

    try:
        target_cur.execute(target_table_sql)
        target_conn.commit()
        logging.info("Ensured target_table exists.")
    except Exception as e:
        logging.error("Error creating target_table: %s", e)

def seed_source_table_if_empty(source_cur, source_conn):
    """
    Check if source_table is empty. If so, insert sample data for testing.
    """
    try:
        source_cur.execute("SELECT COUNT(*) FROM source_table;")
        count = source_cur.fetchone()[0]
        if count == 0:
            sample_data = [
                ("This is a sample record for AI transformation testing.",),
                ("Short text.",),
                ("Another sample record with a bit more content to test the sentiment analysis functionality.",)
            ]
            source_cur.executemany("INSERT INTO source_table (content) VALUES (%s);", sample_data)
            source_conn.commit()
            logging.info("Inserted sample data into source_table as it was empty.")
        else:
            logging.info("source_table already has data. Skipping seeding.")
    except Exception as e:
        logging.error("Error checking/inserting sample data in source_table: %s", e)

def migrate_data(source_conn_params, target_conn_params):
    try:
        # Connect to the source PostgreSQL database using provided parameters
        source_conn = psycopg2.connect(**source_conn_params)
        source_cur = source_conn.cursor()
        logging.info("Connected to source database.")

        # Connect to the target PostgreSQL database
        target_conn = psycopg2.connect(**target_conn_params)
        target_cur = target_conn.cursor()
        logging.info("Connected to target database.")

        # Ensure both tables exist
        ensure_tables(source_cur, target_cur, source_conn, target_conn)
        
        # Seed source_table with sample data if it's empty
        seed_source_table_if_empty(source_cur, source_conn)

        # Retrieve data from the source table
        source_cur.execute("SELECT id, content FROM source_table;")
        rows = source_cur.fetchall()
        logging.info(f"Retrieved {len(rows)} rows from source_table.")

        migrated_count = 0
        skipped_count = 0

        for row in rows:
            record_id, content = row

            # Data validation: skip if content is None or empty
            if not content:
                logging.warning(f"Skipping record id {record_id} due to empty content.")
                skipped_count += 1
                continue

            # Use AI integration to transform the content and simulate sentiment analysis
            transformed_content, sentiment = ai_transform(content)

            try:
                # Insert the transformed data into the target table
                target_cur.execute(
                    "INSERT INTO target_table (id, content, ai_sentiment) VALUES (%s, %s, %s)",
                    (record_id, transformed_content, sentiment)
                )
                migrated_count += 1
                logging.info(f"Migrated record id {record_id} with sentiment: {sentiment}")
            except Exception as e:
                logging.error(f"Error inserting record id {record_id}: {e}")

        # Commit the changes to the target database
        target_conn.commit()
        logging.info(f"Migration completed: {migrated_count} records migrated, {skipped_count} skipped.")

    except Exception as e:
        logging.error(f"Error during migration: {e}")
    finally:
        # Clean up by closing all connections and cursors
        if 'source_cur' in locals():
            source_cur.close()
        if 'target_cur' in locals():
            target_cur.close()
        if 'source_conn' in locals():
            source_conn.close()
        if 'target_conn' in locals():
            target_conn.close()
        logging.info("Database connections closed.")

if __name__ == "__main__":
    # Connection parameters for the source PostgreSQL database using the provided parameters
    source_conn_params = {
        'dbname': 'AI2',
        'user': 'postgres',
        'password': 'aak101010',
        'host': 'localhost',
        'port': '5432'
    }
    
    # Connection parameters for the target PostgreSQL database (modify these as needed)
    target_conn_params = {
        'dbname': 'target_db',
        'user': 'postgres',
        'password': 'aak101010',
        'host': 'localhost',
        'port': '5432'
    }

    # Perform the migration
    migrate_data(source_conn_params, target_conn_params)
