import psycopg2
import random
import time
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "AiIntegration"
DB_USER = "postgres"
DB_PASSWORD = "aak101010"

# ------------------------------------------------------------
# 1. Connect to PostgreSQL
# ------------------------------------------------------------
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    print("Connected to PostgreSQL successfully.")
except Exception as e:
    print("Error connecting to PostgreSQL:", e)
    exit(1)

# ------------------------------------------------------------
# 2. Reset pg_stat_statements (Optional, for a clean slate)
# ------------------------------------------------------------
try:
    cursor.execute("SELECT pg_stat_statements_reset();")
    conn.commit()
    print("pg_stat_statements has been reset. Starting fresh data collection.")
except Exception as e:
    print("Warning: Could not reset pg_stat_statements:", e)

# ------------------------------------------------------------
# 3. Generate Synthetic Load
# ------------------------------------------------------------
# We'll run multiple types of queries:
#   1) pg_sleep(...) queries
#   2) SELECT random() < some_param
# Each query will have a random parameter appended to the text to help avoid normalization.

print("Generating synthetic queries...")

# Define "groups" for sleep times: quick, medium, slow
groups = {
    "quick": (0.01, 0.05),   # ~10-50 ms
    "medium": (0.1, 0.5),    # ~100-500 ms
    "slow": (1.0, 2.0)       # ~1-2 seconds
}

# Possible query templates (varied to produce distinct query texts)
query_templates = [
    "SELECT pg_sleep({sleep_time}); -- group={group_name}, param={param}",
    "SELECT random() < {param}; -- group={group_name}"
]

for group_name, (low, high) in groups.items():
    # Number of distinct queries in this group
    distinct_queries = 3  
    for i in range(distinct_queries):
        # Pick a random template
        template = random.choice(query_templates)
        
        # Sleep time (used only if the template calls for it)
        sleep_time = round(random.uniform(low, high), 2)
        
        # A random integer parameter for second query type or appended comment
        param = random.randint(1, 100)
        
        # Fill in the placeholders in the template
        query_text = template.format(
            sleep_time=sleep_time,
            group_name=group_name,
            param=param
        )
        
        # Run each distinct query multiple times
        repetitions = random.randint(5, 10)
        for _ in range(repetitions):
            cursor.execute(query_text)
        conn.commit()
        
        print(f"Executed '{query_text}' {repetitions} times.")

print("Synthetic query generation complete.")

# ------------------------------------------------------------
# 4. Wait a Moment for Stats to Update
# ------------------------------------------------------------
time.sleep(2)

# ------------------------------------------------------------
# 5. Retrieve Performance Data from pg_stat_statements
# ------------------------------------------------------------
query = """
SELECT query, calls, total_exec_time / calls AS mean_exec_time
FROM pg_stat_statements
WHERE calls > 0
ORDER BY calls DESC
"""

try:
    cursor.execute(query)
    rows = cursor.fetchall()
    print("Fetched query performance data from pg_stat_statements.")
except Exception as e:
    print("Error executing query:", e)
    cursor.close()
    conn.close()
    exit(1)

df = pd.DataFrame(rows, columns=['query', 'calls', 'mean_exec_time'])
print("\nSample data from pg_stat_statements:")
print(df.head(10))

if df.empty:
    print("No query performance data found. Exiting.")
    cursor.close()
    conn.close()
    exit(0)

# ------------------------------------------------------------
# 6. Anomaly Detection
# ------------------------------------------------------------
features = df[['calls', 'mean_exec_time']].copy()

if features.empty:
    print("No valid queries found for anomaly detection. Exiting.")
    cursor.close()
    conn.close()
    exit(0)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(features_scaled)

df['anomaly'] = model.predict(features_scaled)

# ------------------------------------------------------------
# 7. Identify & Display Anomalies
# ------------------------------------------------------------
anomalies = df[df['anomaly'] == -1]
print("\n=== Detected Anomalous Queries ===")
print(anomalies[['query', 'calls', 'mean_exec_time']])

# ------------------------------------------------------------
# 8. Visualize the Results
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(
    df['calls'],
    df['mean_exec_time'],
    c=df['anomaly'],
    cmap='coolwarm',
    alpha=0.7
)
plt.xlabel('Number of Calls')
plt.ylabel('Mean Execution Time (ms)')
plt.title('Anomaly Detection in PostgreSQL Query Performance')
plt.colorbar(label='Anomaly (1: Normal, -1: Anomaly)')
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
cursor.close()
conn.close()
print("PostgreSQL connection closed.")
