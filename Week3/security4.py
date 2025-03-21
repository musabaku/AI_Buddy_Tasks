import random
import pandas as pd
import numpy as np
from faker import Faker
import psycopg2
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

##############################
# PART 1: Synthetic Data Generation & Insertion into PostgreSQL
##############################

# Initialize Faker for synthetic data generation
fake = Faker()

# Parameters for synthetic data
num_entries = 1000  # number of synthetic login records
devices = ['desktop', 'mobile', 'tablet']
locations = ['New York', 'London', 'Tokyo', 'Sydney', 'Mumbai']

data = []
for _ in range(num_entries):
    user_id = random.randint(1, 100)  # simulate 100 different users
    login_time = fake.date_time_between(start_date="-30d", end_date="now")
    device = random.choice(devices)
    location = random.choice(locations)
    
    # Assign multi-class risk levels:
    # ~5% High risk (2), ~10% Medium risk (1), ~85% Low risk (0)
    r = random.random()
    if r < 0.05:
        risk_label = 2  # high risk
    elif r < 0.15:
        risk_label = 1  # medium risk
    else:
        risk_label = 0  # low risk

    data.append({
        'user_id': user_id,
        'login_time': login_time,
        'device': device,
        'location': location,
        'risk_label': risk_label
    })

# Create a DataFrame
df = pd.DataFrame(data)
print("Synthetic Data Sample:")
print(df.head())

# PostgreSQL connection parameters – update with your settings
conn = psycopg2.connect(
    dbname="AI2", 
    user="postgres", 
    password="aak101010", 
    host="localhost", 
    port="5432"
)
cur = conn.cursor()

# Create the table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS user_logins (
    user_id INTEGER,
    login_time TIMESTAMP,
    device VARCHAR(50),
    location VARCHAR(100),
    risk_label INTEGER
);
"""
cur.execute(create_table_query)
conn.commit()

# Insert synthetic data into the table
insert_query = """
INSERT INTO user_logins (user_id, login_time, device, location, risk_label)
VALUES (%s, %s, %s, %s, %s);
"""
records = list(df.itertuples(index=False, name=None))
cur.executemany(insert_query, records)
conn.commit()

cur.close()
conn.close()

print("Synthetic data inserted into PostgreSQL.")

##############################
# PART 2: Model Training & Evaluation
##############################

# Connect to PostgreSQL and extract the synthetic data
conn = psycopg2.connect(
    dbname="AI2", 
    user="postgres", 
    password="aak101010", 
    host="localhost", 
    port="5432"
)
query = "SELECT user_id, login_time, device, location, risk_label FROM user_logins;"
df = pd.read_sql_query(query, conn)
conn.close()

# Convert login_time to datetime and extract the hour
df['login_time'] = pd.to_datetime(df['login_time'])
df['login_hour'] = df['login_time'].dt.hour

# --- Feature Engineering ---
# Create cyclical features for login hour (to capture its periodic nature)
df['login_hour_sin'] = np.sin(2 * np.pi * df['login_hour'] / 24)
df['login_hour_cos'] = np.cos(2 * np.pi * df['login_hour'] / 24)

# Select features for the model: cyclical features + categorical features
features = df[['login_hour_sin', 'login_hour_cos', 'device', 'location']]
target = df['risk_label']

# One-hot encode categorical features: device and location
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(features[['device', 'location']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['device', 'location']))

# Combine cyclical and encoded features
features_final = pd.concat([features[['login_hour_sin', 'login_hour_cos']].reset_index(drop=True), encoded_df], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, target, test_size=0.3, random_state=42)

# Train a Random Forest model with balanced class weights (for multi-class classification)
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report on PostgreSQL Data:")
print(classification_report(y_test, y_pred))

##############################
# PART 3: Simulated New Login Events for Risk Scoring (Hardcoded)
##############################

# Let's simulate 30 new login events
simulated_logins = []
base_time = datetime(2025, 3, 21, 14, 0, 0)
for i in range(30):
    # Create a time offset for diversity
    login_time = base_time + timedelta(minutes=random.randint(0, 180))
    # Randomly pick device and location
    device = random.choice(devices)
    location = random.choice(locations)
    simulated_logins.append({
        "login_time": login_time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "location": location
    })

print("\nSimulated Login Events:")
print(pd.DataFrame(simulated_logins).head(10))  # show a sample of 10

# Process each simulated event and predict risk
results = []
for event in simulated_logins:
    # Process the login time
    new_login_dt = datetime.strptime(event["login_time"], "%Y-%m-%d %H:%M:%S")
    new_hour = new_login_dt.hour
    new_hour_sin = np.sin(2 * np.pi * new_hour / 24)
    new_hour_cos = np.cos(2 * np.pi * new_hour / 24)
    
    # Build a new DataFrame row for the event
    new_features = {
        "login_hour_sin": new_hour_sin,
        "login_hour_cos": new_hour_cos,
        "device": event["device"],
        "location": event["location"]
    }
    new_df = pd.DataFrame([new_features])
    
    # One-hot encode the categorical features using the existing encoder
    new_encoded = encoder.transform(new_df[['device', 'location']])
    new_encoded_df = pd.DataFrame(new_encoded, columns=encoder.get_feature_names_out(['device', 'location']))
    new_features_final = pd.concat([new_df[['login_hour_sin', 'login_hour_cos']], new_encoded_df], axis=1)
    
    # Predict risk label and probability distribution
    predicted_label = clf.predict(new_features_final)[0]
    predicted_prob = clf.predict_proba(new_features_final)[0]  # probabilities for each class
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    
    results.append({
        "login_time": event["login_time"],
        "device": event["device"],
        "location": event["location"],
        "predicted_label": predicted_label,
        "risk_level": risk_mapping[predicted_label],
        "prob_low": predicted_prob[0],
        "prob_med": predicted_prob[1],
        "prob_high": predicted_prob[2]
    })

results_df = pd.DataFrame(results)
print("\nSimulated New Login Risk Scoring for 30 Events:")
print(results_df)
