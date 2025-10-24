import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("strava_data.csv")

# Convert pace (mm:ss) to minutes
def pace_to_minutes(pace):
    try:
        m, s = map(float, pace.split(":"))
        return m + s / 60
    except:
        return None

df["avg_pace_min"] = df["avg_pace_per_km"].apply(pace_to_minutes)

# Convert elapsed time (hh:mm:ss) to minutes
def time_to_minutes(t):
    try:
        h, m, s = map(float, t.split(":"))
        return h * 60 + m + s / 60
    except:
        return None

df["elapsed_time_min"] = df["elapsed_time_hh:mm:ss"].apply(time_to_minutes)

df["steps"] = df["steps"].replace({",": ""}, regex=True).astype(float)

features = ["distance_km", "avg_pace_min", "elapsed_time_min", "steps"] # Independent variables
X = df[features]
y = df["sport_type"]  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict sport type using the model
df["predicted_sport_type"] = model.predict(X)

# Print results
result = df[["activity_id", "distance_km", "avg_pace_min", "elapsed_time_min", "steps", "predicted_sport_type"]]
print(result.to_string(index=False))