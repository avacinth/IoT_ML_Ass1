import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("strava_data.csv")

# Converting (mm:ss) to minutes
def pace_to_minutes(pace):
    try:
        m, s = map(float, pace.split(":"))
        return m + s / 60
    except:
        return None

df["avg_pace_min"] = df["avg_pace_per_km"].apply(pace_to_minutes)

# Defining the activity type (gawagawa)
def classify_activity(row):
    if row["distance_km"] < 0.5 and row["steps"] < 300:
        return "Sitting"
    elif row["distance_km"] < 3 and row["avg_pace_min"] >= 2:
        return "Walk"
    elif row["distance_km"] >= 8 and row["avg_pace_min"] < 10:
        return "Ride"
    elif row["distance_km"] >= 3 and row["avg_pace_min"] < 5:
        return "Run"
    else:
        return "Unknown"

df["activity_type"] = df.apply(classify_activity, axis=1)

# Printing the given data and the results (activity_type)
features = ["distance_km", "avg_pace_min", "elapsed_time_min", "steps"]
X = df[features]
y = df["activity_type"]

# Splitting the data per column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the activity type for all data
df["predicted_activity"] = model.predict(X)

# Printing the predicted activity
result = df[["activity_id", "distance_km", "avg_pace_min", "steps", "predicted_activity"]]
print(result.to_string(index=False))