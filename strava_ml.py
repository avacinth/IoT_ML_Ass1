import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("strava_data_1.csv")

df["Activity ID"] = range(1, len(df) + 1)

df["Total Steps"] = df["Total Steps"].replace({",": ""}, regex=True).astype(float)

features = ["Distance", "Average Speed", "Elapsed Time", "Elevation Gain", "Total Steps"] # Independent variables
X = df[features]
y = df["Activity Type"]  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict sport type using the model
df["Predicted Activity Type"] = model.predict(X)

# Print results
result = df[["Activity ID", "Distance", "Average Speed", "Elapsed Time", "Elevation Gain", "Total Steps", "Predicted Activity Type"]]
print(result.to_string(index=False))