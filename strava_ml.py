import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("FINALdataset.csv")

features = ["Distance_km", "Elapsed Time", "Moving Time", "Elevation Gain", "Average Speed", "Calories", "Average Heart Rate"] # Independent variables
X = df[features]
y = df["Activity Type"]  # Dependent/Target variable

# Splitting the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict sport type using the model
df["Predicted Activity Type"] = model.predict(X)

# Print results
result = df[["Distance_km", "Elapsed Time", "Moving Time", "Elevation Gain", "Average Speed", "Calories", "Average Heart Rate", "Predicted Activity Type"]].copy()
result.insert(0, "Activity ID", range(1, len(result) + 1))  #Created an Activity ID column

# Print results
print(result.to_string(index=False))