import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = pd.read_csv("FINALdataset.csv")

features = ["Distance_km", "Elapsed Time", "Moving Time", "Elevation Gain", "Average Speed", "Calories", "Average Heart Rate"]
X = df[features]  # Independent variables
y = df["Activity Type"]  # Dependent/Target variable

# Splitting the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict activity type
df["Predicted Activity Type"] = model.predict(X)

# Print results
result = df[["Distance_km", "Elapsed Time", "Moving Time", "Elevation Gain", "Average Speed", "Calories", "Average Heart Rate", "Predicted Activity Type"]].copy()
result.insert(0, "Activity ID", range(1, len(result) + 1))  #Created an Activity ID column
print(result.to_string(index=False))

# Predict on test set for evaluation
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred)) # Show precision, recall, f1-score per class

# Accuracy Chart
accuracy_per_class = cm.diagonal() / cm.sum(axis=1)  # Accuracy per class
class_names = model.classes_

overall_accuracy = accuracy_score(y_test, y_pred)  # Overall model accuracy
overall_accuracy_pct = overall_accuracy * 100  # Convert to percentage

plt.figure(figsize=(8, 6))
plt.plot(class_names, accuracy_per_class, marker='o', linestyle='-', color='green', linewidth=2, label='Per-Class Accuracy')
plt.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy ({overall_accuracy_pct:.2f}%)')

# Add value labels above points
for i, v in enumerate(accuracy_per_class):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
plt.title("Accuracy Chart")
plt.xlabel("Activity Type")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nOverall Model Accuracy: {overall_accuracy_pct:.2f}%")  # Print overall accuracy