import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1.  Load dataset
df_train = pd.read_csv("FINALpublicdataset.csv")    # Training Data
df_test = pd.read_csv("FINALpersonaldataset.csv")   # Testing Data

# 2. Defining features
features = ["Distance_km", "Elapsed Time", "Moving Time", "Elevation Gain", "Average Speed"]

X_train = df_train[features]         # Independent variables
y_train = df_train["Activity Type"]  # Dependent/Target variable

X_test = df_test[features]
y_test = df_test["Activity Type"]

# 3.  Training vs Testing data distribution graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training Data
sns.countplot(y=df_train["Activity Type"], ax=axes[0], order=df_train["Activity Type"].value_counts().index, palette="Blues")
axes[0].set_title("TRAINING Data\n(FINALpublicdataset.csv)")
axes[0].set_xlabel("Count")
axes[0].grid(axis='x', alpha=0.3)

# Testing Data
sns.countplot(y=df_test["Activity Type"], ax=axes[1], order=df_test["Activity Type"].value_counts().index, palette="Oranges")
axes[1].set_title("TESTING Data\n(FINALpersonaldataset.csv)")
axes[1].set_xlabel("Count")
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# 4.  Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5.  Predict activity type on the TEST dataset
df_test["Predicted Activity Type"] = model.predict(X_test)

# Print results
result = df_test[["Activity ID"] + features + ["Predicted Activity Type"]].copy()
print("Prediction Results on the TEST dataset (First 10 rows)")
print(result.head(10).to_string(index=False))

# 6.  Accuracy & Metrics
y_pred = df_test["Predicted Activity Type"]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, labels=model.classes_, zero_division=0))

# Accuracy Chart

# Overall Accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
overall_accuracy_pct = overall_accuracy * 100

# Per-Class Accuracy
row_sums = cm.sum(axis=1)
accuracy_per_class = np.divide(cm.diagonal(), row_sums, out=np.zeros_like(cm.diagonal(), dtype=float), where=row_sums != 0)
class_names = model.classes_

plt.figure(figsize=(8, 6))
plt.plot(class_names, accuracy_per_class, marker='o', linestyle='-', color='green', linewidth=2, label='Per-Class Accuracy')
plt.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Accuracy ({overall_accuracy_pct:.2f}%)')

# Add text labels above the points
for i, v in enumerate(accuracy_per_class):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
plt.title("Accuracy Chart per Activity Type")
plt.xlabel("Activity Type")
plt.ylabel("Accuracy (0.0 - 1.0)")
plt.ylim(0, 1.15)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nOverall Model Accuracy: {overall_accuracy_pct:.2f}%")