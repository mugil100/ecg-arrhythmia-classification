import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
data_file = r"D:\Features.csv"
labels_file = r"D:\Labels.csv"

# Load the data
data_df = pd.read_csv(data_file)
labels_df = pd.read_csv(labels_file)

# Clean filenames for merging
data_df["Filename"] = data_df["Filename"].str.replace(".mat", "", regex=False)
labels_df["file"] = labels_df["file"].str.replace(".hea", "", regex=False)

# Merge datasets
merged_df = data_df.merge(labels_df, left_on="Filename", right_on="file").drop(columns=["Filename", "file", "age", "sex", "code"])

# Separate features and labels
X = merged_df.drop(columns=["rhythm"])
y = merged_df["rhythm"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Predictions
y_pred_dt = dt.predict(X_test_scaled)

# Evaluate model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy_dt:.2f}")
print("Classification Report:\n", class_report_dt)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Decision Tree Confusion Matrix")
plt.show()