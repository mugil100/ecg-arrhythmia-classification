import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load datasets
data_df = pd.read_csv(r"F:\ML_Code\Features.csv")
labels_df = pd.read_excel(r"F:\ML_Code\Labels.csv")

# Clean filenames
data_df["Filename"] = data_df["Filename"].str.replace(".mat", "", regex=False)
labels_df["file"] = labels_df["file"].str.replace(".hea", "", regex=False)

# Merge datasets
merged_df = data_df.merge(labels_df, left_on="Filename", right_on="file")
merged_df.drop(columns=["Filename", "file", "age", "sex", "code"], inplace=True)

# Handle missing values (fill with median)
merged_df.fillna(merged_df.median(numeric_only=True), inplace=True)

# Separate features and target variable
X = merged_df.drop(columns=["rhythm"])
y = merged_df["rhythm"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning for KNN
param_grid = {"n_neighbors": range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_resampled, y_train_resampled)
best_k = grid_search.best_params_["n_neighbors"]

# Train optimized KNN model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Best k: {best_k}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", class_report)