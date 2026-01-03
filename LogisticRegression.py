import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_auc_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

features_data = pd.read_csv('Features.csv')
labels_data = pd.read_csv('Labels.csv')

features_data['clean_filename'] = features_data['Filename'].str.replace(r'\.(mat|hea)$', '', regex=True)
labels_data['clean_filename'] = labels_data['file'].str.extract(r'(JS\d+)')[0]

arrhythmia_codes = {
    '426177001': 0, '713427006': 0, '251146004': 0,

    '426783006': 1, '698252002': 1, '445211001': 1,
    '427084000': 1, '164889003': 1, '164890007': 1,
    '195080001': 1, '17338001': 1, '270492004': 1,
    '233917008': 1, '5375003': 1, '251166008': 1,
    '251170000': 1, '426648003': 1, '195060002': 1,
    '233916004': 1
}

unknown_codes = set(labels_data['code'].dropna().unique()) - set(arrhythmia_codes.keys())
if unknown_codes:
    print(f"Warning: {len(unknown_codes)} unknown rhythm codes found")
    print("Sample unknown codes:", list(unknown_codes)[:3])
    
    for code in unknown_codes:
        rhythm_desc = labels_data[labels_data['code']==code]['rhythm'].iloc[0].lower()
        arrhythmia_codes[code] = 1 if any(keyword in rhythm_desc 
                                       for keyword in ['tachy', 'fibrill', 'flutter', 'brady']) else 0
        print(f"Classified {code} ({rhythm_desc}) as {arrhythmia_codes[code]}")

labels_data['Arrhythmia'] = labels_data['code'].map(arrhythmia_codes)

data = pd.merge(
    features_data,
    labels_data[['clean_filename', 'Arrhythmia', 'rhythm', 'code']],
    on='clean_filename',
    how='inner'
)

print(f"\nFinal dataset size: {len(data)}")
print("Class distribution:")
print(data['Arrhythmia'].value_counts(normalize=True))

X = data.drop(['Filename', 'clean_filename', 'Arrhythmia', 'rhythm', 'code'], 
              axis=1, errors='ignore')
y = data['Arrhythmia']

X.fillna(X.median(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
logreg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    penalty='l2',
    solver='liblinear'
)
logreg.fit(X_train_scaled, y_train)

# Evaluate
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Arrhythmia'],
            yticklabels=['Normal', 'Arrhythmia'])
plt.title('Confusion Matrix')
plt.show()

# Feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0],
    'Absolute_Coefficient': np.abs(logreg.coef_[0])
}).sort_values('Absolute_Coefficient', ascending=False)

print("\nTop 10 Features:")
print(coefficients.head(10))