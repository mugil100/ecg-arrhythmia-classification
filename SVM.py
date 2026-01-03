import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           balanced_accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def get_rhythm_mapping():
    """Returns a clinically validated rhythm classification mapping"""
    return {
        # Normal Rhythms
        '426177001': 'Normal Sinus Rhythm',
        '713427006': 'Normal ECG',
        '251146004': 'Sinus Rhythm',
        
        # Bradyarrhythmias
        '426783006': 'Sinus Bradycardia',
        '698252002': 'Bradycardia',
        '445211001': 'Sinus Node Dysfunction',
        
        # Tachyarrhythmias
        '427084000': 'Sinus Tachycardia',
        '164889003': 'Atrial Fibrillation',
        '164890007': 'Atrial Flutter',
        '195080001': 'Ventricular Tachycardia',
        '17338001': 'Ventricular Fibrillation',
        
        # Conduction Disorders
        '270492004': 'Heart Block',
        '233917008': 'AV Block',
        '5375003': 'WPW Syndrome',
        
        # Default Categories
        'UNKNOWN_BRADY': 'Other Bradycardia',
        'UNKNOWN_TACHY': 'Other Tachycardia',
        'UNKNOWN': 'Unclassified Rhythm'
    }

def prepare_data(features_path, labels_path):
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    
    features['clean_filename'] = features['Filename'].str.extract(r'(JS\d+)')[0]
    labels['clean_filename'] = labels['file'].str.extract(r'(JS\d+)')[0]

    rhythm_map = get_rhythm_mapping()
    labels['Rhythm_Type'] = labels['code'].map(rhythm_map)
    
    unclassified = labels['Rhythm_Type'].isna()
    labels.loc[unclassified, 'Rhythm_Type'] = labels.loc[unclassified, 'rhythm'].apply(
        lambda x: classify_unknown_rhythm(x, rhythm_map))
    data = pd.merge(features, labels[['clean_filename', 'Rhythm_Type', 'code', 'rhythm']], 
                   on='clean_filename', how='inner')
    
    return data, rhythm_map

def classify_unknown_rhythm(rhythm_desc, rhythm_map):
    desc = str(rhythm_desc).lower()
    if 'brady' in desc:
        return rhythm_map['UNKNOWN_BRADY']
    elif any(kw in desc for kw in ['tachy', 'fast', 'accelerated']):
        return rhythm_map['UNKNOWN_TACHY']
    elif any(kw in desc for kw in ['fibrill', 'flutter', 'block']):
        return 'Unclassified Arrhythmia'
    return rhythm_map['UNKNOWN']

def train_clinical_svm(X_train, y_train, rhythm_types):
    clinical_features = [
        col for col in X_train.columns 
        if any(lead in col for lead in ['Lead_1', 'Lead_2', 'Lead_5'])
        and any(metric in col for metric in ['Rate', 'HRV', 'Interval'])
    ]
    X_train = X_train[clinical_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale'],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    
    svm = GridSearchCV(
        SVC(probability=True, decision_function_shape='ovo'),
        param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    svm.fit(X_train_scaled, y_train)
    
    return svm, scaler, clinical_features

def clinical_evaluation(model, scaler, X_test, y_test, rhythm_types, feature_names):
    X_test_scaled = scaler.transform(X_test[feature_names])
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    print("\n=== Clinical Classification Report ===")
    print(classification_report(
        y_test, y_pred, 
        target_names=rhythm_types,
        digits=3
    ))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=rhythm_types
    )
    disp.plot(
        cmap='Blues',
        xticks_rotation=45,
        values_format='d',
        colorbar=False
    )
    plt.title('Arrhythmia Type Confusion Matrix')
    plt.tight_layout()
    plt.show()

    confidence = y_prob.max(axis=1)
    high_conf = np.mean(confidence > 0.8)
    medium_conf = np.mean((confidence >= 0.6) & (confidence <= 0.8))
    low_conf = np.mean(confidence < 0.6)
    
    print(f"\nPrediction Confidence:")
    print(f"High (>80%): {high_conf:.1%}")
    print(f"Medium (60-80%): {medium_conf:.1%}")
    print(f"Low (<60%): {low_conf:.1%}") 

def main():
    FEATURES_PATH = "Features.csv"
    LABELS_PATH = "Labels.csv"
    
    print("Loading and preparing clinical ECG data...")
    data, rhythm_map = prepare_data(FEATURES_PATH, LABELS_PATH)
    
    rhythm_types = sorted(data['Rhythm_Type'].unique())
    print("\nDetected Arrhythmia Types:")
    for i, rhythm in enumerate(rhythm_types):
        print(f"{i}: {rhythm}")
    
    X = data.drop(['Filename', 'clean_filename', 'Rhythm_Type', 'code', 'rhythm'], axis=1)
    y = data['Rhythm_Type'].astype('category').cat.codes  # Convert to numeric codes
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining clinical SVM classifier...")
    svm_model, scaler, feature_names = train_clinical_svm(X_train, y_train, rhythm_types)
    print(f"Best SVM parameters: {svm_model.best_params_}")
    
    clinical_evaluation(svm_model, scaler, X_test, y_test, rhythm_types, feature_names)
    
    print("\nSaving clinical model...")
    model_package = {
        'model': svm_model,
        'scaler': scaler,
        'features': feature_names,
        'rhythm_types': rhythm_types,
        'rhythm_map': rhythm_map
    }
    joblib.dump(model_package, 'clinical_arrhythmia_classifier.pkl')

if __name__ == "__main__":
    main()