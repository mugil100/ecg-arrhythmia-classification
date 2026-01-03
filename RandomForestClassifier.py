import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_specificity(cm):
    specificity = []
    for i in range(cm.shape[0]):
        true_neg = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_pos = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(true_neg / (true_neg + false_pos))
    return specificity

def load_and_validate_data(features_path, labels_path):
    features_df = pd.read_csv(features_path, encoding_errors='replace')
    labels_df = pd.read_csv(labels_path, encoding_errors='replace')
    
    print(f"\nLoaded data shapes - Features: {features_df.shape}, Labels: {labels_df.shape}")
    
    if 'Filename' not in features_df.columns:
        raise ValueError("Required column 'Filename' missing in features data")
    if 'file' not in labels_df.columns:
        raise ValueError("Required column 'file' missing in labels data")
    
    merged_df = pd.merge(features_df, labels_df, 
                        left_on='Filename', 
                        right_on='file',
                        how='inner')
    
    if len(merged_df) == 0:
        print("\nData merging issue detected:")
        print("Sample Filenames:", features_df['Filename'].head().tolist())
        print("Sample file IDs:", labels_df['file'].head().tolist())
        raise ValueError("No matching records found - check your merge columns")
    
    return merged_df

try:
    data = load_and_validate_data('Features.csv', 'Labels.csv')
    print(f"\nProcessing {len(data)} ECG records...")
    X = data.drop(['file', 'Filename', 'code', 'rhythm'], axis=1, errors='ignore')  
    y = data['rhythm']
    
    # Handle categorical data
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"\nEncoding categorical columns: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Feature selection
    if X.shape[0] > 1 and X.shape[1] > 1:
        k = min(100, X.shape[1]//2)
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        print(f"\nSelected top {k} most important features")
    else:
        X_selected = X.values
        print("\nUsing all available features")
    
    # Split data (stratify by 'rhythm')
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model performance:")
    y_pred = rf.predict(X_test)
    class_names = sorted(y.unique().astype(str))  # Get unique rhythm names
    
    # Confusion matrix visualization (using rhythm names)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, 
               yticklabels=class_names)
    plt.title('ECG Rhythm Classification Confusion Matrix')
    plt.ylabel('Actual Rhythm')
    plt.xlabel('Predicted Rhythm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = calculate_specificity(cm)
    
    # Print results
    print("\n" + "="*50)
    print("ECG Rhythm Classification Performance Metrics")
    print("="*50)
    print(f"{'Overall Accuracy:':<25} {accuracy:.4f}")
    print(f"{'Sensitivity (Recall):':<25} {recall:.4f}")
    print(f"{'Average Specificity:':<25} {np.mean(specificity):.4f}")
    print(f"{'F1-Score:':<25} {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nClass-Specific Specificity:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<30} {specificity[i]:.4f}")
    
except Exception as e:
    print(f"\nError encountered: {str(e)}")
    print("\nTroubleshooting suggestions:")
    print("- Verify both data files exist in the specified paths")
    print("- Check column names match exactly (case-sensitive)")
    print("- Ensure no missing values in key columns")
    print("- Confirm data types are consistent in merge columns")