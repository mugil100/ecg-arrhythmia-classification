# ECG Arrhythmia Detection & Classification

A machine learning pipeline for automated detection and classification of cardiac arrhythmias from 12-lead ECG signals — covering signal preprocessing, novel feature extraction, and a comparative study across five supervised models plus unsupervised clustering.

**Best result:** Random Forest — 94.13% accuracy across 9 fine-grained rhythm classes (the most directly comparable, hardest task in the study).

---

## Why This Project

Manual ECG interpretation is time-consuming and error-prone, especially at hospital scale. This project builds an end-to-end pipeline — from raw signal to classified rhythm — and systematically compares five classification algorithms to identify which approach best handles the class imbalance and subtlety inherent in real clinical ECG data.

This was built as a team project (4 contributors) and formalized into an IEEE-style research paper.

---

## Dataset

- **Source:** 12-lead ECG recordings from Shaoxing People's Hospital and Ningbo First Hospital of Zhejiang University (published via PhysioNet), collected 2013–2018
- **Scale:** 40,258 ECGs from 22,599 male and 17,659 female patients (majority aged 51–80)
- **Format:** 10-second recordings sampled at 500Hz, stored as `.mat` files
- **Class balance:** 20% normal sinus rhythm, 80% abnormal — covering rhythms including NSR, Atrial Fibrillation, Sinus Bradycardia, Sinus Tachycardia, Supraventricular Tachycardia, and Ventricular Fibrillation

---

## Pipeline

**1. Signal Preprocessing**
- Butterworth low-pass filtering (noise above 50Hz)
- LOESS smoothing for baseline wander removal
- Baseline correction (zero-centering)
- Non-Local Means denoising — chosen specifically to preserve P-wave/QRS/T-wave morphology while removing noise, avoiding the manual threshold tuning that wavelet-based denoising requires

**2. Feature Extraction**
- Temporal features: RR, PR, QT intervals, heart rate variability (HRV)
- Morphological features: QRS duration, P/T-wave characteristics, peak/valley statistics
- Explored 11 feature-set variants, from an 11-feature baseline up to a 39,830-feature comprehensive set
- Fixed-length feature vectors built via empirical frequency distribution (100-bin histograms) to handle variable peak/valley counts per patient

**3. Classification — Model Comparison**

Note: the five models were evaluated on three different task framings (a limitation worth being upfront about), so results aren't strictly apples-to-apples — each is reported against the task it was actually trained on.

**9-class fine-grained rhythm classification** (AFib, AFlutter, 1st-degree AV block, NSR, SB, Sinus Tachycardia, SVT, VFib, VTach) — test set of 2,407–3,611 samples:

| Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---|---|---|
| **Random Forest** | **94.13%** | 0.70 | 0.93 | `n_estimators=150, max_depth=15, class_weight='balanced'` |
| Decision Tree | 91% | 0.67 | 0.91 | Strong on NSR/SB, weak on 1AVB, SVT |
| KNN | 75% | 0.61 | 0.76 | k=1 (cross-validated); SMOTE-balanced training set |

Across all three, dominant classes (NSR, SB) classify with high precision/recall (~0.9+), while rare classes (SVT, 1st-degree AV block, VFib) show much weaker recall — e.g., Random Forest's SVT recall was 0.64, KNN's SVT precision only 0.17. This is a direct, expected consequence of severe class imbalance and is flagged rather than hidden.

**Binary classification (Normal vs. Arrhythmia)** — separate derived dataset (12,034 samples):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 96.8% | 98.1% | 96.1% | 97.1% | 99.3% |

**4-class grouped "clinical" classification** (Other Bradycardia, Other Tachycardia, Unclassified Arrhythmia, Unclassified Rhythm):

| Model | Accuracy | Macro Precision | Macro Recall |
|---|---|---|---|
| SVM (RBF kernel, GridSearchCV-tuned) | 91.4% | 88.7% | 88.2% |

- Class imbalance addressed via SMOTE (KNN training set) and `class_weight='balanced'` (Random Forest, Logistic Regression)
- Hyperparameters tuned via GridSearchCV / cross-validation across all models
- **K-Means clustering** (unsupervised) validated k=3 as optimal via Elbow Method + silhouette score, used for exploratory pattern discovery independent of labels

---

## Tech Stack

Python · scikit-learn (RandomForest, SVM, KNN, Logistic Regression, K-Means) · pandas/NumPy · SelectKBest feature selection · imbalanced-learn (SMOTE) · Matplotlib/Seaborn (confusion matrices)

---

## Repository Structure

```
├── DecisionTreeClassifier.py
├── RandomForestClassifier.py
├── LogisticRegression.py
├── KNN.py
├── SVM.py
├── K-Means.py
├── featureextractSample-checkpoint.ipynb   # Feature engineering pipeline
├── clean3-checkpoint.ipynb                 # Data cleaning
├── clean5-checkpoint.ipynb                 # Data cleaning
├── Features.csv / cleancsvFeatures.csv     # Extracted feature sets
├── Labels.xlsx
└── rawcsv.csv
```

---

## Running the Models

Each classifier script is self-contained. Example:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python RandomForestClassifier.py
```

Scripts expect `Features.csv` and `Labels.csv` in the same directory, merged on a shared filename key, and will print accuracy/precision/recall/F1/specificity plus a confusion matrix.

---

## Paper

This work was formalized as an IEEE-style paper: *"Automated Detection and Classification of Arrhythmias in ECG Signals"* — co-authored by Gowri Shankar A B, Dinesh Karthik V, Duvvuru Akshaya Saketh Reddy, and Mugilan S S (Amrita Vishwa Vidyapeetham, Coimbatore).

---

## About

Built by Mugilan S S as part of a team research project during B.Tech CCE at Amrita Vishwa Vidyapeetham.
