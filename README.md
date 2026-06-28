# 🏥 NHANES Age Group Prediction — Summer Analytics 2026
**Hosted by:** Consulting & Analytics Club, IIT Guwahati  
**Dataset:** CDC National Health and Nutrition Examination Survey (NHANES)  
**Task:** Binary classification — predict if a person is a Senior (65+) or Adult (<65)

---

## 📌 Problem Statement
Using a subset of the CDC NHANES dataset (1,966 training rows, 312 test rows), the goal is to classify respondents as **Adult (0)** or **Senior (1)** based on health, lifestyle, and lab features.

---

## 📂 Repository Structure
```
├── Prediction_Model.ipynb   # Full pipeline notebook
├── train.csv                # Training data
├── test.csv                 # Test data
├── sample_submission.csv    # Submission format
├── submission.csv           # Final predictions
└── README.md
```

---

## 🔬 Dataset Features
| Feature | Description |
|---|---|
| SEQN | Sequence ID |
| RIAGENDR | Gender (1=Male, 2=Female) |
| PAQ605 | Physical activity response |
| BMXBMI | Body Mass Index |
| LBXGLU | Fasting glucose level |
| DIQ010 | Diabetes questionnaire (1=Yes, 2=No, 3=Borderline) |
| LBXGLT | Oral glucose tolerance |
| LBXIN | Insulin level |

---

## ⚙️ Pipeline Overview

### Phase 1 — Setup & Data Loading
- Loaded train, test, and sample submission files
- Inspected shapes, dtypes, and first rows

### Phase 2 — Exploratory Data Analysis (EDA)
- Identified **83.9% Adult vs 16.1% Senior** class imbalance
- Detected missing values across all features
- Found `LBXGLT` and `LBXGLU` as strongest Senior indicators via box plots and correlation analysis

### Phase 3 — Preprocessing
- Dropped SEQN (identifier, not a feature)
- Mode imputation for categorical features (`RIAGENDR`, `PAQ605`, `DIQ010`)
- Median imputation for skewed continuous features (`BMXBMI`, `LBXGLU`, `LBXGLT`, `LBXIN`)
- IQR outlier capping on continuous columns only
- Retained `DIQ010=3` (Borderline diabetes) as clinically meaningful
- Applied `get_dummies` on BMI categories for linear model compatibility

### Phase 4 — Feature Engineering
Created 10 new features from domain knowledge:
| Feature | Description |
|---|---|
| `GLU_INS_RATIO` | Glucose to insulin ratio |
| `GLT_GLU_RATIO` | Glucose tolerance to fasting glucose ratio |
| `IS_DIABETIC` | Confirmed diabetes flag |
| `IS_BORDERLINE` | Borderline diabetes flag |
| `IS_ACTIVE` | Physical activity flag |
| `BMI_CAT` | Ordinal BMI category |
| `HIGH_GLU` | Fasting glucose > 100 flag |
| `HIGH_GLT` | Glucose tolerance > 140 flag |
| `METABOLIC_RISK` | Combined metabolic risk score |
| `INACTIVE_OBESE` | Inactive + obese interaction flag |

### Phase 5 — Class Imbalance Handling
- Visualized imbalance before and after
- Applied **SMOTE** on training data only → balanced to 1,638 vs 1,638

### Phase 6 — Model Comparison
Compared 5 models via 5-Fold Stratified Cross-Validation:
| Model | F1 Weighted | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.796 | 0.891 |
| Decision Tree | 0.817 | 0.817 |
| Gradient Boosting | 0.853 | 0.931 |
| XGBoost | 0.877 | 0.950 |
| **Random Forest** | **0.891** | **0.957** |

### Phase 7 — Hyperparameter Tuning
- Tuned both **Random Forest** and **XGBoost** using `RandomizedSearchCV` (50 iterations, 5-Fold CV)
- Both tuned on SMOTE-balanced full training data
- Random Forest selected as final model based on highest honest CV scores

### Phase 8 — Final Model & Evaluation
- Final model: **Random Forest (Tuned)**
- ROC-AUC: **0.957** | F1 Score: **0.891**
- Feature importance plotted
- ROC curve generated using cross-validated predictions

### Phase 9 — Submission
- Predicted on 312 test rows
- Distribution: 86.5% Adult, 13.5% Senior (consistent with real-world distribution)
- Exported `submission.csv` with `age_group` column (0 or 1)

---

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- scikit-learn
- xgboost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## 📊 Key Findings
- Glucose tolerance (`LBXGLT`) was the single strongest predictor of senior status
- Engineered metabolic risk features significantly improved model signal
- SMOTE effectively addressed the 5:1 class imbalance
- Random Forest outperformed XGBoost on both F1 and ROC-AUC after fair comparison

---

## 👤 Author
**Vincent** — Student, Ghana  
Summer Analytics 2026 | IIT Guwahati x CDC NHANES Hackathon
