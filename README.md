# Heart Disease Risk Stratification — End to End ML Project

A full end-to-end machine learning project that predicts cardiovascular disease risk using the UCI Heart Disease dataset. Built from scratch covering every major ML concept — from raw data to a deployed web application.

**GitHub:** https://github.com/pulkit96177/heart-risk-ml

---

## Final Results

| Metric | Score |
|---|---|
| Best Model | Logistic Regression |
| F1 Score | 0.830 |
| ROC-AUC | 0.913 |
| Recall (disease class) | 89% |
| Accuracy | 83% |

---

## Project Phases

### Phase 1 — EDA & Feature Engineering
- Loaded and inspected UCI Heart Disease dataset (920 rows, 16 columns)
- Handled missing values — dropped high-missing columns (ca: 66%, thal: 53%)
- Fixed cholesterol zeros (clinically impossible), removed bad rows
- Converted target `num` from 5 classes to binary (0 = healthy, 1 = disease)
- Key finding: `thalch` (-0.38) and `oldpeak` (+0.37) are strongest predictors

### Phase 2 — Unsupervised Learning
- One hot encoded categorical features (13 → 23 columns)
- Train test split (80/20, stratified) before any scaling
- StandardScaler fit on training data only
- K-Means clustering with K=3 (elbow + silhouette score)
- Discovered 3 natural patient profiles without using labels:
  - Cluster 1: Younger lower risk — 34% disease rate
  - Cluster 0: Middle aged average risk — 42% disease rate
  - Cluster 2: Older higher risk — **82% disease rate**
- DBSCAN for outlier detection
- PCA for 2D cluster visualization
- Cluster label added as new feature for Phase 3

### Phase 3 — Supervised Learning
- sklearn Pipeline with SMOTE + StandardScaler + Model
- 7 algorithms trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes
- Stratified 5-Fold Cross Validation on all models
- Metrics: F1, ROC-AUC, Precision, Recall
- RandomizedSearchCV tuning on top 3 models
- Best baseline: Logistic Regression (F1: 0.829, ROC-AUC: 0.883)

### Phase 4 — Ensemble Learning & Optuna
- Voting Classifier (soft voting — LR + GB + SVM): F1 0.828
- Stacking Classifier (meta-learner = Logistic Regression): F1 0.823
- Optuna Bayesian optimization on Gradient Boosting (30 trials): F1 0.830
- Final winner: Logistic Regression (F1: 0.830, ROC-AUC: 0.913)

### Phase 5 — Model Explainability (SHAP)
- PermutationExplainer on full pipeline
- Summary bar plot — global feature importance
- Beeswarm plot — direction and magnitude per patient
- Waterfall plot — individual patient explanation
- Force plot — push/pull visualization per patient
- Top features: `cp_asymptomatic`, `oldpeak`, `restecg_normal`, `thalch`
- Cluster label from Phase 2 confirmed as contributing feature

### Phase 6 — Full Stack Deployment
- FastAPI backend with 2 endpoints: `/health` and `/predict`
- Pydantic input validation
- Preprocessing pipeline: binary encoding → one hot encoding → scaling → KMeans cluster assignment → prediction
- Models loaded once at startup via lifespan event
- Plain HTML/CSS/JS frontend — no frameworks
- Color coded results — red high risk, green low risk
- Probability bar animation, cluster profile display, medical disclaimer

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML | scikit-learn, xgboost |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | scipy, optuna |
| Explainability | shap |
| Deployment | fastapi, uvicorn, joblib |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
heart-risk-ml/
├── data/
│   ├── raw/                        ← original dataset (never modified)
│   └── processed/                  ← cleaned, encoded, split data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_unsupervised.ipynb
│   ├── 03_supervised.ipynb
│   ├── 04_ensemble_tuning.ipynb
│   └── 05_explainability.ipynb
├── models/
│   ├── lr_best_model.pkl           ← final pipeline (SMOTE + Scaler + LR)
│   ├── gb_optuna_model.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
├── api/
│   └── main.py                     ← FastAPI backend
├── frontend/
│   └── index.html                  ← single page web UI
├── requirements.txt
├── HOW_TO_RUN.md
└── README.md
```

---

## Quick Start

```bash
# clone
git clone https://github.com/pulkit96177/heart-risk-ml.git
cd heart-risk-ml

# setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# run API
cd api
uvicorn main:app --reload

# open frontend/index.html in browser
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed setup instructions.

---

## Concepts Covered

- Supervised Learning (7 algorithms)
- Unsupervised Learning (K-Means, DBSCAN, PCA)
- Ensemble Learning (Bagging, Boosting, Voting, Stacking)
- Cross Validation (Stratified K-Fold)
- Hyperparameter Tuning (RandomizedSearchCV, Optuna)
- Class Imbalance (SMOTE)
- sklearn Pipelines (no data leakage)
- Model Explainability (SHAP)
- REST API (FastAPI)
- Frontend (HTML/CSS/JS)
