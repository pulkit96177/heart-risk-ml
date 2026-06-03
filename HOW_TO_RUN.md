# How to Run — Heart Disease Risk Stratification

## Prerequisites
- Python 3.10 or higher
- Git
- A terminal (PowerShell on Windows, Terminal on Mac/Linux)

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/pulkit96177/heart-risk-ml.git
cd heart-risk-ml
```

---

## Step 2 — Create & Activate Virtual Environment

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run the Notebooks (Optional)

Open VS Code and select `venv` as the Python interpreter.

Run notebooks in order:
```
notebooks/01_eda.ipynb
notebooks/02_unsupervised.ipynb
notebooks/03_supervised.ipynb
notebooks/04_ensemble_tuning.ipynb
notebooks/05_explainability.ipynb
```

> Note: Pre-trained models are already saved in the `models/` folder.
> You can skip directly to Step 5 without running the notebooks.

---

## Step 5 — Start the FastAPI Backend

```bash
cd api
uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open.

---

## Step 6 — Open the Frontend

Open a new terminal window. Navigate to the `frontend/` folder and open `index.html` in your browser.

**Windows:**
```powershell
start frontend/index.html
```

**Mac:**
```bash
open frontend/index.html
```

**Or** simply double-click `frontend/index.html` in your file explorer.

---

## Step 7 — Make a Prediction

Fill in the patient details in the form and click **Run Prediction**.

### Example Patient (High Risk)
| Field | Value |
|---|---|
| Age | 55 |
| Sex | Male |
| Resting BP | 130 |
| Cholesterol | 250 |
| Fasting Blood Sugar > 120 | No |
| Max Heart Rate | 120 |
| Exercise Induced Angina | Yes |
| ST Depression | 2.5 |
| Chest Pain Type | Asymptomatic |
| Resting ECG | Normal |
| Slope | Flat |
| Hospital | Cleveland |

Expected output: **High Risk — ~94% disease probability**

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if API is running |
| `/predict` | POST | Submit patient data, get prediction |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

---

## Project Structure

```
heart-risk-ml/
├── data/
│   ├── raw/                    ← original dataset
│   └── processed/              ← cleaned and split data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_unsupervised.ipynb
│   ├── 03_supervised.ipynb
│   ├── 04_ensemble_tuning.ipynb
│   └── 05_explainability.ipynb
├── models/
│   ├── lr_best_model.pkl       ← final trained pipeline
│   ├── gb_optuna_model.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
├── api/
│   └── main.py                 ← FastAPI backend
├── frontend/
│   └── index.html              ← web UI
├── requirements.txt
└── HOW_TO_RUN.md
```

---

## Troubleshooting

**`ModuleNotFoundError`** — virtual environment not activated. Run the activate command from Step 2.

**`Authentication failed` on git push** — use a GitHub Personal Access Token instead of your password. Generate one at GitHub → Settings → Developer Settings → Tokens (classic).

**Frontend shows `Error: Failed to fetch`** — make sure the FastAPI server is running (Step 5) before opening the frontend.

**PowerShell script execution error** — run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Results

| Metric | Score |
|---|---|
| Best Model | Logistic Regression |
| F1 Score | 0.830 |
| ROC-AUC | 0.913 |
| Recall (disease) | 89% |
| Accuracy | 83% |
