# 🏠 Bangalore House Price Predictor

A machine learning web application that predicts house prices in Bangalore based on location, square footage, number of bedrooms (BHK), and bathrooms. Built with **Flask** and a **Ridge Regression** model trained on the [Bengaluru House Price dataset](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) from Kaggle.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-orange)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Roopanshi-Marwaha/Bangalore_House_Price_Prediction/blob/main/Bangalore_House_Price_Prediction.ipynb)

### 🌐 [Live Demo](https://bangalore-house-price-prediction-pb1d.onrender.com)

> **Note:** The app is hosted on Render's free tier, which spins down after periods of inactivity. If the link seems to hang, wait 30–50 seconds for the server to wake up — subsequent loads will be fast.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Outlier Removal](#outlier-removal)
- [Model Training](#model-training)
- [How the Web App Works](#how-the-web-app-works)
- [Installation & Local Setup](#installation--local-setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

This project predicts the price of a house in Bangalore given four inputs:

- **Location** (dropdown of real Bangalore neighborhoods)
- **BHK** (number of bedrooms)
- **Number of bathrooms**
- **Total square footage**

The model was trained in Google Colab on the raw Bengaluru house price dataset (downloaded via `kagglehub`), cleaned and de-outliered through several stages, and finally trained using **Ridge Regression** — chosen after comparing it against plain Linear Regression and Lasso Regression on R² score.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn — `Ridge` inside a `Pipeline` with `OneHotEncoder` + `StandardScaler` |
| Data Handling | pandas, numpy |
| Frontend | HTML, Bootstrap 4, vanilla JavaScript (XMLHttpRequest) |
| Model Persistence | pickle |
| Training Environment | Google Colab |

## Project Structure

```
bangalore-house-price-prediction/
│
├── main.py                                    # Flask application (routes + prediction logic)
├── Bangalore_House_Price_Prediction.ipynb      # Data cleaning, feature engineering & model training notebook
├── Cleaned_data.csv                            # Cleaned & de-outliered dataset (output of the notebook)
├── RidgeModel.pkl                              # Trained Ridge regression pipeline (pickled)
├── templates/
│   └── index.html                              # Frontend form + AJAX prediction call
├── requirements.txt                            # Python dependencies
├── .gitignore
└── README.md
```

## Dataset & Preprocessing

The raw dataset (`Bengaluru_House_Data.csv`, ~13,000+ rows) went through the following cleaning steps in the notebook:

1. **Dropped irrelevant columns:** `area_type`, `availability`, `society`, `balcony`.
2. **Handled missing values:**
   - `location` → filled the single missing value with `'Sarjapur Road'`.
   - `size` (e.g. "2 BHK", "3 Bedroom") → filled missing values with `'2 BHK'`.
   - `bath` → filled missing values with the column's median.
3. **Feature engineering:**
   - Extracted `bhk` as an integer from the `size` column (e.g. "4 BHK" → `4`).
   - Cleaned `total_sqft`, which originally contained ranges like `"2100-2850"` — these were converted to their average value using a custom `convertRange()` function; unparseable values became `NaN`.
   - Created a temporary `price_per_sqft` column (`price * 100000 / total_sqft`) purely to help detect and remove pricing outliers.
4. **Location grouping:** Locations appearing **10 times or fewer** in the dataset were grouped into a single `'other'` category, since one-hot encoding hundreds of rare, sparsely-represented locations would make the model unstable and prone to overfitting.

## Outlier Removal

Several rounds of outlier removal were applied to improve model quality:

- **Unrealistic size filter:** Removed rows where `total_sqft / bhk < 300`, since anything smaller than ~300 sq. ft. per bedroom is not a realistic residential unit.
- **Price-per-sqft outliers (per location):** For each location, rows with `price_per_sqft` more than one standard deviation away from that location's mean were removed — this strips out properties that are priced wildly differently from comparable homes in the same area.
- **BHK-based outliers:** For each location, if a lower BHK (e.g. 2 BHK) had a *higher* average `price_per_sqft` than the next BHK up (e.g. 3 BHK) — despite there being enough data points (>5) to trust the comparison — those anomalous higher-BHK rows priced below the lower-BHK average were removed. This catches cases where, for example, a 3 BHK is illogically cheaper per sq. ft. than a 2 BHK in the same area.

After cleaning, the `size` and `price_per_sqft` helper columns were dropped, and the final cleaned dataset (`location`, `total_sqft`, `bath`, `bhk`, `price`) was saved to `Cleaned_data.csv`.

## Model Training

- **Features (X):** `location`, `total_sqft`, `bath`, `bhk`
- **Target (y):** `price` (in **lakhs of rupees**)
- **Train/test split:** 80/20 (`random_state=0`)
- **Pipeline:** `OneHotEncoder` (applied only to `location`) → `StandardScaler` → regression model
- **Models compared** (via R² score on the test set):
  - Linear Regression (no regularization)
  - Lasso Regression
  - Ridge Regression
- **Final model:** Ridge Regression was selected and serialized to `RidgeModel.pkl` using `pickle`.

> The exact R² scores for each model are printed at the end of the notebook — check `Bangalore_House_Price_Prediction.ipynb` for the latest run's numbers, since they can shift slightly if the notebook is re-run (e.g. after a package version change).

## How the Web App Works

1. **`GET /`** — Loads the home page and populates the location dropdown from the unique values in `Cleaned_data.csv`.
2. User fills in location, BHK, bathrooms, and square footage, then clicks **Predict Price**.
3. JavaScript intercepts the form submission and sends the data via `XMLHttpRequest` to **`POST /predict`** — without a page reload.
4. Flask reads the form data, builds a single-row DataFrame matching the model's expected input schema (`location`, `total_sqft`, `bath`, `bhk`), and passes it to the trained pipeline.
5. The pipeline one-hot encodes the location, scales the numeric features, and feeds them into the Ridge model to get a raw prediction.
6. Since the model was trained on prices expressed in **lakhs**, the raw prediction is multiplied by `1e5` to convert it into rupees before being sent back to the frontend.

## Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Roopanshi-Marwaha/Bangalore_House_Price_Prediction.git
cd Bangalore_House_Price_Prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python main.py
```

The app will be available at **http://127.0.0.1:5000**.

> **Note:** The scikit-learn version used to train `RidgeModel.pkl` must match (or be compatible with) the version installed in your environment, or you may encounter unpickling errors. If you retrain the model (e.g. re-run the notebook), make sure to install the same scikit-learn version locally before running the app.

## Usage

1. Open the app in your browser.
2. Select a **location** from the dropdown.
3. Enter the **BHK**, **number of bathrooms**, and **total square footage**.
4. Click **Predict Price**.
5. The estimated price (in ₹) appears instantly below the form.

## Deployment

This app can be deployed on any platform that supports Python/Flask apps, such as **Render**, **Railway**, or **PythonAnywhere**. Key steps for a typical deployment (e.g. on Render):

1. Generate a pinned `requirements.txt` with `pip freeze > requirements.txt`.
2. Add `gunicorn` as a production WSGI server.
3. Update `main.py` to use `host='0.0.0.0'` and read the port from the `PORT` environment variable, and remove `debug=True`.
4. Push the repo to GitHub (make sure `RidgeModel.pkl` and `Cleaned_data.csv` are committed, not ignored).
5. Create a new Web Service on the hosting platform, pointing to the repo, with build command `pip install -r requirements.txt` and start command `gunicorn main:app`.

## Known Limitations

- **Sparse locations after outlier removal:** Even though locations with ≤10 total entries were grouped into `'other'` *before* outlier removal, the subsequent outlier-removal steps (price-per-sqft and BHK-based filtering) can shrink a location's row count well below 10 in the final cleaned dataset. Predictions for these thinly-represented locations can occasionally be less reliable than for well-represented areas like Whitefield or Sarjapur Road.
- **Unusual input combinations:** Inputs far outside the training data's typical range (e.g. a very high bathroom count with very low square footage) may produce less realistic predictions, since the model has seen few similar examples after outlier filtering.
- **Development server:** By default, the app runs with Flask's built-in development server, which is not intended for production use — see the [Deployment](#deployment) section.

## Future Improvements

- Re-check location counts *after* the outlier-removal steps and re-group any that have fallen below a safe threshold
- Add input validation (e.g. reasonable min/max ranges for BHK, bathrooms, square footage relative to each other)
- Replace the Flask dev server with a production WSGI server (e.g. Gunicorn) for deployment
- Add automated tests for the `/predict` endpoint
- Improve UI/UX with better styling and clearer error messages for invalid inputs

## Author

**Roopanshi Marwaha**
GitHub: [@Roopanshi-Marwaha](https://github.com/Roopanshi-Marwaha)

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
