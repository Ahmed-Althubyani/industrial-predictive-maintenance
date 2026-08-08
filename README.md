# Industrial Predictive Maintenance

An end-to-end machine learning project for predicting industrial machine failures using sensor and operational data from the **AI4I 2020 Predictive Maintenance Dataset**.

The project covers data exploration, preprocessing, feature engineering, model comparison and tuning, evaluation, model serialization, and serving predictions through a **FastAPI REST API** deployed in **Docker**.

## Project Pipeline

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering
   ↓
Model Training & Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Saved Model Artifacts
   ↓
FastAPI Prediction API
   ↓
Docker Container
```

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, containing 10,000 observations of industrial machine operating conditions.

Key features include:

* Product type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

The target variable is `Machine failure`, indicating whether a machine failure occurred.

## Machine Learning

### Models

The following classification models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Hyperparameter tuning was performed for Random Forest and XGBoost.

### Evaluation

Because machine failure is an imbalanced classification problem, the models were evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

### Results

| Model                 | Precision |    Recall |        F1 |   ROC-AUC |
| --------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression   |     0.667 |     0.235 |     0.348 |     0.910 |
| Decision Tree         |     0.722 |     0.765 |     0.743 |     0.877 |
| Random Forest         |     0.973 |     0.706 |     0.818 |     0.967 |
| XGBoost               |     0.837 |     0.706 |     0.766 |     0.979 |
| Random Forest (tuned) |     0.949 |     0.725 |     0.822 |     0.970 |
| **XGBoost (tuned)**   | **0.927** | **0.745** | **0.826** | **0.984** |

**Best model: XGBoost (tuned)**

## Project Structure

```text
industrial-predictive-maintenance/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── raw/
│       └── ai4i2020Raw.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model.ipynb
│
├── src/
│   └── industrial_predictive_maintenance/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── predict.py
│       └── preprocessing.py
│
├── models/
│   └── model_artifacts.pkl
│
├── tests/
│
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## API

The trained model is exposed through a **FastAPI REST API**.

### Run Locally

```bash
uv run uvicorn app.main:app --reload --port 9696
```

API documentation:

```text
http://localhost:9696/docs
```

### Example Request

```json
{
  "type": "L",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1500.0,
  "torque": 40.0,
  "tool_wear": 100.0
}
```

### Example Response

```json
{
  "prediction": 0,
  "failure_probability": 0.0006575718871317804
}
```

Where:

* `prediction = 0` → No failure predicted
* `prediction = 1` → Failure predicted
* `failure_probability` → Model probability of machine failure

## Docker

The API can be run as a Docker container.

### Build

```bash
docker build -t industrial-predictive-maintenance .
```

### Run

```bash
docker run --rm -p 9696:9696 industrial-predictive-maintenance
```

API documentation:

```text
http://localhost:9696/docs
```

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **FastAPI**
* **Docker**
* **uv**
* **Git & GitHub**

## Key Learning Outcomes

This project demonstrates experience with:

* End-to-end machine learning workflows
* Exploratory data analysis
* Data preprocessing and feature engineering
* Imbalanced classification
* Model comparison and hyperparameter tuning
* Model evaluation
* Model serialization and inference
* REST API development
* Docker containerization
* Reproducible Python environments

## Future Improvements

* Deploy the API to AWS
* Add CI/CD with GitHub Actions
* Add model monitoring
* Add a web dashboard for predictions

## License

This project is for educational and portfolio purposes.
