# Multi-Class Classification using Random Forest and AdaBoost

## Course

Programming for AI

---

# Project Overview

This project demonstrates a complete machine learning workflow for solving a multi-class classification problem using:

- Random Forest Classifier
- AdaBoost Classifier

The project also integrates introductory MLOps tools including:

- Git
- GitHub
- DVC
- MLflow

---

# Dataset

Dry Bean Dataset

Dataset Source:
https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

---

# Project Features

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Multi-Class Classification
- Hyperparameter Tuning
- Class Imbalance Handling
- Experiment Tracking
- Dataset Version Control
- Model Saving

---
# Project Directory
ai-multiclass-lab/
│
├── data/
│   ├── raw/
│   │   └── Dry_Bean_Dataset.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── multiclass_lab.ipynb
│
├── experiments/
│   ├── exp_01_baseline_rf.py
│   ├── exp_02_tuned_rf.py
│   ├── exp_03_balanced_rf.py
│   ├── exp_04_adaboost.py
│   └── experiment_results.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── config.py
│   └── utils.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── experiment_analysis.md
│
├── mlruns/
│
├── README.md
├── requirements.txt
├── .gitignore
├── dvc.yaml
└── main.py

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- MLflow
- DVC
- Git

---

# Installation

## Create Virtual Environment

python -m venv venv

## Activate Environment

venv\Scripts\activate

## Install Dependencies

pip install -r requirements.txt

---

# Run Main Pipeline

python main.py

---

# Run MLflow UI

mlflow ui

Open browser:

http://127.0.0.1:5000

---

# Models Used

1. Random Forest Classifier
2. AdaBoost Classifier

---

# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Author
Imran Ghafoor
