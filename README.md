# 🚀 Predictive Maintenance IoT Project

## 🔧 Contextual Predictive Maintenance System

### 🎯 Project Objective

Predict machine failures using IoT telemetry data and Machine Learning techniques before breakdowns occur, enabling proactive maintenance and reducing operational downtime.

---

## 📖 Problem Statement

Industrial equipment continuously generates large volumes of sensor telemetry data. Unexpected machine failures can lead to:

* Production downtime
* Increased maintenance costs
* Reduced operational efficiency
* Significant financial losses

This project leverages Machine Learning algorithms to analyze IoT sensor data and predict potential machine failures before they occur, allowing preventive maintenance actions.

---

## 📊 Dataset Information

### Dataset

**AI4I 2020 Predictive Maintenance Dataset**

### Sensor Features

| Feature                 | Description                 |
| ----------------------- | --------------------------- |
| Air Temperature [K]     | Ambient air temperature     |
| Process Temperature [K] | Machine process temperature |
| Rotational Speed [rpm]  | Machine rotational speed    |
| Torque [Nm]             | Applied torque              |
| Tool Wear [min]         | Tool wear duration          |

### Target Variable

| Variable        | Description                 |
| --------------- | --------------------------- |
| Machine Failure | 0 = No Failure, 1 = Failure |

---

## 👥 Team Members

| Member   | Role                           | Name           |
| -------- | ------------------------------ | -------------- |
| Member 1 | Data Engineer                  | Tarun Saxena   |
| Member 2 | ML Engineer                    | Vaibhav Gautam |
| Member 3 | Context & Integration Engineer | Vaibhav Gautam |
| Member 4 | Evaluation & Deployment Lead   | Tarun Saxena   |

---

## 🎯 Target KPIs

| Metric         | Target |
| -------------- | ------ |
| Macro F1 Score | ≥ 0.85 |
| Accuracy       | ≥ 90%  |
| Precision      | ≥ 85%  |
| Recall         | ≥ 85%  |

---

## 🛠️ Development Environment

### Environment Setup

* Python Virtual Environment Configured
* GitHub Repository Initialized
* Project Structure Created
* Jupyter Notebook Environment Configured

### Project Structure

```text
predictive-maintenance-iot/
│
├── data/
├── notebooks/
├── src/
├── models/
├── docs/
├── README.md
└── requirements.txt
```

---

## 📦 Installed Dependencies

```text
pandas
numpy
lightgbm
imbalanced-learn
shap
matplotlib
seaborn
scikit-learn
```

---

## 🔄 Project Workflow

1. Data Collection & Validation
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Fusion & Context Integration
5. Model Training
6. Model Evaluation
7. Explainability using SHAP
8. Deployment Preparation

---

## 📈 Expected Outcome

Develop an intelligent predictive maintenance system capable of identifying machine failure risks in advance, enabling proactive maintenance and minimizing operational disruptions.

---

# 🚀 Predictive Maintenance IoT Project

## 📅 Week 1 Progress Report

---

# Day 1 Progress

## 👨‍💻 Member 1 – Data Engineer (Tarun Saxena)

### ✅ Tasks Completed

* Created project folder structure:

  * `data/`
  * `notebooks/`
  * `src/`
  * `models/`
* Set up Python virtual environment.
* Installed required dependencies:

  * pandas
  * numpy
  * lightgbm
  * imbalanced-learn
  * shap
  * matplotlib
  * seaborn
  * scikit-learn
* Verified environment setup and package installation.
* Committed changes to GitHub repository.

### 📦 Deliverables

* Project structure created
* Virtual environment configured
* Required packages installed

### 🎯 Outcome

Development environment successfully prepared for project implementation.

**Status:** ✅ Completed

---

## 🤖 Member 2 – ML Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Set up Jupyter Notebook development environment.
* Verified Jupyter Notebook installation.
* Installed and configured **nbstripout**.
* Enabled Git pre-commit hook.
* Created notebook:

  * `week1_eda.ipynb`
* Added notebook sections:

  * Project Title
  * Introduction
  * Dataset Overview
  * EDA
  * Observations

### 📦 Deliverables

* Jupyter Notebook environment configured
* nbstripout activated
* EDA notebook skeleton created

### 🎯 Outcome

EDA workspace successfully prepared for analysis activities.

**Status:** ✅ Completed

---

## 🔗 Member 3 – Context & Integration Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Created notebook:

  * `week2_fusion.ipynb`
* Added sections:

  * Project Overview
  * Data Sources
  * Data Fusion Strategy
  * Feature Integration
  * Results & Observations
* Installed GeoPandas and dependencies.
* Reviewed AI4I Predictive Maintenance dataset documentation.

### 📦 Deliverables

* Fusion notebook skeleton
* GeoPandas environment configured
* Dataset documentation reviewed

### 🎯 Outcome

Feature integration framework prepared for Week 2.

**Status:** ✅ Completed

---

## 📊 Member 4 – Evaluation & Deployment Lead (Tarun Saxena)

### ✅ Tasks Completed

* Created GitHub repository:

  * `predictive-maintenance-iot`
* Added `.gitignore`
* Created `README.md`
* Set up GitHub Kanban Board
* Created project issues and labels
* Assigned issues to team members

### 📦 Deliverables

* Repository initialized
* Project documentation added
* Kanban workflow configured

### 🎯 Outcome

Project management structure established.

**Status:** ✅ Completed

---

# Day 2 Progress

## 👨‍💻 Member 1 – Data Engineer (Tarun Saxena)

### ✅ Tasks Completed

* Downloaded AI4I 2020 Predictive Maintenance Dataset.
* Loaded dataset using Pandas.
* Performed initial inspection and validation.

### 📊 Dataset Inspection Results

| Metric          | Value           |
| --------------- | --------------- |
| Rows            | 10,000          |
| Columns         | 14              |
| Target Variable | Machine Failure |

### 🔑 Key Features

* Product ID
* Type
* Air Temperature [K]
* Process Temperature [K]
* Rotational Speed [rpm]
* Torque [Nm]
* Tool Wear [min]
* Machine Failure
* TWF, HDF, PWF, OSF, RNF

### 🎯 Outcome

Dataset successfully validated and ready for EDA.

**Status:** ✅ Completed

---

## 🤖 Member 2 – ML Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Loaded dataset into EDA notebook.
* Generated class distribution visualization.
* Calculated class imbalance ratio.

### 📊 Class Distribution

| Class      | Samples | Percentage |
| ---------- | ------- | ---------- |
| No Failure | 9661    | 96.61%     |
| Failure    | 339     | 3.39%      |

**Imbalance Ratio:** `28.5 : 1`

### 🎯 Outcome

Class imbalance identified and documented for future SMOTE implementation.

**Status:** ✅ Completed

---

## 🔗 Member 3 – Context & Integration Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Created `src/feature_engineering.py`
* Defined `SENSOR_COLUMNS`
* Implemented `sort_and_reset()`
* Added rolling feature generator skeleton
* Added documentation and comments

### 📦 Deliverables

* Feature engineering framework
* Sensor column definitions
* Utility functions

### 🎯 Outcome

Foundation established for rolling feature generation.

**Status:** ✅ Completed

---

## 📊 Member 4 – Evaluation & Deployment Lead (Tarun Saxena)

### ✅ Tasks Completed

* Reviewed repository structure.
* Created `AGENTS.md`.
* Documented:

  * Team roles
  * Deliverables
  * GitHub workflow
  * Governance guidelines

### 🎯 Outcome

Team collaboration process fully documented.

**Status:** ✅ Completed

---

# Day 3 Progress

## 👨‍💻 Member 1 – Data Engineer (Tarun Saxena)

### ✅ Tasks Completed

* Performed missing value analysis.
* Checked duplicate records.
* Validated dataset structure.

### 📊 Data Quality Results

| Check             | Result |
| ----------------- | ------ |
| Missing Values    | 0      |
| Duplicate Records | 0      |

### 🎯 Outcome

Dataset verified and ready for preprocessing and feature engineering.

**Status:** ✅ Completed

---

## 🤖 Member 2 – ML Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Continued EDA analysis.
* Documented class imbalance observations.
* Prepared notebook for correlation analysis.

### 🎯 Outcome

EDA phase progressing successfully.

**Status:** ✅ Completed

---

## 🔗 Member 3 – Context & Integration Engineer (Vaibhav Gautam)

### ✅ Tasks Completed

* Extended feature engineering framework.
* Prepared rolling feature pipeline for implementation.
* Maintained project documentation and comments.

### 🎯 Outcome

Feature engineering workflow ready for next development phase.

**Status:** ✅ Completed

---

## 📊 Member 4 – Evaluation & Deployment Lead (Tarun Saxena)

### ✅ Tasks Completed

* Created governance documentation.
* Added project context memory files.
* Documented dataset metadata.
* Added sprint objectives.

### 🎯 Outcome

Project governance and documentation framework strengthened.

**Status:** ✅ Completed

---

# Day 3 Progress 

### 🔹 Member 1 — Data Engineer(Tarun Saxena)
**Task Completed:**
- Encoded categorical `Type` column using `LabelEncoder`.
- Created new feature: `Type_enc`.
- Verified encoding accuracy by comparing value counts before and after transformation.
- Ensured no data loss occurred during preprocessing.

**Deliverables:**
- Cleaned and encoded dataset.
- Updated preprocessing workflow.

**Status:** ✅ Completed

---

### 🔹 Member 2 — ML Engineer(Vaibhav Gautam)
**Task Completed:**
- Generated distribution plots for all sensor variables.
- Compared sensor behavior across:
  - Failure = 0 (Normal Operation)
  - Failure = 1 (Machine Failure)
- Saved visualization outputs for future model analysis.

**Deliverables:**
- Sensor distribution histograms.
- Failure class comparison analysis.

**Status:** ✅ Completed

---

### 🔹 Member 3 — Context & Integration Engineer(Vaibhav Gautam)
**Task Completed:**
- Implemented rolling statistical features:
  - Rolling Mean
  - Rolling Standard Deviation
  - Rolling Variance
- Applied feature generation to all sensor columns.
- Removed NaN values generated during rolling window calculations.
- Validated resulting dataset dimensions.

**Deliverables:**
- Enhanced feature engineering pipeline.
- Statistical rolling feature set.

**Status:** ✅ Completed

---

### 🔹 Member 4 — Evaluation & Deployment Lead(Tarun Saxena)
**Task Completed:**
- Created centralized `progress_tracker.md`.
- Monitored task completion across all team members.
- Reviewed completed pull requests from Week 1 Day 1–3.
- Verified issue references and repository standards.
- Merged approved contributions into the main branch.
- Updated project documentation and sprint tracking.

**Deliverables:**
- Team progress tracker.
- Reviewed and merged PRs.
- Updated project governance records.

**Status:** ✅ Completed

---

# Week 1 Summary

## Sprint Goal

Prepare and validate the AI4I Predictive Maintenance dataset for machine learning workflows.

## Data Loading

- Downloaded AI4I Predictive Maintenance Dataset
- Verified dataset structure
- Verified column names and data types

## Data Cleaning

- Missing values checked
- Duplicate rows checked
- Dataset consistency verified
## Feature Engineering

- Encoded Type column
- Created Type_enc feature

## Dataset Statistics

- Rows: 10000
- Columns: 14
- Missing Values: 0
- Duplicate Rows: 0

## Product Type Distribution

- L : 6000
- M : 2997
- H : 1003

## Deliverables

- data_loader.py
- cleaned_ai4i.csv
- dataset review scripts

## Week 1 Status

✅ Completed Successfully

---
## 📅 Week 2 Progress Report

---

# Day 1 Progress

## 👨‍💻 Member 1 – Data Engineer (Tarun Saxena)
