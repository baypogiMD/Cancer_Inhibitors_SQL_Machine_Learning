# Cancer Inhibitors – SQL & Machine Learning Analysis

This repository provides a full **cheminformatics and machine learning pipeline**
for analyzing **cancer kinase inhibitors**, integrating:

- SQL-based data querying
- Python-based descriptor engineering
- QSAR modeling (classification & regression)
- Model interpretability and virtual screening

Dataset source:
https://www.kaggle.com/datasets/xiaotawkaggle/inhibitors

---

## Objectives

- Explore structure–activity relationships (SAR)
- Predict inhibitor activity using machine learning
- Identify key molecular features driving potency
- Build a reproducible SQL → ML drug discovery workflow

---

## Tech Stack

- **Python**: pandas, numpy, scikit-learn, RDKit, SHAP
- **SQL**: SQLite / PostgreSQL compatible
- **ML Models**: Random Forest, XGBoost
- **Cheminformatics**: ECFP fingerprints, physchem descriptors

---

## Workflow Overview

1. Raw CSV ingestion
2. SQL-based preprocessing & statistics
3. Descriptor generation using RDKit
4. Classification (Active vs Inactive)
5. Regression (pIC50 prediction)
6. Model explainability
7. Virtual screening

