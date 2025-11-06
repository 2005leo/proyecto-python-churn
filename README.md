# Customer Churn — EDA & ML with Python (Synthetic Dataset)

**ES:** Proyecto de portfolio que muestra un flujo completo de analítica y machine learning para predicción de churn de clientes, usando datos sintéticos y un pipeline reproducible.

**EN:** Portfolio project showcasing an end-to-end analytics and machine learning workflow for customer churn prediction using synthetic data and a fully reproducible pipeline.

---

## 1. Problem Statement

**ES**

Una empresa de suscripción digital quiere anticipar qué clientes tienen mayor riesgo de darse de baja (“churn”) para priorizar acciones de retención. El objetivo es:

- Explorar los datos y entender los drivers del churn.
- Entrenar modelos de clasificación interpretables y comparables.
- Generar métricas y visualizaciones claras para negocio.

**EN**

A digital subscription company wants to anticipate which customers are at higher risk of churn, in order to prioritize retention actions. The goals are to:

- Explore the data and understand churn drivers.
- Train interpretable and comparable classification models.
- Produce clear metrics and business-friendly visualizations.

> ⚠️ **Disclaimer (ES):** El dataset es 100% sintético (generado en este repo). No contiene datos reales de clientes.  
> ⚠️ **Disclaimer (EN):** The dataset is 100% synthetic (generated within this repo). No real customer data is used.

---

## 2. Tech Stack

- Python, pandas, NumPy  
- scikit-learn  
- matplotlib, seaborn  
- Jupyter (VS Code)

---

## 3. Project Structure

```text
proyecto-python-churn/
├─ data/
│  └─ raw/
│     └─ churn_synthetic.csv          # synthetic dataset
├─ notebooks/
│  └─ 01_eda.ipynb                    # EDA (ES/EN)
├─ src/
│  ├─ make_dataset.py                 # generate synthetic dataset
│  ├─ train.py                        # train and evaluate models
│  └─ utils.py                        # helper functions (save metrics)
├─ reports/
│  ├─ figs/
│  │  ├─ roc_curve_logreg.png
│  │  ├─ roc_curve_rf.png
│  │  ├─ conf_matrix_logreg.png
│  │  ├─ conf_matrix_rf.png
│  │  └─ feature_importance.png
│  └─ metrics.json                    # consolidated metrics
├─ models/
│  ├─ model_logreg.pkl                # Logistic Regression pipeline
│  └─ model_rf.pkl                    # Random Forest pipeline
├─ requirements.txt
└─ README.md
