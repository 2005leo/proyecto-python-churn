# src/train.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
)
import joblib
from utils import save_metrics

def main():
    df = pd.read_csv("data/raw/churn_synthetic.csv")

    X = df.drop(columns=["churn"])
    y = df["churn"].astype(int)

    num_cols = ["tenure_months","monthly_charges","support_tickets_90d","app_logins_30d"]
    cat_cols = ["contract_type","payment_method"]
    bin_cols = ["is_fiber","promo_active"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("bin", "passthrough", bin_cols)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

    # Modelos
    models = {
        "logreg": LogisticRegression(max_iter=200, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=7)
    }

    results = {}
    Path("reports/figs").mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
        report = classification_report(y_test, y_pred, output_dict=True)

        # ROC
        if y_proba is not None:
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
            ax.set_title(f"ROC — {name.upper()}")
            fig.savefig(f"reports/figs/roc_curve_{name}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

        # Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title(f"Confusion Matrix — {name.upper()}")
        fig.savefig(f"reports/figs/conf_matrix_{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        # Importancias (solo RF)
        if name == "rf":
            # tomar nombres de features del preprocesador
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            cat_names = ohe.get_feature_names_out(cat_cols)
            feature_names = np.r_[num_cols, cat_names, bin_cols]
            imps = pipe.named_steps["clf"].feature_importances_
            order = np.argsort(imps)[::-1][:12]

            fig, ax = plt.subplots(figsize=(7,4.5))
            ax.barh(feature_names[order][::-1], imps[order][::-1])
            ax.set_title("Feature Importance — RandomForest")
            fig.tight_layout()
            fig.savefig("reports/figs/feature_importance.png", dpi=140)
            plt.close(fig)

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(roc, 4) if not np.isnan(roc) else None,
            "report": report
        }

        # guarda modelo
        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipe, f"models/model_{name}.pkl")

    # guarda métricas agregadas
    save_metrics(results, "reports/metrics.json")
    print("[ok] trained models. See reports/ and models/")

if __name__ == "__main__":
    main()
