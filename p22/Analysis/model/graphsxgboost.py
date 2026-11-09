# Main writer: Elena Gostiukhina
# Reviewer: Luca Dragutanu
# Contributors: Aadya Charlewar

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


GRAPH_DIR = Path("graphsxgboost")
GRAPH_DIR.mkdir(exist_ok=True)


def save_confusion_matrix(model, X, y, split_name, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No AKI", "AKI"],
    )

    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {split_name}")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"confusion_matrix_{split_name}.png", dpi=300)
    plt.close()


def save_roc_curve(model, X, y, split_name):
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"ROC Curve - {split_name}")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"roc_curve_{split_name}.png", dpi=300)
    plt.close()


def save_pr_curve(model, X, y, split_name):
    PrecisionRecallDisplay.from_estimator(model, X, y)
    plt.title(f"Precision-Recall Curve - {split_name}")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"pr_curve_{split_name}.png", dpi=300)
    plt.close()


def save_feature_importance(model, feature_names, top_n=15):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    importance.sort_values().plot(kind="barh", figsize=(8, 10))
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "feature_importance_top30.png", dpi=300)
    plt.close()


def save_shap_summary(model, X, split_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"shap_summary_{split_name}.png", dpi=300)
    plt.close()
