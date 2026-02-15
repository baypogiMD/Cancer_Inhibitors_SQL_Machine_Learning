import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------
# CLASSIFICATION METRICS
# ---------------------------------------------------------

def evaluate_classification(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        "ROC_AUC": roc_auc_score(y_true, y_probs),
        "PR_AUC": average_precision_score(y_true, y_probs),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_probs)
    }

    return metrics


# ---------------------------------------------------------
# REGRESSION METRICS
# ---------------------------------------------------------

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ---------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------

def bootstrap_metric(
    y_true,
    y_pred,
    metric_function,
    n_bootstrap=1000,
    random_state=42
):
    np.random.seed(random_state)
    scores = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), n, replace=True)
        score = metric_function(
            y_true[idx],
            y_pred[idx]
        )
        scores.append(score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return {
        "Mean": np.mean(scores),
        "CI_lower": lower,
        "CI_upper": upper
    }


# ---------------------------------------------------------
# CALIBRATION CURVE DATA
# ---------------------------------------------------------

def get_calibration_data(y_true, y_probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_probs,
        n_bins=n_bins
    )

    return pd.DataFrame({
        "Predicted Probability": prob_pred,
        "True Probability": prob_true
    })


# ---------------------------------------------------------
# CROSS-VALIDATION SUMMARY
# ---------------------------------------------------------

def summarize_cv_scores(scores_list):
    return {
        "Mean": np.mean(scores_list),
        "Std": np.std(scores_list),
        "Min": np.min(scores_list),
        "Max": np.max(scores_list)
    }


# ---------------------------------------------------------
# REGRESSION ERROR DISTRIBUTION
# ---------------------------------------------------------

def regression_error_distribution(y_true, y_pred):
    errors = y_true - y_pred

    return pd.DataFrame({
        "Error": errors,
        "Absolute_Error": np.abs(errors)
    })


# ---------------------------------------------------------
# PUBLICATION-READY REPORT FORMAT
# ---------------------------------------------------------

def format_metrics_for_report(metrics_dict, decimals=4):
    return {
        k: round(v, decimals)
        for k, v in metrics_dict.items()
    }
