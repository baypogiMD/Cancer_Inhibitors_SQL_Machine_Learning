import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem


# ---------------------------------------------------------
# SCAFFOLD SPLIT (CHEMICALLY REALISTIC)
# ---------------------------------------------------------

def scaffold_split(df, smiles_col="smiles", test_size=0.2, random_state=42):
    scaffolds = {}

    for idx, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaffold, []).append(idx)

    scaffold_sets = list(scaffolds.values())
    np.random.seed(random_state)
    np.random.shuffle(scaffold_sets)

    train_idx, test_idx = [], []
    total_len = len(df)

    for scaffold_group in scaffold_sets:
        if len(test_idx) / total_len < test_size:
            test_idx.extend(scaffold_group)
        else:
            train_idx.extend(scaffold_group)

    return df.iloc[train_idx], df.iloc[test_idx]


# ---------------------------------------------------------
# CLASSIFICATION MODEL
# ---------------------------------------------------------

def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)

    return {
        "AUC": auc,
        "Accuracy": acc
    }


# ---------------------------------------------------------
# REGRESSION MODEL
# ---------------------------------------------------------

def train_regression_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    return {
        "RMSE": rmse,
        "MAE": mae
    }


# ---------------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------------

def cross_validate_model(model, X, y, task="regression", n_splits=5):
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []

    for train_idx, test_idx in kf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])

        if task == "classification":
            metrics = evaluate_classification(
                m,
                X.iloc[test_idx],
                y.iloc[test_idx]
            )
            scores.append(metrics["AUC"])
        else:
            metrics = evaluate_regression(
                m,
                X.iloc[test_idx],
                y.iloc[test_idx]
            )
            scores.append(metrics["RMSE"])

    return {
        "Mean Score": np.mean(scores),
        "Std Dev": np.std(scores)
    }


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def get_feature_importance(model, feature_names, top_k=20):
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    return importance_df.head(top_k)


# ---------------------------------------------------------
# COMPLETE PIPELINE FUNCTION
# ---------------------------------------------------------

def run_full_experiment(
    df,
    feature_df,
    target_column,
    task="regression",
    scaffold_based=False
):
    if scaffold_based:
        train_df, test_df = scaffold_split(df)
        X_train = feature_df.loc[train_df.index]
        X_test = feature_df.loc[test_df.index]
        y_train = train_df[target_column]
        y_test = test_df[target_column]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df,
            df[target_column],
            test_size=0.2,
            random_state=42,
            stratify=df[target_column] if task == "classification" else None
        )

    if task == "classification":
        model = train_classification_model(X_train, y_train)
        metrics = evaluate_classification(model, X_test, y_test)
    else:
        model = train_regression_model(X_train, y_train)
        metrics = evaluate_regression(model, X_test, y_test)

    return model, metrics
