import os, sys
import numpy as np
import pandas as pd
import itertools
from rdkit import Chem
from rdkit.Chem import PandasTools, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import swifter
swifter.set_defaults(progress_bar=False)

import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


df_pairs = pd.read_pickle('xgb_reaction_pairs.pkl')

# Positive data
X_pos = (df_pairs.apply(lambda row: np.abs(row['Feature_vec_R'] - row['Feature_vec_P']),
    axis=1).tolist())
y_pos = [1] * len(X_pos)

# Negative data
X_neg, y_neg = [], []
for seed in range(20):
    shuffled = df_pairs.copy().reset_index(drop=True)
    shuffled['ROMol_P'] = df_pairs['ROMol_P'].sample(
        frac=1, random_state=seed).reset_index(drop=True)

    shuffled['Feature_vec_P'] = df_pairs['Feature_vec_P'].sample(
        frac=1, random_state=seed).reset_index(drop=True)

    # Filter by atom count difference
    mask = shuffled.swifter.apply(lambda row: abs(
        rdMolDescriptors.CalcNumAtoms(row['ROMol_R']) -
        rdMolDescriptors.CalcNumAtoms(row['ROMol_P'])) < 6, axis=1)
    shuffled = shuffled[mask]

    # Calculate difference of precomputed vectors
    X_neg.extend(np.abs(shuffled['Feature_vec_R'] -
        shuffled['Feature_vec_P']).tolist())
    y_neg.extend([0] * len(shuffled))


# Combine positive and negative data
X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)


# Train model with 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=0)

reports = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)

joblib.dump(model, 'xgb_classifier.joblib')

# Print performance
df_reports = pd.DataFrame(reports)
summary = {
    "accuracy": df_reports["accuracy"].mean(),
    "precision_macro": df_reports["macro avg"].apply(lambda x: x["precision"]).mean(),
    "recall_macro": df_reports["macro avg"].apply(lambda x: x["recall"]).mean(),
    "f1_macro": df_reports["macro avg"].apply(lambda x: x["f1-score"]).mean(),
    "precision_weighted": df_reports["weighted avg"].apply(lambda x: x["precision"]).mean(),
    "recall_weighted": df_reports["weighted avg"].apply(lambda x: x["recall"]).mean(),
    "f1_weighted": df_reports["weighted avg"].apply(lambda x: x["f1-score"]).mean()}
print("\n=== Mean performance across folds ===")
for k, v in summary.items():
    print(f"{k:20s}: {v:.3f}")