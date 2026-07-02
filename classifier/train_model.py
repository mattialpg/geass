import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


#!------ Load data ------#
df_pairs = pd.read_pickle('xgb_reaction_pairs.pkl').reset_index(drop=True)


#!------ Build positive samples ------#
X_pos = np.stack((df_pairs['Feature_vec_R'] - df_pairs['Feature_vec_P']).abs().to_list()).astype(np.int32)
y_pos = np.ones(len(X_pos), dtype=np.int32)


#!------ Build negative samples (capped) ------#
neg_ratio = 5
max_neg = neg_ratio * len(X_pos)

X_neg_chunks = []
neg_count = 0

for seed in range(20):
    if neg_count >= max_neg:
        break

    shuffled = df_pairs.copy()
    shuffled['Feature_vec_P'] = df_pairs['Feature_vec_P'].sample(
        frac=1, random_state=seed).reset_index(drop=True)

    diffs_neg = np.stack(
        (shuffled['Feature_vec_R'] - shuffled['Feature_vec_P']).abs().to_list()
    ).astype(np.int32)

    remaining = max_neg - neg_count
    if diffs_neg.shape[0] > remaining:
        idx = np.random.RandomState(seed).choice(diffs_neg.shape[0], size=remaining, replace=False)
        diffs_neg = diffs_neg[idx]

    X_neg_chunks.append(diffs_neg)
    neg_count += diffs_neg.shape[0]

X_neg = np.vstack(X_neg_chunks)
y_neg = np.zeros(len(X_neg), dtype=np.int32)


#!------ Combine data ------#
diffs = np.vstack([X_pos, X_neg])
y = np.concatenate([y_pos, y_neg])


#!------ Explicit distance features (monotonic anchors) ------#
l1 = diffs.sum(axis=1, keepdims=True)
l2 = np.sqrt((diffs * diffs).sum(axis=1, keepdims=True))
linf = diffs.max(axis=1, keepdims=True)
nnz = (diffs > 0).sum(axis=1, keepdims=True)

X = np.hstack([diffs, l1, l2, linf, nnz]).astype(np.float32)


#!------ Cross validation ------#
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
reports = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    n_all = X_train.shape[1]
    monotone_constraints = '(' + ','.join(['-1'] * n_all) + ')'

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=0,
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        max_bin=256,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)


#!------ Train final model on all data ------#
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
scale_pos_weight = n_neg / max(n_pos, 1)

n_base = X.shape[1] - 4
monotone_constraints = '(' + ','.join(['0'] * n_base + ['-1', '-1', '-1', '-1']) + ')'

final_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=0,
    eval_metric='logloss',
    n_jobs=-1,
    tree_method='hist',
    max_bin=256,
    scale_pos_weight=scale_pos_weight,
    monotone_constraints=monotone_constraints
)

final_model.fit(X, y)
joblib.dump(final_model, 'xgb_classifier.joblib')


#!------ Report performance ------#
df_reports = pd.DataFrame(reports)

summary = {
    'accuracy': df_reports['accuracy'].mean(),
    'precision_macro': df_reports['macro avg'].apply(lambda x: x['precision']).mean(),
    'recall_macro': df_reports['macro avg'].apply(lambda x: x['recall']).mean(),
    'f1_macro': df_reports['macro avg'].apply(lambda x: x['f1-score']).mean(),
    'precision_weighted': df_reports['weighted avg'].apply(lambda x: x['precision']).mean(),
    'recall_weighted': df_reports['weighted avg'].apply(lambda x: x['recall']).mean(),
    'f1_weighted': df_reports['weighted avg'].apply(lambda x: x['f1-score']).mean()
}

print('\n=== Mean performance across folds ===')
for k, v in summary.items():
    print(f'{k:20s}: {v:.3f}')
