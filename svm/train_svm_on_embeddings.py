# train_svm_on_embeddings.py
import os
import json
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

FEATURES_FILE = "svm/data/clean_options_synth.csv"  # Person A's file
EMB_DIR = "data/embeddings"               # Person B's outputs
OUT_DIR = "outputs/week3_4"

LABEL_COL = "label_uf_over"               # {-1,0,1}
KEYS = ["date","ticker","option_type","K","tau_days"]

# ---------------- Metrics helpers ----------------
def classification_metrics(y_true, y_prob, y_pred, labels=(-1,0,1)):
    # one-vs-rest AUC (macro)
    label_to_idx = {lab:i for i, lab in enumerate(labels)}
    Y = np.zeros((len(y_true), len(labels)))
    for i, lab in enumerate(y_true):
        Y[i, label_to_idx[lab]] = 1.0
    auc_ovr = roc_auc_score(Y, y_prob, multi_class="ovr", average="macro")
    f1_mac = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "f1_macro": f1_mac, "auc_macro_ovr": auc_ovr}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ---------------- Data loaders ----------------
def load_features():
    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    # sanity: ensure label exists
    assert LABEL_COL in df.columns, f"Missing {LABEL_COL} in {FEATURES_FILE}"
    return df

def load_embeddings(kernel: str, n_components: int):
    path = os.path.join(EMB_DIR, f"{kernel}_kpca_{n_components}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing embeddings: {path}")
    emb = pd.read_csv(path, parse_dates=["date"])
    # embeddings expected as pc1..pcK
    pc_cols = [c for c in emb.columns if c.startswith("pc")]
    assert len(pc_cols) == n_components, "Mismatched # of components vs filename"
    return emb, pc_cols

def merge_on_keys(df_feat: pd.DataFrame, emb: pd.DataFrame):
    # Merge in a stable way
    return df_feat.merge(emb, on=KEYS, how="inner")

# ---------------- Feature Engineering ----------------
def add_engineered_features(df_full, pc_cols):
    """Add engineered features from principal components"""
    df_eng = df_full.copy()
    
    # PC interactions (financial factors often interact)
    df_eng['pc1_pc2'] = df_eng['pc1'] * df_eng['pc2']
    df_eng['pc1_squared'] = df_eng['pc1'] ** 2
    df_eng['pc2_squared'] = df_eng['pc2'] ** 2
    
    # PC ratios (relative factor strength)
    df_eng['pc1_pc2_ratio'] = df_eng['pc1'] / (df_eng['pc2'] + 1e-8)  # avoid division by zero
    
    # Time-based features (options are time-sensitive)
    df_eng['tau_pc1'] = df_eng['tau_days'] * df_eng['pc1']
    df_eng['log_tau'] = np.log(df_eng['tau_days'] + 1)
    
    # Moneyness-related (if K column exists)
    if 'K' in df_eng.columns:
        df_eng['K_pc1'] = df_eng['K'] * df_eng['pc1']
    
    new_features = ['pc1_pc2', 'pc1_squared', 'pc2_squared', 'pc1_pc2_ratio', 
                   'tau_pc1', 'log_tau']
    if 'K' in df_eng.columns:
        new_features.append('K_pc1')
    
    return df_eng, pc_cols + new_features

# ---------------- Splitting ----------------
def time_aware_split(df, date_col="date", test_fraction=0.2):
    df = df.sort_values(date_col)
    n = len(df)
    n_test = int(np.ceil(n * test_fraction))
    test_idx = df.index[-n_test:]
    train_idx = df.index[:-n_test]
    return train_idx, test_idx

# ---------------- Models ----------------
def make_rbf_svm():
    return Pipeline([
        ("scaler", StandardScaler()),   # embeddings are often centered; scaler is still safe
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

def make_linear_svm():
    # Linear SVM baseline on embeddings
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lsvc", LinearSVC(class_weight="balanced", max_iter=5000))
    ])

def grid_search_rbf_svm(X_train, y_train, cv_splits=3):  # Reduced CV folds
    pipe = make_rbf_svm()
    param_grid = {
        "svc__C": [0.1, 1, 10, 100],  # Reduced from 9 to 4 values
        "svc__gamma": ["scale", 0.01, 0.1, 1.0]  # Reduced from 9 to 4 values
    }
    tscv = TimeSeriesSplit(n_splits=cv_splits)  # Back to original CV
    gs = GridSearchCV(
        pipe, param_grid,
        scoring="f1_macro", cv=tscv, n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)
    return gs

def train_eval_for_embedding(df_full, pc_cols, kernel, n_components, use_feature_engineering=True):
    # Apply feature engineering if enabled
    if use_feature_engineering:
        df_full, enhanced_pc_cols = add_engineered_features(df_full, pc_cols)
        feature_cols = enhanced_pc_cols
        print(f"  Using {len(enhanced_pc_cols)} features (including engineered)")
    else:
        feature_cols = pc_cols
        print(f"  Using {len(pc_cols)} base features")
    
    # Split
    train_idx, test_idx = time_aware_split(df_full, "date", test_fraction=0.2)

    X_train = df_full.loc[train_idx, feature_cols].values
    y_train = df_full.loc[train_idx, LABEL_COL].values.astype(int)
    X_test  = df_full.loc[test_idx,  feature_cols].values
    y_test  = df_full.loc[test_idx,  LABEL_COL].values.astype(int)

    # RBF SVM (grid search)
    gs = grid_search_rbf_svm(X_train, y_train, cv_splits=5)
    best = gs.best_estimator_
    y_prob = best.predict_proba(X_test)
    y_pred = best.predict(X_test)
    m_rbf = classification_metrics(y_test, y_prob, y_pred)
    cm_rbf = confusion_matrix(y_test, y_pred, labels=[-1,0,1]).tolist()

    # Linear SVM baseline (no grid; could wrap in CV if desired)
    lsvc = make_linear_svm().fit(X_train, y_train)
    # need probs; approximate with one-hot on predictions for metrics function
    y_pred_lin = lsvc.predict(X_test)
    y_prob_lin = np.zeros((len(y_pred_lin), 3))
    idx_map = {-1:0, 0:1, 1:2}
    for i, lab in enumerate(y_pred_lin):
        y_prob_lin[i, idx_map[lab]] = 1.0
    m_lin = classification_metrics(y_test, y_prob_lin, y_pred_lin)
    cm_lin = confusion_matrix(y_test, y_pred_lin, labels=[-1,0,1]).tolist()

    summary = {
        "kernel": kernel,
        "n_components": n_components,
        "best_params_rbf": gs.best_params_,
        "metrics_rbf": m_rbf,
        "confusion_rbf": cm_rbf,
        "metrics_linear": m_lin,
        "confusion_linear": cm_lin,
    }
    return summary

def main():
    ensure_dir(OUT_DIR)
    df_feat = load_features()

    results = []
    # Sweep PCA kernels for Week 3–4 (optimized for speed)
    sweep = [
        ("rbf", 5),       # Keep the best performing ones
        ("sigmoid", 5),
        ("linear", 5),
        ("linear", 7),    # Test linear with more components since it's fastest
    ]
    for kernel, n_comp in sweep:
        try:
            emb, pc_cols = load_embeddings(kernel, n_comp)
            df_full = merge_on_keys(df_feat, emb)
            summary = train_eval_for_embedding(df_full, pc_cols, kernel, n_comp, use_feature_engineering=False)  # Disable for speed
            results.append(summary)
            print(f"[OK] {kernel} KPCA ({n_comp}): {summary['metrics_rbf']}")
        except Exception as e:
            print(f"[SKIP] {kernel} KPCA ({n_comp}) — {e}")

    # Save results
    out_json = os.path.join(OUT_DIR, "kpca_svm_comparison.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    main()
