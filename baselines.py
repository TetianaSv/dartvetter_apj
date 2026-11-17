from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import yaml
import pandas as pd
import numpy as np
import os

# Random coin, Majority, LR, RF on Kepler

SEED = 42
N_BOOT = 1000
cfg = yaml.safe_load(open("config_cnn.yaml"))
out_dir = cfg["local_path"]
os.makedirs(out_dir + "results", exist_ok=True)
os.makedirs(out_dir + "predictions", exist_ok=True)


def load_csv(path):
    v = np.genfromtxt(path, delimiter=",")
    return v[:, :-1], v[:, -1].astype(int)


def point_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p)}


def bootstrap_ci(y, p, keys=("F1", "ROC-AUC", "PR-AUC")):
    rng = np.random.default_rng(SEED)
    samples = {k: [] for k in keys}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():          # need both classes
            continue
        m = point_metrics(y[idx], p[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in samples.items()}

# Probabilities from the reproduced DART-Vetter (BatchNorm-recalibrated)


def dartvetter_probs(x):
    import torch
    import m1_class
    model = m1_class.DartVetterFiscale2024(cfg["cbn"], cfg["cfs"], cfg["cbff"], cfg["psz"],
                                           cfg["pst"], cfg["rateval"], cfg["fc_layers_num"],
                                           cfg["fc_units"], cfg["fc_output_size"])
    path = out_dir + "trained_models/cnn" + cfg["training_set_name"] + "_bnfix.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        probs = [torch.sigmoid(model.get_classification_output(
            model.get_feature_extraction_output(b))).squeeze(-1)
            for b in torch.split(xt, cfg["batch_size"])]
    print(f"DART-Vetter weights: {path}")
    return torch.cat(probs).numpy()


x_tr, y_tr = load_csv(cfg["training_set_path"])
x_te, y_te = load_csv(cfg["test_set_path"])
print(f"train: {len(y_tr)} (planets {y_tr.sum()}), test: {len(y_te)} (planets {y_te.sum()})\n")

rng = np.random.default_rng(SEED)
models = {
    "Random coin": rng.random(len(y_te)),
    "Majority class": np.zeros(len(y_te)),
}
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=5000, class_weight="balanced"))
models["Logistic Regression"] = lr.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED, n_jobs=-1)
models["Random Forest"] = rf.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
try:
    models["DART-Vetter (reproduced)"] = dartvetter_probs(x_te)
except Exception as exc:                          # keep the baselines even if torch fails
    print(f"DART-Vetter skipped: {exc}")

meta = pd.read_csv(cfg["test_set_path"].replace(".csv", "_meta.csv"), dtype={"signal_id": str})
rows = []
for name, p in models.items():
    m = point_metrics(y_te, p)
    ci = bootstrap_ci(y_te, p)
    rows.append({"model": name, **{k: round(v, 3) for k, v in m.items()},
                 **{f"{k} 95% CI": f"[{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in ci.items()}})
    slug = name.lower().split(" (")[0].replace(" ", "_").replace("-", "")
    pd.DataFrame({"signal_id": meta["signal_id"], "label": y_te, "p": p}).to_csv(
        out_dir + f"predictions/test_{slug}.csv", index=False)

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(table[["model", "precision", "recall", "F1", "ROC-AUC", "PR-AUC"]].to_string(index=False))
print("\n95% bootstrap confidence intervals:")
print(table[["model", "F1 95% CI", "ROC-AUC 95% CI", "PR-AUC 95% CI"]].to_string(index=False))
table.to_csv(out_dir + "results/baselines.csv", index=False)
print(f"\nsaved {out_dir}results/baselines.csv")
