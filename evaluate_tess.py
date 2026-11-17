
import glob
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# transfer to TESS, data for Table 2
SEED = 42
N_BOOT = 1000
CLIP = 3.0
cfg = yaml.safe_load(open("config_cnn.yaml"))
out_dir = cfg["local_path"]
os.makedirs(out_dir + "results", exist_ok=True)
os.makedirs(out_dir + "predictions", exist_ok=True)


def load_csv(path):
    v = np.genfromtxt(path, delimiter=",")
    return v[:, :-1], v[:, -1].astype(int)


def load_tess():
    views, metas = [], []
    for path in sorted(glob.glob("data/tess_toi_shard*of4.csv")):
        views.append(np.genfromtxt(path, delimiter=","))
        metas.append(pd.read_csv(path.replace(".csv", "_meta.csv"), dtype={"signal_id": str}))
    v, meta = np.vstack(views), pd.concat(metas, ignore_index=True)
    keep = ~meta["signal_id"].duplicated().values
    v, meta = v[keep], meta[keep].reset_index(drop=True)
    x = np.clip(v[:, :-1], -CLIP, CLIP)
    return x, v[:, -1].astype(int), meta


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)}


def ci(y, p, key):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            vals.append(metrics(y[idx], p[idx])[key])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def dartvetter(x):
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
        return torch.cat([torch.sigmoid(model.get_classification_output(
            model.get_feature_extraction_output(b))).squeeze(-1)
            for b in torch.split(xt, cfg["batch_size"])]).numpy()


x_kp, y_kp = load_csv(cfg["training_set_path"])
x_kt, y_kt = load_csv(cfg["test_set_path"])
x_ts, y_ts, meta_ts = load_tess()
print(f"Kepler train {len(y_kp)} (planets {y_kp.mean():.0%}) | Kepler test {len(y_kt)} "
      f"(planets {y_kt.mean():.0%}) | TESS {len(y_ts)} (planets {y_ts.mean():.0%}), "
      f"{meta_ts['star_id'].nunique()} stars\n")

rng = np.random.default_rng(SEED)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
lr.fit(x_kp, y_kp)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                            random_state=SEED, n_jobs=-1).fit(x_kp, y_kp)
models = {
    "Random coin": (rng.random(len(y_kt)), rng.random(len(y_ts))),
    "Majority class": (np.zeros(len(y_kt)), np.zeros(len(y_ts))),
    "Logistic Regression": (lr.predict_proba(x_kt)[:, 1], lr.predict_proba(x_ts)[:, 1]),
    "Random Forest": (rf.predict_proba(x_kt)[:, 1], rf.predict_proba(x_ts)[:, 1]),
}
try:
    models["DART-Vetter"] = (dartvetter(x_kt), dartvetter(x_ts))
except Exception as exc:
    print(f"DART-Vetter skipped: {exc}")

rows = []
for name, (p_kt, p_ts) in models.items():
    mk, mt = metrics(y_kt, p_kt), metrics(y_ts, p_ts)
    rows.append({"model": name,
                 "Kepler ROC-AUC": round(mk["ROC-AUC"], 3), "TESS ROC-AUC": round(mt["ROC-AUC"], 3),
                 "TESS ROC-AUC 95% CI": ci(y_ts, p_ts, "ROC-AUC"),
                 "Kepler F1": round(mk["F1"], 3), "TESS F1": round(mt["F1"], 3),
                 "Kepler precision": round(mk["precision"], 3), "TESS precision": round(mt["precision"], 3),
                 "Kepler recall": round(mk["recall"], 3), "TESS recall": round(mt["recall"], 3),
                 "TESS PR-AUC": round(mt["PR-AUC"], 3)})
    slug = name.lower().replace(" ", "_").replace("-", "")
    pd.DataFrame({"signal_id": meta_ts["signal_id"], "star_id": meta_ts["star_id"],
                  "original_label": meta_ts["original_label"], "label": y_ts, "p": p_ts}
                 ).to_csv(out_dir + f"predictions/tess_{slug}.csv", index=False)

table = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("ROC-AUC (independent of class balance):")
print(table[["model", "Kepler ROC-AUC", "TESS ROC-AUC", "TESS ROC-AUC 95% CI"]].to_string(index=False))
print("\nAt threshold 0.5:")
print(table[["model", "Kepler precision", "TESS precision", "Kepler recall", "TESS recall",
             "Kepler F1", "TESS F1"]].to_string(index=False))
print(f"\nTESS PR-AUC (random level = planet share = {y_ts.mean():.2f}):")
print(table[["model", "TESS PR-AUC"]].to_string(index=False))

print("\nTESS breakdown by original label (share predicted 'planet' at 0.5):")
breakdown = {}
for name in ("Random Forest", "DART-Vetter"):
    if name in models:
        pred = models[name][1] >= 0.5
        breakdown[name] = pd.Series(pred).groupby(meta_ts["original_label"].values).mean().round(3)
if breakdown:
    b = pd.DataFrame(breakdown)
    b["count"] = meta_ts["original_label"].value_counts()
    print(b.to_string())
    print("CP/KP rows = recall on that group; FP/FA rows = false-positive rate on that group")

table.to_csv(out_dir + "results/kepler_vs_tess.csv", index=False)
print(f"\nsaved {out_dir}results/kepler_vs_tess.csv and predictions/tess_*.csv")
