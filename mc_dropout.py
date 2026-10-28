import m1_class
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import yaml
import torch
import pandas as pd
import numpy as np
import os
import copy

"""
mc_dropout.py
=============
Monte Carlo dropout inference for the trained DART-Vetter (Gal & Ghahramani, 2016).

Why: DART-Vetter was always trained with dropout active (dropout -> max-pool ->
BatchNorm in every block). With dropout switched off at test time the network
sees a different signal than in training and its outputs are distorted.
MC dropout keeps dropout ON at test time, runs every signal T times and
averages the T probabilities:
  - the mean   = the prediction (stable, because it is an average)
  - the std    = how much the model disagrees with itself = its uncertainty

BatchNorm stays in eval mode with the ORIGINAL running statistics, which were
collected with dropout active, so they match this setting. Predictions do not
depend on which other signals are in the batch.

Output:
  - metrics on train and test for several T
  - outputs/predictions/<split>_mc.csv with signal_id, label, p_mean, p_std
    (input for baselines, calibration and conformal prediction)

Run from the project root:
  python mc_dropout.py
"""


T_VALUES = (1, 5, 10, 30, 50)    # number of stochastic passes to compare
T_FINAL = 30                     # T used for the saved predictions
cfg = yaml.safe_load(open("config_cnn.yaml"))


def load_csv(path):
    values = np.genfromtxt(path, delimiter=",")
    return torch.tensor(values[:, :-1], dtype=torch.float32), values[:, -1].astype(int)


def mc_dropout_probs(model, x, T, seed=0):
    """Return an array of shape (T, n): T stochastic probabilities per signal."""
    m = copy.deepcopy(model)
    m.eval()                                          # BatchNorm: running statistics
    for layer in m.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.train()                             # dropout ON
    torch.manual_seed(seed)
    passes = []
    with torch.no_grad():
        for _ in range(T):
            out = [torch.sigmoid(m.get_classification_output(
                m.get_feature_extraction_output(b))).squeeze(-1)
                for b in torch.split(x, cfg["batch_size"])]
            passes.append(torch.cat(out).numpy())
    return np.stack(passes)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "P@0.5": precision_score(y, pred, zero_division=0),
            "R@0.5": recall_score(y, pred, zero_division=0),
            "F1@0.5": f1_score(y, pred, zero_division=0)}


model = m1_class.DartVetterFiscale2024(cfg["cbn"], cfg["cfs"], cfg["cbff"], cfg["psz"],
                                       cfg["pst"], cfg["rateval"], cfg["fc_layers_num"],
                                       cfg["fc_units"], cfg["fc_output_size"])
weights = cfg["local_path"] + "trained_models/cnn" + cfg["training_set_name"] + ".pt"
model.load_state_dict(torch.load(weights, map_location="cpu"))
os.makedirs(cfg["local_path"] + "predictions", exist_ok=True)

for split, path in (("train", cfg["training_set_path"]), ("test", cfg["test_set_path"])):
    x, y = load_csv(path)
    print(f"\n{split}: {len(y)} signals, planets={y.sum()}")
    probs = mc_dropout_probs(model, x, max(T_VALUES))

    rows = []
    for T in T_VALUES:
        m = metrics(y, probs[:T].mean(axis=0))
        rows.append({"T": T, **{k: round(v, 3) for k, v in m.items()}})
    print(pd.DataFrame(rows).to_string(index=False))

    p_mean, p_std = probs[:T_FINAL].mean(axis=0), probs[:T_FINAL].std(axis=0)
    wrong = (p_mean >= 0.5).astype(int) != y
    print(f"uncertainty (std over T={T_FINAL}): correct={p_std[~wrong].mean():.3f}, "
          f"wrong={p_std[wrong].mean():.3f}   <- higher on wrong = useful uncertainty")

    meta = pd.read_csv(path.replace(".csv", "_meta.csv"), dtype={"signal_id": str})
    out = cfg["local_path"] + f"predictions/{split}_mc.csv"
    pd.DataFrame({"signal_id": meta["signal_id"], "original_label": meta["original_label"],
                  "label": y, "p_mean": p_mean, "p_std": p_std}).to_csv(out, index=False)
    print(f"saved {out}")
