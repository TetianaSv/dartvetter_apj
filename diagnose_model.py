import m1_class
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import yaml
import torch
import numpy as np
import copy

# recalculates BatchNorm and saves _bnfix.pt
torch.manual_seed(0)
cfg = yaml.safe_load(open("config_cnn.yaml"))


def load_csv(path):
    values = np.genfromtxt(path, delimiter=",")
    return torch.tensor(values[:, :-1], dtype=torch.float32), values[:, -1].astype(int)


def predict(model, x, mode):
    m = copy.deepcopy(model)             # train-mode forward would update BatchNorm stats
    m.train() if mode == "train" else m.eval()
    probs = []
    with torch.no_grad():
        for batch in torch.split(x, cfg["batch_size"]):
            logits = m.get_classification_output(m.get_feature_extraction_output(batch))
            probs.append(torch.sigmoid(logits).squeeze(-1).numpy())
    return np.concatenate(probs)


def recalibrate_batchnorm(model, x_train):
    """Fix for the dropout -> BatchNorm 'variance shift' (Li et al., 2019).
    BatchNorm running statistics were collected while dropout was active; at
    test time dropout is off, so they no longer match. Here we recompute them
    on the training data with dropout OFF. No weights are trained or changed."""
    m = copy.deepcopy(model)
    m.eval()                                          # dropout off
    for layer in m.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.reset_running_stats()
            layer.momentum = None                     # exact average over all batches
            layer.train()                             # only BatchNorm collects statistics
    with torch.no_grad():
        for batch in torch.split(x_train, cfg["batch_size"]):
            m.get_classification_output(m.get_feature_extraction_output(batch))
    m.eval()
    return m


def report(name, y, p):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    pred = (p >= 0.5).astype(int)
    print(f"{name:14s} probs min/med/p90/max = {p.min():.3f}/{np.median(p):.3f}/"
          f"{np.percentile(p, 90):.3f}/{p.max():.3f} | ROC-AUC={roc_auc_score(y, p):.3f} "
          f"PR-AUC={average_precision_score(y, p):.3f} | @0.5: "
          f"P={precision_score(y, pred, zero_division=0):.3f} "
          f"R={recall_score(y, pred, zero_division=0):.3f} "
          f"F1={f1_score(y, pred, zero_division=0):.3f} | best F1={best_f1:.3f} at t={best_t:.2f}")


model = m1_class.DartVetterFiscale2024(cfg["cbn"], cfg["cfs"], cfg["cbff"], cfg["psz"],
                                       cfg["pst"], cfg["rateval"], cfg["fc_layers_num"],
                                       cfg["fc_units"], cfg["fc_output_size"])
weights = cfg["local_path"] + "trained_models/cnn" + cfg["training_set_name"] + ".pt"
model.load_state_dict(torch.load(weights, map_location="cpu"))
print(f"Loaded {weights}\n")

x_train, _ = load_csv(cfg["training_set_path"])
fixed = recalibrate_batchnorm(model, x_train)

for split, path in (("train", cfg["training_set_path"]), ("test", cfg["test_set_path"])):
    x, y = load_csv(path)
    print(f"{split}: {len(y)} signals, planets={y.sum()}")
    for mode in ("eval", "train"):
        report(f"  {mode} mode", y, predict(model, x, mode))
    # Leak check
    # Shuffle the order, predict, then put the predictions back in the original order.
    perm = np.random.default_rng(0).permutation(len(y))
    p_shuffled = np.empty(len(y))
    p_shuffled[perm] = predict(model, x[perm], "train")
    report("  train shuffled", y, p_shuffled)
    report("  eval+BN fix", y, predict(fixed, x, "eval"))
    print()

torch.save(fixed.state_dict(), weights.replace(".pt", "_bnfix.pt"))
print(f"Saved recalibrated model to {weights.replace('.pt', '_bnfix.pt')}")
