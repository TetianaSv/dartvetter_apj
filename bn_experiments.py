import m1_class
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import yaml
import torch
import numpy as np
import copy

"""
bn_experiments.py
=================
Which BatchNorm statistics make the trained DART-Vetter work at test time?

Facts so far (test set):
  eval, stored BN statistics, dropout off ............ ROC-AUC 0.844
  BN statistics recomputed with dropout OFF .......... ROC-AUC 0.859
  MC dropout, stored BN statistics ................... ROC-AUC 0.840
  batch statistics + dropout on (train mode) ......... ROC-AUC 0.940 (shuffled too)

Hypothesis: the stored statistics are stale (an exponential moving average
over the last training batches while the weights were still changing).
Test: recompute them precisely on the TRAINING data with dropout ON
("Precise BN"), then predict with MC dropout. No weights are trained.

Also checks how the batch-statistics result depends on the batch size:
a method whose output depends on the other signals in the batch cannot
classify a single new signal.

Run from the project root:
  python bn_experiments.py
"""


cfg = yaml.safe_load(open("config_cnn.yaml"))
BS = cfg["batch_size"]
T = 30


def load_csv(path):
    v = np.genfromtxt(path, delimiter=",")
    return torch.tensor(v[:, :-1], dtype=torch.float32), v[:, -1].astype(int)


def set_modes(m, dropout_on, bn_batch_stats):
    m.eval()
    for layer in m.modules():
        if isinstance(layer, torch.nn.Dropout) and dropout_on:
            layer.train()
        if isinstance(layer, torch.nn.BatchNorm1d) and bn_batch_stats:
            layer.train()


def forward_probs(m, x, bs):
    with torch.no_grad():
        return torch.cat([torch.sigmoid(m.get_classification_output(
            m.get_feature_extraction_output(b))).squeeze(-1)
            for b in torch.split(x, bs)]).numpy()


def precise_bn(model, x_train, dropout_on, passes=5, seed=0):
    """Recompute BatchNorm running statistics as an exact average over the
    shuffled training data (several passes when dropout is on)."""
    m = copy.deepcopy(model)
    set_modes(m, dropout_on=dropout_on, bn_batch_stats=True)
    for layer in m.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.reset_running_stats()
            layer.momentum = None
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for _ in range(passes if dropout_on else 1):
        forward_probs(m, x_train[rng.permutation(len(x_train))], BS)
    return m


def predict(m, x, dropout_on, bn_batch_stats=False, bs=BS, seed=0):
    m = copy.deepcopy(m)
    set_modes(m, dropout_on, bn_batch_stats)
    torch.manual_seed(seed)
    if not dropout_on:
        return forward_probs(m, x, bs)
    return np.mean([forward_probs(m, x, bs) for _ in range(T)], axis=0)


def line(name, y, p):
    best = max(f1_score(y, (p >= t).astype(int), zero_division=0)
               for t in np.linspace(0.01, 0.99, 99))
    print(f"  {name:44s} ROC-AUC={roc_auc_score(y, p):.3f}  PR-AUC={average_precision_score(y, p):.3f}"
          f"  F1@0.5={f1_score(y, (p >= 0.5).astype(int), zero_division=0):.3f}  best F1={best:.3f}")


model = m1_class.DartVetterFiscale2024(cfg["cbn"], cfg["cfs"], cfg["cbff"], cfg["psz"],
                                       cfg["pst"], cfg["rateval"], cfg["fc_layers_num"],
                                       cfg["fc_units"], cfg["fc_output_size"])
weights = cfg["local_path"] + "trained_models/cnn" + cfg["training_set_name"] + ".pt"
model.load_state_dict(torch.load(weights, map_location="cpu"))

x_tr, y_tr = load_csv(cfg["training_set_path"])
x_te, y_te = load_csv(cfg["test_set_path"])
pbn_on = precise_bn(model, x_tr, dropout_on=True)
pbn_off = precise_bn(model, x_tr, dropout_on=False)

print(f"\nTEST ({len(y_te)} signals, planets={y_te.sum()})")
print(" A. Stored statistics (reference)")
line("eval, dropout off", y_te, predict(model, x_te, dropout_on=False))
line(f"MC dropout T={T}", y_te, predict(model, x_te, dropout_on=True))
print(" B. Precise BN on training data")
line("precise BN (dropout off) + dropout off", y_te, predict(pbn_off, x_te, dropout_on=False))
line("precise BN (dropout ON)  + dropout off", y_te, predict(pbn_on, x_te, dropout_on=False))
line(f"precise BN (dropout ON)  + MC dropout T={T}", y_te, predict(pbn_on, x_te, dropout_on=True))
print(" C. Batch statistics (depends on the other signals in the batch)")
perm = np.random.default_rng(0).permutation(len(y_te))
for bs in (8, 32, 128, len(y_te)):
    p = np.empty(len(y_te))
    p[perm] = predict(model, x_te[perm], dropout_on=True, bn_batch_stats=True, bs=bs)
    line(f"batch stats + MC dropout, batch size {bs}", y_te, p)

torch.save(pbn_on.state_dict(), weights.replace(".pt", "_precisebn.pt"))
print(f"\nSaved {weights.replace('.pt', '_precisebn.pt')}")
