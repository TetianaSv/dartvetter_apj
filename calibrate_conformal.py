from dfo import dfo_minimise
from sklearn.model_selection import GroupShuffleSplit
from scipy.optimize import minimize
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use("Agg")                     # save figures to files

# Calibration, conformal prediction, Monte Carlo
SEED = 42
ALPHA = 0.05                              # ALPHA = 95% target coverage
N_BINS = 15                               # bins for ECE and reliability diagrams
N_CAL_SIZES = [25, 50, 100, 200, 500]     # TESS calibration sizes for scenario C
N_REPEATS = 200                           # Monte Carlo repetitions for scenario C
EPS = 1e-4                                # keeps probabilities away from exact 0/1
MODELS = {"DART-Vetter": "dartvetter", "Random Forest": "random_forest"}

cfg = yaml.safe_load(open("config_cnn.yaml"))
OUT = cfg["local_path"]
os.makedirs(OUT + "results", exist_ok=True)
os.makedirs(OUT + "plots", exist_ok=True)


# Data: saved probabilities + star ids, split into calibration/evaluation


def load_predictions(slug):
    """Kepler test and TESS probabilities of one model, with star ids."""
    kepler = pd.read_csv(OUT + f"predictions/test_{slug}.csv", dtype={"signal_id": str})
    meta = pd.read_csv(cfg["test_set_path"].replace(".csv", "_meta.csv"),
                       dtype={"signal_id": str})[["signal_id", "star_id"]]
    kepler = kepler.merge(meta, on="signal_id", how="left")
    tess = pd.read_csv(OUT + f"predictions/tess_{slug}.csv", dtype={"signal_id": str})
    return kepler, tess


def split_by_star(df, frac=0.5, seed=SEED):
    """Two halves with no star shared between them."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(df, groups=df["star_id"]))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)

# Calibration metrics


def ece(y, p, n_bins=N_BINS):
    """Expected Calibration Error: split [0, 1] into equal bins; in each bin compare
    the mean predicted probability with the real share of planets; average the
    gaps weighted by how many signals fall in the bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if in_bin.any():
            total += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return total


def brier(y, p):
    return np.mean((p - y) ** 2)


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_row(y, p):
    return {"ECE": ece(y, p), "Brier": brier(y, p), "NLL": nll(y, p)}


# Platt scaling, fitted with DFO

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def platt(p, a, b):
    """New probability = sigmoid(a * logit(old) + b)."""
    return sigmoid(a * logit(p) + b)


def fit_platt(y, p, method="dfo", seed=SEED):
    """Find a, b that minimise the NLL on the calibration data."""
    def loss(ab): return nll(y, platt(p, ab[0], ab[1]))
    if method == "dfo":
        ab, best, history = dfo_minimise(loss, lower=[0.01, -10], upper=[5, 10], seed=seed)
        return ab, best, history
    res = minimize(loss, x0=[1.0, 0.0], method="L-BFGS-B", bounds=[(0.01, 5), (-10, 10)])
    return res.x, res.fun, None


# Split conformal prediction (binary)

def conformal_threshold(y_cal, p_cal, alpha=ALPHA):
    """Nonconformity score = 1 - (probability given to the TRUE class).
    The threshold q is the ceil((n+1)(1-alpha))/n quantile of the calibration
    scores: at most alpha of new scores are expected to be above it."""
    p_true = np.where(y_cal == 1, p_cal, 1 - p_cal)
    scores = 1 - p_true
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")


def prediction_sets(p, q):
    """A class enters the set if its score (1 - its probability) is <= q."""
    has_planet = (1 - p) <= q
    has_not = p <= q
    return has_planet, has_not


def conformal_metrics(y, p, q):
    has_planet, has_not = prediction_sets(p, q)
    covered = np.where(y == 1, has_planet, has_not)
    size = has_planet.astype(int) + has_not.astype(int)
    only_planet = has_planet & ~has_not
    return {"coverage": covered.mean(),
            "coverage planets": covered[y == 1].mean(),
            "coverage non-planets": covered[y == 0].mean(),
            "avg set size": size.mean(),
            "share 'not sure' (both)": (size == 2).mean(),
            "share empty": (size == 0).mean(),
            "planets among confident {planet}": y[only_planet].mean() if only_planet.any() else np.nan}


# Plots

def reliability_curve(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:                          # skip nearly empty bins
            conf.append(p[m].mean())
            acc.append(y[m].mean())
    return conf, acc


def plot_reliability(name, panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (title, y, curves) in zip(np.atleast_1d(axes), panels):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for label, p in curves:
            c, a = reliability_curve(y, p)
            ax.plot(c, a, "o-", label=f"{label} (ECE={ece(y, p):.3f})")
        ax.set(xlabel="predicted probability of planet", ylabel="observed share of planets",
               title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Reliability diagrams - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# Experiments


cal_rows, conf_rows, ncal_rows = [], [], []
dfo_histories = {}

for name, slug in MODELS.items():
    kepler, tess = load_predictions(slug)
    k_cal, k_eval = split_by_star(kepler)           # Kepler test: half to calibrate, half to evaluate
    t_pool, t_eval = split_by_star(tess)            # TESS: calibration pool/evaluation half
    yk_cal, pk_cal = k_cal["label"].values, k_cal["p"].values
    yk_ev, pk_ev = k_eval["label"].values, k_eval["p"].values
    yt_pool, pt_pool = t_pool["label"].values, t_pool["p"].values
    yt_ev, pt_ev = t_eval["label"].values, t_eval["p"].values
    print(f"\n=== {name} ===  Kepler cal/eval {len(k_cal)}/{len(k_eval)}, "
          f"TESS pool/eval {len(t_pool)}/{len(t_eval)}")

    # Calibration
    ab_k, nll_k, hist_k = fit_platt(yk_cal, pk_cal, "dfo")       # fitted on Kepler
    ab_k_sp, nll_k_sp, _ = fit_platt(yk_cal, pk_cal, "scipy")
    ab_t, nll_t, hist_t = fit_platt(yt_pool, pt_pool, "dfo")     # fitted on TESS
    ab_t_sp, nll_t_sp, _ = fit_platt(yt_pool, pt_pool, "scipy")
    dfo_histories[f"{name}, fitted on Kepler"] = hist_k
    dfo_histories[f"{name}, fitted on TESS"] = hist_t
    print(f"Platt on Kepler: DFO a={ab_k[0]:.3f} b={ab_k[1]:.3f} NLL={nll_k:.4f} | "
          f"scipy a={ab_k_sp[0]:.3f} b={ab_k_sp[1]:.3f} NLL={nll_k_sp:.4f}")
    print(f"Platt on TESS:   DFO a={ab_t[0]:.3f} b={ab_t[1]:.3f} NLL={nll_t:.4f} | "
          f"scipy a={ab_t_sp[0]:.3f} b={ab_t_sp[1]:.3f} NLL={nll_t_sp:.4f}")

    variants = [("Kepler eval", "uncalibrated", yk_ev, pk_ev),
                ("Kepler eval", "Platt fitted on Kepler (DFO)", yk_ev, platt(pk_ev, *ab_k)),
                ("TESS eval", "uncalibrated", yt_ev, pt_ev),
                ("TESS eval", "Platt fitted on Kepler (DFO)", yt_ev, platt(pt_ev, *ab_k)),
                ("TESS eval", "Platt fitted on TESS (DFO)", yt_ev, platt(pt_ev, *ab_t))]
    for data, variant, y, p in variants:
        cal_rows.append({"model": name, "data": data, "calibration": variant,
                         **{k: round(v, 4) for k, v in calibration_row(y, p).items()}})

    plot_reliability(name, [
        ("Kepler (evaluation half)", yk_ev,
         [("uncalibrated", pk_ev), ("Platt on Kepler", platt(pk_ev, *ab_k))]),
        ("TESS (evaluation half)", yt_ev,
         [("uncalibrated", pt_ev), ("Platt on Kepler", platt(pt_ev, *ab_k)),
          ("Platt on TESS", platt(pt_ev, *ab_t))])],
        OUT + f"plots/reliability_{slug}.png")

    # Conformal prediction
    q_k = conformal_threshold(yk_cal, pk_cal)
    q_t = conformal_threshold(yt_pool, pt_pool)
    for scenario, y, p, q in [("A. Kepler -> Kepler", yk_ev, pk_ev, q_k),
                              ("B. Kepler -> TESS", yt_ev, pt_ev, q_k),
                              ("C. TESS (whole pool) -> TESS", yt_ev, pt_ev, q_t)]:
        conf_rows.append({"model": name, "scenario": scenario, "threshold q": round(q, 4),
                          **{k: round(v, 3) for k, v in conformal_metrics(y, p, q).items()}})

    # Scenario C with small TESS calibration sets, Monte Carlo repetitions.
    rng = np.random.default_rng(SEED)
    for n in N_CAL_SIZES + [len(t_pool)]:
        covs, sizes = [], []
        for _ in range(N_REPEATS if n < len(t_pool) else 1):
            idx = rng.choice(len(t_pool), size=n, replace=False)
            q = conformal_threshold(yt_pool[idx], pt_pool[idx])
            m = conformal_metrics(yt_ev, pt_ev, q)
            covs.append(m["coverage"])
            sizes.append(m["avg set size"])
        ncal_rows.append({"model": name, "n TESS calibration": n,
                          "coverage mean": round(np.mean(covs), 3),
                          "coverage 5th pct": round(np.percentile(covs, 5), 3),
                          "share of runs >= 95%": round(np.mean(np.array(covs) >= 1 - ALPHA), 3),
                          "avg set size mean": round(np.mean(sizes), 3)})


# Tables, summary plots

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
cal_df, conf_df, ncal_df = pd.DataFrame(cal_rows), pd.DataFrame(conf_rows), pd.DataFrame(ncal_rows)

print("\nPART 1 - CALIBRATION (lower is better)")
print(cal_df.to_string(index=False))
print(f"\nPART 2 - CONFORMAL PREDICTION (target coverage {1 - ALPHA:.0%})")
print(conf_df.to_string(index=False))
print(f"\nScenario C - TESS calibration size ({N_REPEATS} Monte Carlo repeats each)")
print(ncal_df.to_string(index=False))

cal_df.to_csv(OUT + "results/calibration.csv", index=False)
conf_df.to_csv(OUT + "results/conformal.csv", index=False)
ncal_df.to_csv(OUT + "results/conformal_vs_ncal.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name in MODELS:
    d = ncal_df[ncal_df["model"] == name]
    line, = ax.plot(d["n TESS calibration"], d["coverage mean"], "o-",
                    label=f"{name}: TESS-calibrated")
    kt = conf_df[(conf_df["model"] == name) & (conf_df["scenario"] == "B. Kepler -> TESS")]
    ax.axhline(kt["coverage"].iloc[0], ls=":", color=line.get_color(),
               label=f"{name}: Kepler-calibrated")
ax.axhline(1 - ALPHA, color="k", ls="--", lw=1, label="95% target")
ax.set(xscale="log", xlabel="number of TESS signals used for calibration",
       ylabel="coverage on TESS evaluation half", title="Restoring the conformal guarantee on TESS")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "plots/conformal_coverage_vs_ncal.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for label, h in dfo_histories.items():
    ax.plot(h, label=label)
ax.set(xlabel="DFO iteration", ylabel="best NLL in the swarm", yscale="log",
       title="DFO convergence when fitting Platt scaling")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT + "plots/dfo_convergence.png", dpi=150)
plt.close(fig)

print(f"\nsaved tables to {OUT}results/ and figures to {OUT}plots/")
