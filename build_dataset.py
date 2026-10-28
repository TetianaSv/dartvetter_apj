import pandas as pd
import numpy as np
import warnings
import urllib.request
import urllib.parse
import time
import tempfile
import signal
import shutil
import os
import io
import argparse

"""
build_dataset.py
================
Builds a DART-Vetter-style dataset ("global views") from public NASA data.

For every transit signal (TCE / TOI) the script:
  1. downloads the light curve from MAST with lightkurve,
  2. removes slow stellar trends (flatten), keeping the transit itself masked,
  3. folds the light curve on the orbital period,
  4. bins the folded curve into 201 points (the "global view"),
  5. normalises it (baseline = 0, transit depth = -1),
  6. appends one CSV row: 201 numbers + label (1 = planet, 0 = not a planet).

Output files (no header in the data file, because train_test_cnn_class.py
does not skip one):
  <out>.csv          201 values + label per row       -> input for DART-Vetter
  <out>_meta.csv     ids / period / label per row     -> same order as <out>.csv,
                                                         needed for splitting by star
  <out>_failed.csv   signals that could not be processed, with the reason

The script can be stopped and restarted: already processed signals are skipped.

Labels
  Kepler (Q1-Q17 DR24 TCE table, column av_training_set):
      PC  -> 1 (planet candidate)
      AFP -> 0 (astrophysical false positive, e.g. eclipsing binary)
      NTP -> 0 (non-transiting phenomenon, e.g. noise / systematics)
      UNK -> dropped
  TESS (TOI table, column tfopwg_disp):
      CP, KP -> 1 (confirmed / known planet)
      FP, FA -> 0 (false positive / false alarm)
      PC, APC and others -> dropped (not decided yet)

Examples
  python build_dataset.py --mission kepler --n 50  --out data/kepler_test_run
  python build_dataset.py --mission kepler --n 3000 --out data/kepler_dr24
  python build_dataset.py --mission tess   --n 1000 --out data/tess_toi

  Faster: 4 terminals in parallel, same --n/--seed, different --shard:
  python build_dataset.py --mission kepler --n 2000 --out data/kepler_dr24 --shard 0/4
  python build_dataset.py --mission kepler --n 2000 --out data/kepler_dr24 --shard 1/4
  ... (2/4 and 3/4); then merge them with split_dataset.py

Data sources: NASA Exoplanet Archive (TAP service) and MAST, via lightkurve.
"""


N_BINS = 201                      # length of the global view used by DART-Vetter
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TESS_BJD_OFFSET = 2457000.0       # TESS light-curve time is BTJD = BJD - 2457000


# ---------------------------------------------------------------------------
# 1. Catalogue: which signals to process and their labels
# ---------------------------------------------------------------------------

def query_tap(adql):
    """Run an ADQL query on the NASA Exoplanet Archive and return a DataFrame."""
    params = urllib.parse.urlencode({"query": adql, "format": "csv"})
    with urllib.request.urlopen(f"{TAP_URL}?{params}", timeout=300) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


def load_catalogue(mission):
    """Return a table with the same columns for both missions:
    star_id, signal_id, period_days, t0, duration_days, label."""
    if mission == "kepler":
        df = query_tap(
            "select kepid, tce_plnt_num, tce_period, tce_time0bk, tce_duration, "
            "av_training_set from q1_q17_dr24_tce where av_training_set is not null"
        )
        df = df[df["av_training_set"].isin(["PC", "AFP", "NTP"])]
        return pd.DataFrame({
            "star_id": df["kepid"].astype(int),
            "signal_id": df["kepid"].astype(str) + "_" + df["tce_plnt_num"].astype(str),
            "period_days": df["tce_period"],
            "t0": df["tce_time0bk"],                 # already BKJD, same as lightkurve
            "duration_days": df["tce_duration"] / 24.0,
            "label": (df["av_training_set"] == "PC").astype(int),
            "original_label": df["av_training_set"],
        })

    if mission == "tess":
        df = query_tap(
            "select tid, toi, pl_orbper, pl_tranmid, pl_trandurh, tfopwg_disp from toi"
        )
        df = df[df["tfopwg_disp"].isin(["CP", "KP", "FP", "FA"])]
        df = df[df["pl_orbper"] > 0]                 # single-transit TOIs have no period
        return pd.DataFrame({
            "star_id": df["tid"].astype(int),
            "signal_id": df["toi"].astype(str),
            "period_days": df["pl_orbper"],
            "t0": df["pl_tranmid"] - TESS_BJD_OFFSET,  # BJD -> BTJD
            "duration_days": df["pl_trandurh"] / 24.0,
            "label": df["tfopwg_disp"].isin(["CP", "KP"]).astype(int),
            "original_label": df["tfopwg_disp"],
        })

    raise ValueError(f"Unknown mission: {mission}")


# ---------------------------------------------------------------------------
# 2. Light curve: download and clean
# ---------------------------------------------------------------------------

def download_light_curve(mission, star_id, cache_dir):
    """Download all available light curves of one star and stitch them.
    stitch() normalises every quarter/sector to a median of 1 before joining."""
    import lightkurve as lk

    if mission == "kepler":
        search = lk.search_lightcurve(f"KIC {star_id}", mission="Kepler",
                                      author="Kepler", exptime="long")
    else:
        search = lk.search_lightcurve(f"TIC {star_id}", mission="TESS",
                                      author="SPOC", exptime=120)
    if len(search) == 0:
        return None
    collection = search.download_all(download_dir=cache_dir, quality_bitmask="default")
    if collection is None or len(collection) == 0:
        return None
    return collection.stitch().remove_nans()


def odd_window(duration_days, cadence_days):
    """Window for flatten(), in cadences: at least 3x the transit duration and
    at least 1 day, so the transit itself is not removed as a 'trend'."""
    window_days = max(3.0 * duration_days, 1.0)
    n = int(np.ceil(window_days / cadence_days))
    return n + 1 if n % 2 == 0 else n


def detrend(lc, period, t0, duration):
    """Remove slow stellar variability while ignoring in-transit points.

    break_tolerance: flatten() splits the light curve at gaps longer than this
    many cadences. The transit mask itself creates gaps, so the tolerance must
    be longer than one masked window. Otherwise, for short periods (< ~1.5 d),
    the light curve is cut into pieces shorter than the window, which flatten()
    divides by themselves -> flux exactly 1.0 -> flat zero bins in the view.
    Real data gaps (days long) are still split as before.
    """
    mask = lc.create_transit_mask(period=period, transit_time=t0,
                                  duration=2.0 * duration)
    cadence = float(np.nanmedian(np.diff(lc.time.value)))
    gap_tolerance = int(np.ceil(2.0 * duration / cadence)) + 5
    return lc.flatten(window_length=odd_window(duration, cadence), mask=mask,
                      break_tolerance=gap_tolerance)


# ---------------------------------------------------------------------------
# 3. Global view: fold, bin, normalise  (pure numpy, easy to test)
# ---------------------------------------------------------------------------

def fold_phase(time, period, t0):
    """Phase in days, centred on the transit: values in [-period/2, period/2)."""
    return (time - t0 + 0.5 * period) % period - 0.5 * period


def bin_median(phase, flux, period, n_bins=N_BINS):
    """Median flux in n_bins equal-width bins across one full period.
    Empty bins are filled by linear interpolation."""
    edges = np.linspace(-0.5 * period, 0.5 * period, n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        values = flux[idx == b]
        if values.size:
            binned[b] = np.median(values)
    empty = np.isnan(binned)
    if empty.mean() > 0.5:
        raise ValueError(f"{empty.sum()} of {n_bins} bins are empty")
    if empty.any():
        centres = 0.5 * (edges[:-1] + edges[1:])
        binned[empty] = np.interp(centres[empty], centres[~empty], binned[~empty])
    return binned


CLIP = 3.0   # see clip_views.py: one unclipped outlier (6959) broke BatchNorm


def normalise_view(view):
    """Baseline (median) -> 0, deepest point -> -1 (same idea as Astronet),
    then clip to [-CLIP, CLIP] so a tiny dip with a big spike cannot explode."""
    view = view - np.median(view)
    depth = np.abs(np.min(view))
    if not np.isfinite(depth) or depth == 0:
        raise ValueError("flat view, no dip")
    return np.clip(view / depth, -CLIP, CLIP)


def global_view(time, flux, period, t0):
    phase = fold_phase(np.asarray(time, float), period, t0)
    return normalise_view(bin_median(phase, np.asarray(flux, float), period))


# ---------------------------------------------------------------------------
# 4. Main loop with restart support
# ---------------------------------------------------------------------------

class StarTimeout(Exception):
    """Raised when one star takes longer than --timeout seconds."""


def _on_alarm(signum, frame):
    raise StarTimeout("timeout: star took too long (download probably stalled)")


def already_done(out):
    done = set()
    for suffix in ("_meta.csv", "_failed.csv"):
        path = out + suffix
        if os.path.exists(path):
            done |= set(pd.read_csv(path, dtype={"signal_id": str})["signal_id"])
    return done


def append_row(path, row, header):
    pd.DataFrame([row]).to_csv(path, mode="a", index=False,
                               header=header and not os.path.exists(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mission", choices=["kepler", "tess"], required=True)
    parser.add_argument("--n", type=int, default=50,
                        help="number of signals to sample from the catalogue")
    parser.add_argument("--out", required=True,
                        help="output path without extension, e.g. data/kepler_dr24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-cache", action="store_true",
                        help="keep downloaded FITS files (uses a lot of disk space)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="max seconds per star before it is skipped (default 300)")
    parser.add_argument("--shard", default="0/1",
                        help="K/N: process only part K of N (run N terminals in parallel, "
                             "each with its own K = 0..N-1 and the same --n and --seed)")
    parser.add_argument("--ids", default=None,
                        help="CSV with a signal_id column: process exactly these signals "
                             "(ignores --n), e.g. to rebuild a subset with fixed code")
    args = parser.parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))
    if shard_n > 1:                          # each shard writes its own files
        args.out = f"{args.out}_shard{shard_k}of{shard_n}"

    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _on_alarm)   # per-star time limit (macOS / Linux)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Downloading {args.mission} catalogue from the NASA Exoplanet Archive...")
    cat = load_catalogue(args.mission)
    print(f"  {len(cat)} labelled signals, planets: {cat['label'].sum()}, "
          f"non-planets: {(cat['label'] == 0).sum()}")

    if args.ids:
        wanted = set(pd.read_csv(args.ids, dtype={"signal_id": str})["signal_id"])
        sample = cat[cat["signal_id"].isin(wanted)]
    else:
        sample = cat.sample(n=min(args.n, len(cat)), random_state=args.seed)
    # Same sample in every shard; each shard keeps only its own stars.
    sample = sample[sample["star_id"] % shard_n == shard_k]
    done = already_done(args.out)
    todo = sample[~sample["signal_id"].isin(done)]
    print(f"  sampled {len(sample)}, already processed {len(sample) - len(todo)}, "
          f"to do {len(todo)}")

    n_ok = n_fail = 0
    start = time.time()
    # Group by star: one download serves all signals of the same star.
    for i, (star_id, signals) in enumerate(todo.groupby("star_id"), start=1):
        cache_dir = tempfile.mkdtemp(prefix="lk_")
        signal.alarm(args.timeout)              # start the clock for this star
        try:
            lc = download_light_curve(args.mission, star_id, cache_dir)
        except Exception as exc:
            lc, star_error = None, f"download error: {exc}"
        else:
            star_error = "no light curve found"

        for _, s in signals.iterrows():
            try:
                if lc is None:
                    raise RuntimeError(star_error)
                flat = detrend(lc, s.period_days, s.t0, s.duration_days)
                view = global_view(flat.time.value, flat.flux.value,
                                   s.period_days, s.t0)
                with open(args.out + ".csv", "a") as f:
                    f.write(",".join(f"{v:.6f}" for v in view) + f",{int(s.label)}\n")
                append_row(args.out + "_meta.csv", s.to_dict(), header=True)
                n_ok += 1
            except Exception as exc:
                append_row(args.out + "_failed.csv",
                           {"signal_id": s.signal_id, "star_id": star_id,
                            "reason": str(exc)[:200]}, header=True)
                n_fail += 1

        signal.alarm(0)                         # stop the clock
        if not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        elapsed = time.time() - start
        print(f"[{i}/{todo['star_id'].nunique()} stars] ok={n_ok} failed={n_fail} "
              f"elapsed={elapsed / 60:.1f} min", flush=True)

    print(f"Done. {n_ok} views written to {args.out}.csv, {n_fail} failed.")


if __name__ == "__main__":
    main()
