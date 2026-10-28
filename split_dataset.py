from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import argparse

"""
split_dataset.py
================
Merges one or more datasets made by build_dataset.py (e.g. parallel shards)
and splits them into train and test files BY STAR, so that no star appears
in both sets (prevents data leakage: one star can have several signals,
and they look alike).
 
Input : <data>.csv (201 values + label, no header) and <data>_meta.csv,
        for every path given in --data
Output: <out>_train.csv and <out>_test.csv (same format, no header)
        + <out>_train_meta.csv / <out>_test_meta.csv for traceability
 
Examples
  python split_dataset.py --data data/kepler_dr24 --out data/kepler_dr24
  python split_dataset.py --data data/kepler_dr24_shard0of4 data/kepler_dr24_shard1of4 \
      data/kepler_dr24_shard2of4 data/kepler_dr24_shard3of4 --out data/kepler_dr24
"""


def load(path):
    views = pd.read_csv(path + ".csv", header=None)
    meta = pd.read_csv(path + "_meta.csv", dtype={"signal_id": str})
    if len(views) != len(meta):
        raise SystemExit(f"{path}: {len(views)} views vs {len(meta)} meta rows")
    return views, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True,
                        help="one or more paths without extension")
    parser.add_argument("--out", required=True, help="output path without extension")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-zero-bins", type=int, default=None,
                        help="quality filter: drop views with more bins exactly equal "
                             "to 0 than this (damaged by detrending, e.g. 10)")
    args = parser.parse_args()

    parts = [load(p) for p in args.data]
    views = pd.concat([v for v, _ in parts], ignore_index=True)
    meta = pd.concat([m for _, m in parts], ignore_index=True)
    # If a signal appears in several files (e.g. rebuilt with fixed code),
    # keep the version from the LAST file given in --data.
    keep_last = ~meta["signal_id"].duplicated(keep="last")
    if (~keep_last).any():
        print(f"replaced {int((~keep_last).sum())} signals by their rebuilt version")
    views = views[keep_last].reset_index(drop=True)
    meta = meta[keep_last].reset_index(drop=True)
    print(f"merged {len(args.data)} file(s): {len(views)} signals, "
          f"{meta['star_id'].nunique()} stars")

    if args.max_zero_bins is not None:
        zero_bins = (views.iloc[:, :-1] == 0).sum(axis=1)
        keep = zero_bins <= args.max_zero_bins
        dropped = meta.loc[~keep, "original_label"].value_counts().to_dict()
        print(f"quality filter (zero bins > {args.max_zero_bins}): "
              f"dropped {int((~keep).sum())} views {dropped}")
        views = views[keep].reset_index(drop=True)
        meta = meta[keep].reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                 random_state=args.seed)
    train_idx, test_idx = next(splitter.split(views, groups=meta["star_id"]))

    for name, idx in (("train", train_idx), ("test", test_idx)):
        views.iloc[idx].to_csv(f"{args.out}_{name}.csv", header=False, index=False)
        meta.iloc[idx].to_csv(f"{args.out}_{name}_meta.csv", index=False)
        labels = views.iloc[idx, -1]
        print(f"{name}: {len(idx)} signals, {meta.iloc[idx]['star_id'].nunique()} stars, "
              f"planets={int(labels.sum())}, non-planets={int((labels == 0).sum())}")

    overlap = set(meta.iloc[train_idx]["star_id"]) & set(meta.iloc[test_idx]["star_id"])
    print(f"stars in both sets: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
