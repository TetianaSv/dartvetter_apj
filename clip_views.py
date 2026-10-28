import pandas as pd
import numpy as np
import sys
import shutil

"""
clip_views.py
=============
Clips global-view values to [-CLIP, CLIP] and writes new CSV files.

Why: a view is normalised so that its baseline is 0 and its deepest point is -1.
If the dip is tiny and there is a large upward spike (flare, artefact), dividing
by the tiny depth makes the spike enormous (one training view reached 6959).
A single such outlier inflates the BatchNorm running statistics learned during
training, which squeezes every prediction at test time. Real dips (-1) and 99%
of the data (|value| < 2.7) are not affected by clipping at 3.

Writes <name>_clip.csv (+ copies <name>_clip_meta.csv) and lists clipped signals.

Run from the project root:
  python clip_views.py data/kepler_dr24_train data/kepler_dr24_test
"""


CLIP = 3.0

for path in sys.argv[1:]:
    data = np.genfromtxt(path + ".csv", delimiter=",")
    views, labels = data[:, :-1], data[:, -1:]
    meta = pd.read_csv(path + "_meta.csv", dtype={"signal_id": str})
    affected = np.abs(views).max(axis=1) > CLIP
    clipped = np.clip(views, -CLIP, CLIP)
    np.savetxt(path + "_clip.csv", np.hstack([clipped, labels]), delimiter=",", fmt="%.6f")
    shutil.copy(path + "_meta.csv", path + "_clip_meta.csv")
    print(f"{path}: {affected.sum()} of {len(views)} views clipped")
    for i in np.where(affected)[0]:
        print(f"   {meta.loc[i, 'signal_id']:>12s} {meta.loc[i, 'original_label']}  "
              f"max |value| = {np.abs(views[i]).max():.1f}")
