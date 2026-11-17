import pandas as pd
import numpy as np
import sys
import shutil

# clips the value to [−3, 3]

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
