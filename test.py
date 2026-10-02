#%%
from enum import unique
import json
import numpy as np

with np.load(r"C:\Users\Laika\Desktop\saurav\Projects\CARE_PD CST GCN UPDRS Predictor\assets\datasets\processed\smpl9\3DGait_smpl9_3d_30f_or_longer_per_frame.npz") as data:
    labels = json.loads(data["__UPDRS_GAIT__"].item())
    key = next(iter(labels))
    skeleton = data[key]              # (frames, 9, 3)
    score = labels[key]["UPDRS_GAIT"]
    print(key, skeleton.shape, score)
    print(skeleton[:,1,:])
# %%

with np.load(r"C:\Users\Laika\Desktop\saurav\Projects\CARE_PD CST GCN UPDRS Predictor\assets\datasets\processed\smpl9\BMCLab_smpl9_3d_30f_or_longer_per_frame.npz") as data:
    print(np.unique(data['__UPDRS_GAIT__']))
# %%
