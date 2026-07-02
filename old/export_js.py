import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import umap

    # X = np.stack(df["FP_map4"].map(np.asarray).values).astype(np.float32)

    # reducer = umap.UMAP(
    # n_components=3,
    # n_neighbors=80,
    # min_dist=0.1,
    # metric="manhattan",
    # random_state=7,
    # n_jobs=1,)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning, module="umap\\.umap_")
    #     xyz = reducer.fit_transform(X)

import json
import numpy as np
from pathlib import Path

def export_js(df):
    coords = np.array(df["coords"].tolist(), dtype=float)
    xyz_centered = coords - coords.mean(axis=0, keepdims=True)

    x = xyz_centered[:, 0]
    y = xyz_centered[:, 1]
    z = xyz_centered[:, 2]

    r = np.sqrt(x * x + y * y + z * z)

    eps = 1e-12
    theta = np.arccos(np.clip(z / np.maximum(r, eps), -1.0, 1.0))
    phi = np.arctan2(y, x)

    sphere_radius = 3.2

    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
    r_web = sphere_radius * np.cbrt(r_norm)

    x_web = r_web * np.sin(theta) * np.cos(phi)
    y_web = r_web * np.sin(theta) * np.sin(phi)
    z_web = r_web * np.cos(theta)

    df_export = df.drop(columns=["coords"])

    points = []
    for i in range(len(df_export)):
        row = df_export.iloc[i].to_dict()

        row["x"] = round(float(x_web[i]), 4)
        row["y"] = round(float(y_web[i]), 4)
        row["z"] = round(float(z_web[i]), 4)

        points.append(row)

    points_js = "window.CHEMSPACE_DATA = " + json.dumps(points, ensure_ascii=False) + ";"
    Path("chemspace_data.js").write_text(points_js, encoding="utf-8")