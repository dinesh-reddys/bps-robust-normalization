"""Geometry for the BPS normalization study.

BPS representation: Prokudin, Lassner and Romero, ICCV 2019.
The independent implementation below uses scipy cKDTree. The random ball
basis and MLP dimensions follow the authors' public BPS-MLP example.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree

PIPELINES = {
    "A": "Mean centre and maximum radius",
    "B": "Median centre and maximum radius",
    "C": "Median centre and 95th percentile radius",
}


def make_basis(n=512, radius=1.7, seed=13):
    # Match the authors' RandomState order without changing global RNG state.
    rng = np.random.RandomState(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(size=(n, 1)) ** (1.0 / 3.0)
    return np.asarray(radius * directions * radii, dtype=np.float32)


def normalize(points, pipeline="A", percentile=95.0):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Expected a nonempty N by 3 point cloud")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline {pipeline}")
    centre = np.mean(points, axis=0) if pipeline == "A" else np.median(points, axis=0)
    centred = points - centre
    radii = np.linalg.norm(centred, axis=1)
    scale = float(np.percentile(radii, percentile)) if pipeline == "C" else float(radii.max())
    scale = max(scale, 1e-8)
    return np.asarray(centred / scale, dtype=np.float32), np.asarray(centre), scale


def encode(points, basis):
    distances, _ = cKDTree(points).query(basis, k=1, workers=1)
    return np.asarray(distances, dtype=np.float32)


def replace_outliers(points, fraction, radius, seed, object_index):
    """Replace a fixed number of points by distant points on a spherical shell.

    Clean centre/radius define corruption severity ONLY. They are never given
    to normalization or the classifier. Selections are nested across fractions
    and matched across radii. This is a controlled synthetic stress test.
    """
    if not 0 <= fraction < 1 or radius <= 0:
        raise ValueError("Invalid corruption severity")
    n = len(points)
    count = int(round(n * fraction))
    result = np.asarray(points, dtype=np.float32).copy()
    replaced = np.zeros(n, dtype=bool)
    if count:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(object_index)]))
        selected = rng.permutation(n)[:count]
        directions = rng.normal(size=(n, 3))[:count]
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        clean_centre = np.mean(points, axis=0)
        clean_radius = np.linalg.norm(points - clean_centre, axis=1).max()
        result[selected] = clean_centre + directions * radius * max(float(clean_radius), 1e-8)
        replaced[selected] = True
    return result, replaced


def load_dataset(root):
    import h5py
    data_dir = Path(root) / "data" / "modelnet40_ply_hdf5_2048"
    arrays = []
    for split in ("train", "test"):
        xs, ys = [], []
        paths = sorted(data_dir.glob(f"ply_data_{split}[0-9].h5"))
        if not paths:
            raise FileNotFoundError("Run python scripts/download_data.py first")
        for path in paths:
            with h5py.File(path, "r") as f:
                xs.append(f["data"][:].astype(np.float32))
                ys.append(f["label"][:].reshape(-1).astype(np.int64))
        arrays.extend([np.concatenate(xs), np.concatenate(ys)])
    names = (data_dir / "shape_names.txt").read_text().splitlines()
    # The PointNet 2048-point HDF5 conversion contains 9840 training clouds.
    # Do not confuse its observed count with raw ModelNet CAD split counts.
    if len(arrays[0]) != 9840 or len(arrays[2]) != 2468 or len(names) != 40:
        raise ValueError("Unexpected dataset sizes or class mapping")
    if arrays[0].shape[1:] != (2048, 3) or arrays[2].shape[1:] != (2048, 3):
        raise ValueError("Unexpected point-cloud dimensions")
    return (*arrays, names)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(root):
    return json.loads((Path(root) / "config.json").read_text())
