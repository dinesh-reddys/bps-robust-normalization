"""Reproduce preprocessing, clean training and controlled outlier evaluation.

Examples:
  .venv/bin/python scripts/run_study.py prepare
  .venv/bin/python scripts/run_study.py train
  .venv/bin/python scripts/run_study.py evaluate
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from btp.core import PIPELINES, encode, hash_file, load_dataset, make_basis, normalize, read_config, replace_outliers

CFG = read_config(ROOT)
OUT = ROOT / "results"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
for path in (OUT, CACHE, MODELS):
    path.mkdir(exist_ok=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def encode_many(clouds, pipeline, basis, n_workers=8):
    def work(cloud):
        normalized, _, _ = normalize(cloud, pipeline, CFG["percentile"])
        return encode(normalized, basis)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return np.stack(list(pool.map(work, clouds)))


def prepare():
    from sklearn.model_selection import train_test_split
    xtr, ytr, xte, yte, names = load_dataset(ROOT)
    # Create validation exclusively from the official training split.
    train_ids, val_ids = train_test_split(
        np.arange(len(ytr)), test_size=CFG["validation_fraction"],
        stratify=ytr, random_state=CFG["split_seed"])
    np.savez(OUT / "split_indices.npz", train=train_ids, validation=val_ids)
    basis = make_basis(CFG["n_basis"], CFG["basis_radius"], CFG["basis_seed"])
    np.save(OUT / "basis.npy", basis)
    feature_spec = {key: CFG[key] for key in ["n_points", "n_basis", "basis_radius", "basis_seed", "percentile"]}
    feature_spec["dataset_provenance_sha256"] = hash_file(ROOT / "data/modelnet40_ply_hdf5_2048/provenance.json")
    manifest_path = CACHE / "feature_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != feature_spec:
        raise RuntimeError("Feature settings or source data changed. Move existing feature caches before preparing a new protocol.")
    save_json(manifest_path, feature_spec)
    timings = []
    for pipeline in CFG["pipelines"]:
        path = CACHE / f"clean_{pipeline}.npz"
        start = time.perf_counter()
        if not path.exists():
            train_features = encode_many(xtr, pipeline, basis)
            # Test features are deterministic transformations, not used for model selection.
            test_features = encode_many(xte, pipeline, basis)
            np.savez_compressed(path, train=train_features, test=test_features)
        timings.append({"pipeline": pipeline, "prepare_seconds_this_run": time.perf_counter()-start})
        print(f"Prepared pipeline {pipeline}: {timings[-1]}", flush=True)
    provenance_path = ROOT / "data/modelnet40_ply_hdf5_2048/provenance.json"
    save_json(OUT / "protocol.json", {
        "config": CFG, "config_sha256": hash_file(ROOT / "config.json"),
        "dataset_provenance_sha256": hash_file(provenance_path),
        "train_objects": len(train_ids), "validation_objects": len(val_ids),
        "test_objects": len(yte), "classes": names,
        "split_overlap": int(len(set(train_ids) & set(val_ids))),
        "basis_sha256": hash_file(OUT / "basis.npy"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preparation_timings": timings,
    })


def train(retrain=False):
    import torch
    from torch.nn import functional as F
    from btp.model import BPSMLP, seed_everything
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ytr, _, _, _ = load_dataset(ROOT)
    splits = np.load(OUT / "split_indices.npz")
    save_json(OUT / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "deterministic_algorithms": True, "tf32": False,
        "threads": torch.get_num_threads(),
    })
    labels = torch.tensor(ytr, dtype=torch.long, device=device)
    summary = []
    for seed in CFG["training_seeds"]:
        for pipeline in CFG["pipelines"]:
            checkpoint = MODELS / f"{pipeline}_seed{seed}.pt"
            metadata_path = MODELS / f"{pipeline}_seed{seed}.json"
            if checkpoint.exists() and metadata_path.exists() and not retrain:
                existing = json.loads(metadata_path.read_text())
                if existing["config_sha256"] != hash_file(ROOT / "config.json"):
                    raise RuntimeError("Saved model belongs to a different configuration")
                summary.append(existing)
                print(f"Reusing completed model {checkpoint.name}", flush=True)
                continue
            seed_everything(seed)
            features = torch.tensor(np.load(CACHE / f"clean_{pipeline}.npz")["train"], device=device)
            train_ids = torch.tensor(splits["train"], dtype=torch.long, device=device)
            val_ids = torch.tensor(splits["validation"], dtype=torch.long, device=device)
            model = BPSMLP(CFG["n_basis"], 40, CFG["hidden_sizes"], CFG["dropout"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
            shuffle = np.random.default_rng(seed + 10000)
            best_key, best_epoch = (-1.0, float("-inf")), 0
            histories = []
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            for epoch in range(1, CFG["max_epochs"] + 1):
                if epoch == CFG["learning_rate_drop_epoch"]:
                    for group in optimizer.param_groups:
                        group["lr"] = CFG["final_learning_rate"]
                model.train()
                ordered = train_ids[torch.tensor(shuffle.permutation(len(train_ids)), device=device)]
                loss_sum, correct = 0.0, 0
                for ids in ordered.split(CFG["batch_size"]):
                    if len(ids) == 1:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features[ids])
                    loss = F.cross_entropy(logits, labels[ids])
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(ids)
                    correct += int((logits.detach().argmax(1) == labels[ids]).sum())
                if epoch == 1 or epoch % CFG["eval_every"] == 0:
                    model.eval()
                    with torch.inference_mode():
                        logits = model(features[val_ids])
                        val_loss = float(F.cross_entropy(logits, labels[val_ids]))
                        val_acc = float((logits.argmax(1) == labels[val_ids]).float().mean())
                    row = {"epoch": epoch, "train_loss": loss_sum/len(train_ids),
                           "train_accuracy": correct/len(train_ids), "validation_loss": val_loss,
                           "validation_accuracy": val_acc, "learning_rate": optimizer.param_groups[0]["lr"]}
                    histories.append(row)
                    key = (val_acc, -val_loss)
                    if key > best_key:
                        best_key, best_epoch = key, epoch
                        torch.save(model.state_dict(), checkpoint)
                    if epoch % 100 == 0 or epoch == 1:
                        print(f"{pipeline} seed {seed} epoch {epoch}: validation {val_acc:.4f}, best {best_key[0]:.4f}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
                    if epoch - best_epoch >= CFG["early_stopping_patience_epochs"]:
                        break
            if device.type == "cuda":
                torch.cuda.synchronize()
            history_path = OUT / f"history_{pipeline}_seed{seed}.csv"
            write_csv(history_path, histories)
            metadata = {"pipeline": pipeline, "seed": seed,
                        "best_epoch": best_epoch, "epochs_run": epoch,
                        "best_validation_accuracy": best_key[0],
                        "best_validation_loss": -best_key[1],
                        "training_seconds": time.perf_counter()-start,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "checkpoint_sha256": hash_file(checkpoint),
                        "config_sha256": hash_file(ROOT / "config.json")}
            save_json(metadata_path, metadata)
            summary.append(metadata)
            write_csv(OUT / "training_summary.csv", summary)
            print(f"FINISHED {metadata}", flush=True)
    write_csv(OUT / "training_summary.csv", summary)


def evaluate(recompute=False):
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from btp.model import BPSMLP
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Freeze verification happens before the first final test prediction.
    for pipeline in CFG["pipelines"]:
        for seed in CFG["training_seeds"]:
            metadata = json.loads((MODELS / f"{pipeline}_seed{seed}.json").read_text())
            if metadata["config_sha256"] != hash_file(ROOT / "config.json"):
                raise RuntimeError("Configuration changed after training")
    _, _, xte, yte, _ = load_dataset(ROOT)
    basis = np.load(OUT / "basis.npy")
    models = {}
    checkpoint_fingerprints = {f"{p}_seed{s}": hash_file(MODELS / f"{p}_seed{s}.pt") for p in CFG["pipelines"] for s in CFG["training_seeds"]}
    for pipeline in CFG["pipelines"]:
        models[pipeline] = []
        for seed in CFG["training_seeds"]:
            model = BPSMLP(CFG["n_basis"], 40, CFG["hidden_sizes"], CFG["dropout"]).to(device)
            model.load_state_dict(torch.load(MODELS / f"{pipeline}_seed{seed}.pt", map_location=device, weights_only=True))
            model.eval()
            models[pipeline].append((seed, model))
    clean_features = {p: np.load(CACHE / f"clean_{p}.npz")["test"] for p in CFG["pipelines"]}
    clean_scale = {p: np.array([normalize(x, p, CFG["percentile"])[2] for x in xte]) for p in CFG["pipelines"]}
    clean_predictions, rows, diagnostics = {}, [], []
    predictions_dir = OUT / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    conditions = [(0.0, 0.0, 0)] + [(f, r, s) for r in CFG["outlier_radii"] for f in CFG["outlier_fractions"] for s in CFG["corruption_seeds"]]
    for fraction, radius, corruption_seed in conditions:
        condition = f"f{fraction:.2f}_r{radius:g}_s{corruption_seed}"
        output_file = predictions_dir / (condition + ".npz")
        if output_file.exists() and not recompute:
            saved_record=json.loads(output_file.with_suffix(".json").read_text())
            if saved_record.get("model_checkpoints") != checkpoint_fingerprints:
                raise RuntimeError("Cached predictions belong to different or unrecorded models. Use evaluate --recompute.")
            print(f"Reusing predictions {condition}", flush=True)
            rows.extend(json.loads(output_file.with_suffix(".json").read_text())["metrics"])
            diagnostics.extend(json.loads(output_file.with_suffix(".json").read_text())["diagnostics"])
            if fraction == 0:
                values = np.load(output_file)
                clean_predictions = {p: values[f"{p}_predictions"] for p in CFG["pipelines"]}
            continue
        condition_rows, condition_diag = [], []
        arrays = {"labels": yte}
        corrupted = [replace_outliers(x, fraction, radius or 1.0, corruption_seed, i) for i, x in enumerate(xte)] if fraction else None
        for pipeline in CFG["pipelines"]:
            start = time.perf_counter()
            if fraction:
                def work(item):
                    i, (cloud, mask) = item
                    normalized, _, scale = normalize(cloud, pipeline, CFG["percentile"])
                    features = encode(normalized, basis)
                    # Main-object radius about its own centre isolates the shrinkage factor.
                    clean_norm, _, _ = normalize(xte[i], pipeline, CFG["percentile"])
                    inlier_radius = np.quantile(np.linalg.norm(normalized[~mask] - normalized[~mask].mean(0), axis=1), 0.9)
                    clean_radius = np.quantile(np.linalg.norm(clean_norm[~mask] - clean_norm[~mask].mean(0), axis=1), 0.9)
                    drift = np.linalg.norm(features-clean_features[pipeline][i])/max(float(np.linalg.norm(clean_features[pipeline][i])), 1e-8)
                    return features, scale/clean_scale[pipeline][i], inlier_radius/max(float(clean_radius), 1e-8), drift
                with ThreadPoolExecutor(max_workers=8) as pool:
                    encoded = list(pool.map(work, enumerate(corrupted)))
                features = np.stack([v[0] for v in encoded])
                diag = np.asarray([v[1:] for v in encoded])
            else:
                features = clean_features[pipeline]
                diag = np.column_stack([np.ones(len(xte)), np.ones(len(xte)), np.zeros(len(xte))])
            prep_seconds = time.perf_counter()-start
            tensor = torch.tensor(features, device=device)
            preds = []
            for training_seed, model in models[pipeline]:
                if device.type == "cuda": torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    pred = torch.cat([model(batch).argmax(1) for batch in tensor.split(512)]).cpu().numpy()
                if device.type == "cuda": torch.cuda.synchronize()
                inference_seconds = time.perf_counter()-tick
                preds.append(pred)
                row = {"pipeline": pipeline, "training_seed": training_seed,
                       "outlier_fraction": fraction, "outlier_count": int(round(fraction*CFG["n_points"])),
                       "actual_fraction": int(round(fraction*CFG["n_points"]))/CFG["n_points"],
                       "outlier_radius": radius, "corruption_seed": corruption_seed,
                       "accuracy": accuracy_score(yte, pred),
                       "mean_class_accuracy": balanced_accuracy_score(yte, pred),
                       "test_objects": len(yte),
                       "feature_and_diagnostics_seconds": prep_seconds,
                       "classifier_seconds": inference_seconds}
                condition_rows.append(row)
            predictions = np.stack(preds)
            if fraction == 0:
                clean_predictions[pipeline] = predictions
            condition_diag.append({"pipeline": pipeline, "outlier_fraction": fraction,
                                   "outlier_radius": radius, "corruption_seed": corruption_seed,
                                   "median_scale_ratio": float(np.median(diag[:, 0])),
                                   "median_inlier_radius_ratio": float(np.median(diag[:, 1])),
                                   "mean_relative_feature_drift": float(np.mean(diag[:, 2]))})
            arrays[f"{pipeline}_predictions"] = predictions
            arrays[f"{pipeline}_diagnostics"] = diag
        np.savez_compressed(output_file, **arrays)
        save_json(output_file.with_suffix(".json"), {"metrics": condition_rows, "diagnostics": condition_diag,"model_checkpoints":checkpoint_fingerprints})
        rows.extend(condition_rows)
        diagnostics.extend(condition_diag)
        write_csv(OUT / "metrics.csv", rows)
        write_csv(OUT / "geometry_diagnostics.csv", diagnostics)
        print(f"EVALUATED {condition}: " + ", ".join(f"{p}={np.mean([r['accuracy'] for r in condition_rows if r['pipeline']==p]):.3f}" for p in CFG["pipelines"]), flush=True)
    write_csv(OUT / "metrics.csv", rows)
    write_csv(OUT / "geometry_diagnostics.csv", diagnostics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "all"])
    parser.add_argument("--retrain",action="store_true",help="Train again even if checkpoints are present")
    parser.add_argument("--recompute",action="store_true",help="Recompute predictions even if cached results are present")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"): prepare()
    if args.stage in ("train", "all"): train(args.retrain)
    if args.stage in ("evaluate", "all"): evaluate(args.recompute)
