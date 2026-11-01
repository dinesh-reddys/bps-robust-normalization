"""Download standard ModelNet40 point-cloud files from an ETH research mirror.

The standard Stanford host is retained as the original data source in provenance.
Only public dataset files are requested. All bytes are hashed after download.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "modelnet40_ply_hdf5_2048"
BASE = "https://share.phys.ethz.ch/~gseg/Predator/data/modelnet40_ply_hdf5_2048/"
FILES = ([f"ply_data_train{i}.h5" for i in range(5)]
         + [f"ply_data_test{i}.h5" for i in range(2)]
         + ["shape_names.txt", "train_files.txt", "test_files.txt"]
         + [f"ply_data_train_{i}_id2file.json" for i in range(5)]
         + [f"ply_data_test_{i}_id2file.json" for i in range(2)])


def get_file(name):
    target = DEST / name
    url = BASE + name
    if not target.exists():
        for attempt in range(3):
            try:
                part = target.with_suffix(target.suffix + ".part")
                start = time.perf_counter()
                with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                part.replace(target)
                print(f"downloaded {name}: {target.stat().st_size/1e6:.1f} MB in {time.perf_counter()-start:.1f} s", flush=True)
                break
            except Exception as exc:
                print(f"retry {attempt+1} {name}: {exc}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    digest = hashlib.sha256()
    with target.open("rb") as inp:
        while chunk := inp.read(1024 * 1024):
            digest.update(chunk)
    actual=digest.hexdigest()
    expected_path=ROOT/"results/dataset_provenance.json"
    if expected_path.exists():
        expected={v["file"]:v["sha256"] for v in json.loads(expected_path.read_text())["files"]}
        if name in expected and actual!=expected[name]:
            raise ValueError(f"{name} does not match the published dataset hash")
    return {"file": name, "url": url, "bytes": target.stat().st_size, "sha256": actual}


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(get_file, FILES))
    provenance = {
        "dataset": "ModelNet40 point clouds sampled by PointNet authors",
        "original_url": "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        "mirror": BASE,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": records,
    }
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("download and hashing complete", flush=True)
