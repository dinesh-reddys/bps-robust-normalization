"""Use a validation-selected BPS classifier on an XYZ point cloud."""
from pathlib import Path
import json
import numpy as np
import torch
from .core import PIPELINES, encode, hash_file, normalize, read_config
from .model import BPSMLP


class BPSClassifier:
    """Load a saved model from an editable checkout of this repository.

    Args:
        pipeline: A (mean/max), B (median/max), or C (median/P95).
        training_seed: One of the saved training seeds: 13, 37, 73.
        device: cpu, cuda, or auto. CPU inference does not require an NVIDIA GPU.
        root: Repository path, if checkpoints are outside the default checkout.
    """

    def __init__(self, pipeline="C", training_seed=13, device="cpu", root=None):
        self.root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.config=read_config(self.root)
        if pipeline not in PIPELINES:
            raise ValueError("pipeline must be A, B or C")
        if training_seed not in self.config["training_seeds"]:
            raise ValueError("No checkpoint for this training seed")
        if device=="auto":device="cuda" if torch.cuda.is_available() else "cpu"
        if device=="cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        self.device=torch.device(device)
        self.pipeline=pipeline
        self.training_seed=training_seed
        self.classes=json.loads((self.root/"results/protocol.json").read_text())["classes"]
        self.basis=np.load(self.root/"results/basis.npy",allow_pickle=False)
        checkpoint=self.root/f"models/{pipeline}_seed{training_seed}.pt"
        metadata=json.loads(checkpoint.with_suffix(".json").read_text())
        if metadata["checkpoint_sha256"]!=hash_file(checkpoint):
            raise ValueError("Checkpoint hash does not match its metadata")
        if metadata["config_sha256"]!=hash_file(self.root/"config.json"):
            raise ValueError("Checkpoint and configuration do not match")
        self.model=BPSMLP(self.config["n_basis"],40,self.config["hidden_sizes"],self.config["dropout"]).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True))
        self.model.eval()

    def predict(self,points):
        """Return the category and model identifier for a finite N by 3 array.

        Training used 2048 canonically oriented CAD surface samples. Predictions
        for arbitrary real scans or other orientations are not validated here.
        """
        points=np.asarray(points,dtype=np.float32)
        normalized,_,_=normalize(points,self.pipeline,self.config["percentile"])
        features=encode(normalized,self.basis)
        with torch.inference_mode():
            logits=self.model(torch.tensor(features[None],device=self.device))[0]
        class_id=int(logits.argmax())
        return {"category":self.classes[class_id],"class_id":class_id,
                "pipeline":self.pipeline,"training_seed":self.training_seed,
                "points":len(points),"device":str(self.device)}


def load_points(path):
    """Read .npy or .npz (with a points key); pickle loading is disabled."""
    path=Path(path)
    if path.suffix==".npy":return np.load(path,allow_pickle=False)
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as data:
            if "points" not in data:raise ValueError("The .npz file must contain a 'points' array")
            return data["points"]
    raise ValueError("Use a .npy array or a .npz archive containing 'points'")
