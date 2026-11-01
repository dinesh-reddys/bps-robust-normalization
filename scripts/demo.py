"""Classify one real test cloud with all three saved pipelines.

The chosen default is the first chair in file order, independently of predictions.
"""
import argparse
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from btp.core import load_dataset,replace_outliers
from btp.inference import BPSClassifier


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--index",type=int)
    parser.add_argument("--fraction",type=float,default=.02)
    parser.add_argument("--radius",type=float,default=5.)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    args=parser.parse_args()
    sample=ROOT/"examples/chair.npy"
    if args.index is None and sample.exists():
        metadata=json.loads(sample.with_suffix(".json").read_text())
        cloud=np.load(sample,allow_pickle=False)
        index=metadata["test_index"]
        true_category=metadata["category"]
    else:
        _,_,clouds,labels,names=load_dataset(ROOT)
        index=args.index if args.index is not None else int(np.flatnonzero(labels==names.index('chair'))[0])
        if not 0<=index<len(clouds):parser.error("test index is out of range")
        cloud=clouds[index]
        true_category=names[int(labels[index])]
    noisy,_=replace_outliers(cloud,args.fraction,args.radius,args.seed,index)
    result={"test_index":index,"true_category":true_category,"outlier_fraction":args.fraction,"outlier_radius":args.radius,"training_seed":13,"predictions":{}}
    for pipeline in "ABC":
        model=BPSClassifier(pipeline,13,args.device)
        outputs={}
        for name,points in [('clean',cloud),('corrupted',noisy)]:
            outputs[name]=model.predict(points)["category"]
        result['predictions'][pipeline]=outputs
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
