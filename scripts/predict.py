"""Classify an XYZ .npy/.npz file with a pretrained BPS classifier."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from btp.inference import BPSClassifier,load_points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--pipeline",choices=["A","B","C"],default="C")
    parser.add_argument("--training-seed",type=int,choices=[13,37,73],default=13)
    parser.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    args=parser.parse_args()
    try:
        result=BPSClassifier(args.pipeline,args.training_seed,args.device).predict(load_points(args.input))
    except (ValueError,FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
