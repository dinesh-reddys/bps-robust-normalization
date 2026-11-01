"""Populate the LaTeX report strictly from completed experimental records."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from btp.core import hash_file


def main():
    out, report = ROOT / "results", ROOT / "report"
    summary = json.loads((out / "summary.json").read_text())
    cfg = json.loads((ROOT / "config.json").read_text())
    report.joinpath("tables").mkdir(exist_ok=True)
    report.joinpath("images").mkdir(exist_ok=True)
    for image in (out / "figures").glob("*"):
        shutil.copy2(image, report / "images" / image.name)
    oa = {(r["pipeline"], r["fraction"], r["radius"]): r for r in summary["overall_accuracy"]}
    mca = {(r["pipeline"], r["fraction"], r["radius"]): r for r in summary["mean_class_accuracy"]}
    fmt = lambda x: f"{100*x:.2f}"
    metadata = [json.loads(p.read_text()) for p in sorted((ROOT / "models").glob("*.json"))]
    if len(metadata) != 9: raise ValueError("Nine completed models are required")
    commands = []
    for p in "ABC":
        commands.extend([rf"\newcommand{{\Clean{p}}}{{{fmt(oa[(p,0,0)]['mean'])}}}",
                         rf"\newcommand{{\Primary{p}}}{{{fmt(oa[(p,.02,5)]['mean'])}}}"])
    commands.append(rf"\newcommand{{\ParameterCount}}{{{metadata[0]['parameters']:,}}}")
    (report / "generated_results.tex").write_text("\n".join(commands)+"\n")

    clean = [r"\begin{table}[!ht]\centering", r"\begin{tabular}{lll}", r"\toprule Pipeline & Overall accuracy (\%) & Mean class accuracy (\%)\\\midrule"]
    for p in "ABC":
        a, b = oa[(p,0,0)], mca[(p,0,0)]
        clean.append(f"{p} & ${fmt(a['mean'])} \\pm {fmt(a['sd'])}$ & ${fmt(b['mean'])} \\pm {fmt(b['sd'])}$\\\\")
    clean += [r"\bottomrule\end{tabular}", r"\caption{Clean test performance. Values are mean and sample standard deviation across three training seeds.}\end{table}"]
    (report / "tables/clean.tex").write_text("\n".join(clean))

    noise = [r"\begin{table}[!ht]\centering\small",r"\begin{tabular}{llll}",r"\toprule Replaced points & A accuracy (\%) & B accuracy (\%) & C accuracy (\%)\\\midrule"]
    for f in [0,.01,.02,.05,.1]:
        values = [oa[(p,f,5 if f else 0)] for p in "ABC"]
        line = f"{100*f:g}\\%" + " & " + " & ".join(f"${fmt(v['mean'])} \\pm {fmt(v['sd'])}$" for v in values) + r"\\"
        noise.append(line)
    noise += [r"\bottomrule\end{tabular}",r"\caption{Overall accuracy for outliers at five times the clean radius. Three corruption draws are averaged within each training seed before reporting mean and sample standard deviation across seeds.}\end{table}"]
    (report / "tables/noise.tex").write_text("\n".join(noise))

    bests = [r["best_epoch"] for r in metadata]
    train_secs = [r["training_seconds"] for r in metadata]
    training_text = f"Across the nine fits, selected checkpoints occur between epochs {min(bests)} and {max(bests)}. Total measured training time is {sum(train_secs)/60:.1f} minutes; the median fit takes {statistics.median(train_secs)/60:.1f} minutes. These times describe this machine and the recorded runs. They exclude dependency installation, data download and feature preparation. A model's final epoch need not be its selected checkpoint."
    (report / "training_commentary.tex").write_text(training_text+"\n")

    delta = 100*(oa[("C",.02,5)]["mean"]-oa[("A",.02,5)]["mean"])
    clean_delta = 100*(oa[("C",0,0)]["mean"]-oa[("A",0,0)]["mean"])
    noise_text = f"At two-percent replacement and radius five, C changes mean accuracy relative to A by {delta:+.2f} percentage points. Its clean-data difference is {clean_delta:+.2f} percentage points. The ten-percent result for C is {fmt(oa[('C',.1,5)]['mean'])}\\%, which must be considered alongside its lower-contamination performance. These comparisons measure the specified corruption distribution rather than all possible outliers."
    (report / "noise_commentary.tex").write_text(noise_text+"\n")

    with (out / "geometry_diagnostics.csv").open() as stream: diagnostics=list(csv.DictReader(stream))
    vals=[]
    for p in "ABC":
        selected=[r for r in diagnostics if r["pipeline"]==p and float(r["outlier_fraction"])==.02 and float(r["outlier_radius"])==5]
        vals.append(statistics.mean(float(r["median_inlier_radius_ratio"]) for r in selected))
    mechanism_text = f"At two-percent replacement and radius five, the median main-object radius ratio, averaged over corruption draws, is {vals[0]:.3f} for A, {vals[1]:.3f} for B and {vals[2]:.3f} for C. A ratio near one indicates preserved main-object size; a ratio near 0.2 indicates roughly five-fold shrinkage. Feature drift additionally captures changes in centre, retained points and nearest-neighbour assignments."
    (report / "mechanism_commentary.tex").write_text(mechanism_text+"\n")

    example=json.loads((out/"example.json").read_text())
    index=example["test_index"]
    names=json.loads((out/"protocol.json").read_text())["classes"]
    clean_pred=np.load(out/"predictions/f0.00_r0_s0.npz")
    noise_pred=np.load(out/"predictions/f0.02_r5_s101.npz")
    category=lambda d,p:names[int(d[f"{p}_predictions"][0,index])].replace("_"," ")
    clean_classes=", ".join(f"{p}: {category(clean_pred,p)}" for p in "ABC")
    noisy_classes=", ".join(f"{p}: {category(noise_pred,p)}" for p in "ABC")
    example_text=f"For training seed 13, the selected object's clean predictions are {clean_classes}. With the illustrated corruption, the predictions are {noisy_classes}. These labels are read directly from the saved per-object prediction arrays."
    (report/"example_commentary.tex").write_text(example_text+"\n")
    conclusion=f"For the radius-five, two-percent condition, the observed C-minus-A change is {delta:+.2f} percentage points, with a clean-data change of {clean_delta:+.2f} points. The centring-only control and the radius diagnostics identify which preprocessing change accompanies the classification difference. The high-contamination condition also demonstrates why the percentile choice requires a stated operating range."
    (report/"conclusion_commentary.tex").write_text(conclusion+"\n")
    manifest={"config_sha256":hash_file(ROOT/"config.json"),"summary_sha256":hash_file(out/"summary.json"),
              "metrics_sha256":hash_file(out/"metrics.csv"),"models":metadata,
              "figures":{p.name:hash_file(p) for p in sorted((report/"images").glob('*.pdf'))}}
    (report/"results_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Report source populated from completed experiment results")


if __name__=="__main__": main()
