"""Aggregate repeated experiments and generate report figures from saved outputs."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from btp.core import load_dataset, normalize, replace_outliers, read_config, PIPELINES
CFG = read_config(ROOT)
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
COLORS = {"A": "#4d6475", "B": "#bd6b2b", "C": "#14816a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "figure.dpi": 150,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def rows_csv(name):
    with (OUT / name).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key != "pipeline":
                try: row[key] = float(value)
                except ValueError: pass
    return rows


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=200)
    plt.close(fig)


def aggregate(rows, metric="accuracy"):
    groups = {}
    for row in rows:
        key = (row["pipeline"], row["outlier_fraction"], row["outlier_radius"])
        groups.setdefault(key, {}).setdefault(row["training_seed"], []).append(row[metric])
    answer = []
    for (p, fraction, radius), seeds in sorted(groups.items()):
        # Average corruption draws within each trained model first. Error bars
        # are SD across three independently trained models, not nine iid trials.
        values = np.array([np.mean(v) for _, v in sorted(seeds.items())])
        answer.append({"pipeline": p, "fraction": fraction, "radius": radius,
                       "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                       "per_training_seed": values.tolist()})
    return answer


def main():
    rows = rows_csv("metrics.csv")
    expected = len(CFG["pipelines"])*len(CFG["training_seeds"])*(1+len(CFG["outlier_radii"])*len(CFG["outlier_fractions"])*len(CFG["corruption_seeds"]))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(rows)}")
    oa, mca = aggregate(rows), aggregate(rows, "mean_class_accuracy")
    lookup = {(r["pipeline"], r["fraction"], r["radius"]): r for r in oa}
    summary = {"overall_accuracy": oa, "mean_class_accuracy": mca,
               "aggregation": "Corruption draws averaged within each training seed; mean and sample SD across 3 training seeds.",
               "metric_rows": len(rows)}
    clean = [lookup[(p, 0, 0)] for p in CFG["pipelines"]]
    summary["clean"] = clean
    summary["primary_condition"] = {p: lookup[(p, .02, 5.)] for p in CFG["pipelines"]}
    summary["heavy_condition"] = {p: lookup[(p, .10, 5.)] for p in CFG["pipelines"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    flat = []
    for row in oa:
        flat.append({k: v for k,v in row.items() if k != "per_training_seed"})
    with (OUT / "aggregated_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    fractions = [0] + CFG["outlier_fractions"]
    for ax, radius in zip(axes, CFG["outlier_radii"]):
        for p in CFG["pipelines"]:
            values = [lookup[(p, f, radius if f else 0)] for f in fractions]
            ax.errorbar(np.array(fractions)*100, [100*v["mean"] for v in values],
                        yerr=[100*v["sd"] for v in values], marker="o", ms=4,
                        capsize=3, label=p, color=COLORS[p], linewidth=1.7)
        ax.set_title(f"Radius {radius:g} × clean radius",fontsize=14)
        ax.set_xlabel("Replaced points (%)",fontsize=14)
        ax.tick_params(labelsize=13)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 1, 2, 5, 10])
    axes[0].set_ylabel("Overall accuracy (%)",fontsize=14)
    axes[-1].legend(title="Pipeline", loc="upper right",fontsize=12,title_fontsize=12)
    fig.tight_layout(); save(fig, "accuracy_outliers")

    diag = rows_csv("geometry_diagnostics.csv")
    geometry_summary=[]
    for p in CFG["pipelines"]:
        for f in fractions:
            selected=[r for r in diag if r['pipeline']==p and r['outlier_fraction']==f and (r['outlier_radius']==5 or f==0)]
            geometry_summary.append({'pipeline':p,'fraction':f,
                'inlier_radius_ratio':float(np.mean([r['median_inlier_radius_ratio'] for r in selected])),
                'feature_drift':float(np.mean([r['mean_relative_feature_drift'] for r in selected]))})
    (OUT/'geometry_summary.json').write_text(json.dumps(geometry_summary,indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for p in CFG["pipelines"]:
        subset = [r for r in diag if r["pipeline"] == p and (r["outlier_radius"] == 5 or r["outlier_fraction"] == 0)]
        for ax, metric in zip(axes, ["median_inlier_radius_ratio", "mean_relative_feature_drift"]):
            values = [np.mean([r[metric] for r in subset if r["outlier_fraction"] == f]) for f in fractions]
            ax.plot(np.array(fractions)*100, values, marker="o", label=p, color=COLORS[p], linewidth=1.8)
            ax.set_xlabel("Replaced points (%)")
            ax.set_xticks([0, 1, 2, 5, 10])
    axes[0].set_ylabel("Main-object radius / clean radius")
    axes[0].set_title("Object shrinkage after normalization")
    axes[1].set_ylabel("Relative BPS feature drift")
    axes[1].set_title("Change in the classifier input")
    axes[1].legend(title="Pipeline")
    fig.tight_layout(); save(fig, "mechanism")

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    for p in CFG["pipelines"]:
        histories = rows_csv(f"history_{p}_seed13.csv")
        ax.plot([r["epoch"] for r in histories], [100*r["validation_accuracy"] for r in histories],
                color=COLORS[p], label=p, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Model selection uses held-out training objects")
    ax.legend(title="Pipeline"); fig.tight_layout(); save(fig, "training")

    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    positions = np.arange(3)
    ax.bar(positions, [100*r["mean"] for r in clean], yerr=[100*r["sd"] for r in clean],
           color=[COLORS[p] for p in CFG["pipelines"]], width=.58, capsize=5)
    ax.set_xticks(positions, ["A  Mean / max", "B  Median / max", "C  Median / P95"])
    ax.set_ylabel("Clean overall accuracy (%)"); ax.set_ylim(0, 100)
    for i, row in enumerate(clean): ax.text(i, 100*row["mean"]+2, f"{100*row['mean']:.2f}", ha="center", fontsize=11)
    fig.tight_layout(); save(fig, "clean_accuracy")

    _, _, xte, yte, names = load_dataset(ROOT)
    object_id = int(np.flatnonzero(yte == names.index("chair"))[0])
    cloud = xte[object_id]
    corrupted, mask = replace_outliers(cloud, .02, 5, 101, object_id)
    fig = plt.figure(figsize=(10.6, 6.2))
    for j, p in enumerate(CFG["pipelines"]):
        points = normalize(corrupted, p, CFG["percentile"])[0]
        # Rigid display rotation: ModelNet's vertical y axis becomes plot z.
        plot_points = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
        for row, limit in [(0, 5.8), (1, 1.3)]:
            ax = fig.add_subplot(2, 3, row*3+j+1, projection="3d")
            visible=np.max(np.abs(plot_points),axis=1)<=limit
            ax.scatter(*plot_points[~mask & visible].T, s=1, c=COLORS[p], alpha=.65, rasterized=True)
            ax.scatter(*plot_points[mask & visible].T, s=7, c="#b33030", alpha=.85, rasterized=True)
            for setter in [ax.set_xlim, ax.set_ylim, ax.set_zlim]: setter(-limit, limit)
            ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=45)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if row == 0: ax.set_title(p + "  " + ("Mean / max" if p == "A" else "Median / max" if p == "B" else "Median / P95"))
    fig.subplots_adjust(wspace=0, hspace=0)
    save(fig, "cloud_example")
    cover_points=normalize(cloud,"C",CFG["percentile"])[0]
    cover_points=np.column_stack([cover_points[:,0],-cover_points[:,2],cover_points[:,1]])
    fig=plt.figure(figsize=(3.4,3.4))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(*cover_points.T,s=3,c='#DDECE6',alpha=.9)
    ax.set_axis_off();ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=45)
    fig.savefig(FIG/'cover_object.png',dpi=220,transparent=True,bbox_inches='tight')
    fig.savefig(FIG/'cover_object.pdf',transparent=True,bbox_inches='tight')
    plt.close(fig)
    (OUT / "example.json").write_text(json.dumps({"test_index": object_id, "class": "chair", "fraction": .02, "radius": 5, "corruption_seed": 101,
                                                "selection": "First chair in the standard test-file order, independent of predictions.",
                                                "plot_note": "Top panels show distant outliers; bottom panels zoom to the object. Zoom panels clip distant outliers; no point removal is performed."}, indent=2))

    # A compact schematic is drawn from the actual study specification.
    fig, ax = plt.subplots(figsize=(10, 2.15))
    ax.set_axis_off()
    boxes = [(0.02, "2048 xyz points", "3D object surface"), (.27, "Normalize", "A, B or C"), (.52, "512 BPS distances", "One shared fixed basis"), (.77, "MLP classifier", "40 category scores")]
    for x, title, detail in boxes:
        ax.text(x+.105, .62, title, ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(x+.105, .35, detail, ha="center", va="center", fontsize=10, color="#4d6475", transform=ax.transAxes)
        if x < .77: ax.annotate("", xy=(x+.25, .6), xytext=(x+.225, .6), xycoords="axes fraction", arrowprops={"arrowstyle":"->", "color":"#14816a", "lw":2})
    save(fig, "pipeline")
    fig, ax = plt.subplots(figsize=(5.8,4.1))
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    shape=np.column_stack([.7*np.cos(angles),.45*np.sin(angles)])
    anchors=np.array([[-1.15,.9],[.15,1.05],[1.1,.25],[-.7,-1.0]])
    ax.scatter(*shape.T,s=15,c="#4d6475",label="Object samples")
    ax.scatter(*anchors.T,s=65,facecolors="none",edgecolors="#14816a",linewidths=2,label="Fixed basis points")
    for i,b in enumerate(anchors):
        nearest=shape[np.linalg.norm(shape-b,axis=1).argmin()]
        ax.plot([b[0],nearest[0]],[b[1],nearest[1]],"--",c="#bd6b2b",lw=1.5)
        ax.text(b[0]+.04,b[1]+.06,f"b{i+1}",fontsize=10)
    ax.set_aspect('equal');ax.set_xlim(-1.45,1.45);ax.set_ylim(-1.3,1.4)
    ax.set_xticks([]);ax.set_yticks([]);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.025),ncol=2,fontsize=9,frameon=False)
    ax.set_title('Each reference point contributes one nearest distance',fontsize=10)
    save(fig,'bps_explanation')
    print(json.dumps({"clean": summary["clean"], "primary_condition": summary["primary_condition"], "heavy_condition": summary["heavy_condition"]}, indent=2))


if __name__ == "__main__": main()
