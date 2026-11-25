from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
DPI = 200
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
HATCHES = ["", "//", "..", "xx", "\\\\", "++", "oo"]
GREYS = ["0.15", "0.45", "0.7", "0.9", "0.3", "0.6"]
KEYS = ["label", "exp", "procs", "threads", "cores", "tile", "block"]


def version_label(row):
    if row["version"] == "v1":
        return "V1"
    if row["version"] == "v2":
        name = f"V2 {row['local_sort']}"
        return f"{name} overlap" if int(row["overlap"]) else name
    return row["version"]


def load(name):
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["label"] = frame.apply(version_label, axis=1)
    frame["cores"] = frame["procs"] * frame["threads"]
    return frame


def medians(frame):
    return frame.groupby(KEYS, as_index=False).agg(
        t_sort=("t_sort", "median"), t_comp=("t_comp", "median"), t_comm=("t_comm", "median")
    )


def style(index):
    return dict(marker=MARKERS[index % len(MARKERS)], linestyle=LINESTYLES[index % len(LINESTYLES)], color="black")


def save(figure, name):
    figure.tight_layout()
    figure.savefig(PLOTS / name, dpi=DPI)
    plt.close(figure)


def best_split(parallel):
    best = parallel.loc[parallel.groupby(["label", "exp", "cores"])["t_sort"].idxmin()]
    return best.sort_values(["label", "cores"])


def sequential_time(summary, exp, label):
    rows = summary[(summary["exp"] == exp) & (summary["label"] == label)]
    return float(rows["t_sort"].min()) if not rows.empty else float("nan")


def plot_scaling(best, exp, t_seq, t_std):
    cores = sorted(best["cores"].unique())
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, (label, group) in enumerate(best.groupby("label")):
        axis.plot(group["cores"], group["t_sort"], label=label, **style(index))
    axis.axhline(t_seq, color="0.4", linestyle=":", label="sequential bitonic")
    axis.axhline(t_std, color="0.6", linestyle="--", label="sequential std::sort")
    axis.set_xscale("log", base=2)
    axis.set_xticks(cores, [str(c) for c in cores])
    axis.set_xlabel("total cores (P x T, best split)")
    axis.set_ylabel("sort time (s)")
    axis.set_title(f"Sort time vs cores, N = 2^{exp}")
    axis.legend()
    save(figure, f"time_vs_cores_{exp}.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, (label, group) in enumerate(best.groupby("label")):
        axis.plot(group["cores"], t_seq / group["t_sort"], label=label, **style(index))
    axis.plot(cores, cores, color="0.5", linestyle=":", label="ideal S(p) = p")
    axis.set_xscale("log", base=2)
    axis.set_xticks(cores, [str(c) for c in cores])
    axis.set_xlabel("total cores p")
    axis.set_ylabel("speedup S(p) = T_seq_bitonic / T(p) (x)")
    axis.set_title(f"Speedup vs cores, N = 2^{exp}")
    axis.legend()
    save(figure, f"speedup_vs_cores_{exp}.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, (label, group) in enumerate(best.groupby("label")):
        axis.plot(group["cores"], t_seq / group["t_sort"] / group["cores"], label=label, **style(index))
    axis.axhline(1.0, color="0.5", linestyle=":", label="ideal E(p) = 1")
    axis.set_xscale("log", base=2)
    axis.set_xticks(cores, [str(c) for c in cores])
    axis.set_xlabel("total cores p")
    axis.set_ylabel("efficiency E(p) = S(p) / p")
    axis.set_title(f"Efficiency vs cores, N = 2^{exp}")
    axis.legend()
    save(figure, f"efficiency_vs_cores_{exp}.png")


def split_name(row):
    return f"{int(row['procs'])}x{int(row['threads'])}"


def plot_splits(parallel, exp):
    largest = parallel["cores"].max()
    at_largest = parallel[parallel["cores"] == largest].sort_values(["label", "procs"], ascending=[True, False])
    labels = sorted(at_largest["label"].unique())
    splits = list(dict.fromkeys(at_largest.apply(split_name, axis=1)))
    width = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, split in enumerate(splits):
        heights = []
        for label in labels:
            rows = at_largest[(at_largest["label"] == label) & (at_largest.apply(split_name, axis=1) == split)]
            heights.append(float(rows["t_sort"].iloc[0]) if not rows.empty else np.nan)
        positions = np.arange(len(labels)) + (index - (len(splits) - 1) / 2) * width
        axis.bar(positions, heights, width, label=f"P x T = {split}", color=GREYS[index % len(GREYS)],
                 hatch=HATCHES[index % len(HATCHES)], edgecolor="black")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_xlabel("version")
    axis.set_ylabel("sort time (s)")
    axis.set_title(f"MPI-only vs hybrid vs OpenMP-only at {largest} cores, N = 2^{exp}")
    axis.legend(title="split")
    save(figure, f"split_comparison_{exp}.png")

    names = [f"{row['label']}\n{split_name(row)}" for _, row in at_largest.iterrows()]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
    axis.bar(positions, at_largest["t_comp"], color="0.35", edgecolor="black", label="comp")
    axis.bar(positions, at_largest["t_comm"], bottom=at_largest["t_comp"], color="0.85", hatch="//",
             edgecolor="black", label="comm")
    axis.set_xticks(positions, names, rotation=60, ha="right", fontsize=7)
    axis.set_xlabel("version and split P x T")
    axis.set_ylabel("time (s)")
    axis.set_title(f"Computation vs communication at {largest} cores, N = 2^{exp}")
    axis.legend()
    save(figure, f"breakdown_{exp}.png")


def plot_tile_sweep(tiles):
    if tiles.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, (exp, group) in enumerate(medians(tiles).groupby("exp")):
        tiled = group[group["tile"] > 0].sort_values("tile")
        untiled = group[group["tile"] == 0]
        axis.plot(tiled["tile"], tiled["t_sort"], label=f"N = 2^{exp}", **style(index))
        if not untiled.empty:
            axis.axhline(float(untiled["t_sort"].iloc[0]), color="0.5", linestyle=LINESTYLES[index % 4],
                         label=f"N = 2^{exp}, untiled")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("tile size (elements, log2 scale)")
    axis.set_ylabel("sort time (s)")
    threads = int(tiles["threads"].iloc[0])
    axis.set_title(f"V1 tile size vs sort time (P = 1, T = {threads})")
    axis.legend()
    save(figure, "tile_size_vs_time.png")


def plot_block_sweep(blocks):
    if blocks.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, (exp, group) in enumerate(medians(blocks).groupby("exp")):
        group = group.sort_values("block")
        axis.plot(group["block"], group["t_sort"], label=f"N = 2^{exp} sort", **style(index))
        axis.plot(group["block"], group["t_comm"], label=f"N = 2^{exp} comm", color="0.5",
                  marker=MARKERS[(index + 3) % len(MARKERS)], linestyle=":")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("overlap block size (elements, log2 scale)")
    axis.set_ylabel("time (s)")
    procs = int(blocks["procs"].iloc[0])
    threads = int(blocks["threads"].iloc[0])
    axis.set_title(f"V2 overlap block size vs time (P = {procs}, T = {threads})")
    axis.legend()
    save(figure, "block_size_vs_time.png")


def write_summary(summary):
    best = summary.loc[summary.groupby(["exp", "label"])["t_sort"].idxmin()].copy()
    baseline = best[best["label"] == "seq_bitonic"].set_index("exp")["t_sort"]
    best["speedup"] = best["exp"].map(baseline) / best["t_sort"]
    best["efficiency"] = best["speedup"] / best["cores"]
    columns = ["exp", "label", "procs", "threads", "cores", "tile", "block", "t_comp", "t_comm", "t_sort", "speedup",
               "efficiency"]
    best[columns].sort_values(["exp", "t_sort"]).to_csv(RESULTS / "summary.csv", index=False, float_format="%.6f")


def main():
    PLOTS.mkdir(exist_ok=True)
    bench = load("bench.csv")
    summary = medians(bench)
    parallel = summary[summary["label"].str.startswith("V")]
    for exp in sorted(summary["exp"].unique()):
        current = parallel[parallel["exp"] == exp]
        plot_scaling(best_split(current), exp, sequential_time(summary, exp, "seq_bitonic"),
                     sequential_time(summary, exp, "seq_stdsort"))
        plot_splits(current, exp)
    plot_tile_sweep(load("tile_sweep.csv"))
    plot_block_sweep(load("block_sweep.csv"))
    write_summary(summary)
    for path in sorted(PLOTS.glob("*.png")):
        print(path.relative_to(ROOT))
    print((RESULTS / "summary.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
