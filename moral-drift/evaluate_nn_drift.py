"""Compare one frozen cluster-conditioned NN with simulated future moral choices.

Run from moral-drift/ after generate.py and train_cluster_conditioned_nn.py:
    python evaluate_nn_drift.py                # all dilemmas
    python evaluate_nn_drift.py --test-only    # held-out test dilemmas only

Reads:
    data/frozen_nn_predictions.json    fixed NN decisions (-1 stay, +1 swerve)
    data/choices.npz                   simulated choices (0 stay, 1 swerve)

Writes:
    reports/nn_drift_summary.csv
    figures/nn_drift_agreement.png
    figures/nn_drift_new_disagreements.png

These are simulation comparisons, NOT formal CORA verification results.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
COLORS = {"western": "#2a78d6", "eastern": "#eb6834", "southern": "#1baf7a"}


def load_data():
    with (DATA / "frozen_nn_predictions.json").open(encoding="utf-8") as f:
        frozen = json.load(f)
    with np.load(DATA / "choices.npz") as f:
        simulated = {name: f[name] for name in f.files}

    clusters = [str(x) for x in simulated["clusters"]]
    scenarios = [str(x) for x in simulated["scenarios"]]
    ids = [str(x) for x in simulated["dilemma_ids"]]

    if frozen["clusters"] != clusters or frozen["dilemma_ids"] != ids:
        raise ValueError("Cluster or dilemma ordering differs between the two files. Regenerate both with the same data.")

    # Frozen labels are [-1, +1], simulated choices are [0, 1].
    fixed_labels = np.asarray(frozen["labels"], dtype=int)
    if fixed_labels.shape != (len(clusters), len(ids)) or not np.isin(fixed_labels, [-1, 1]).all():
        raise ValueError("Unexpected shape or labels in frozen_nn_predictions.json")
    fixed = ((fixed_labels + 1) // 2).astype(np.int8)

    choices = simulated["choice"]   # [scenario, run, year, cluster, dilemma]
    if (choices.ndim != 5 or choices.shape[0] != len(scenarios)
            or choices.shape[3:] != fixed.shape or not np.isin(choices, [0, 1]).all()):
        raise ValueError("Unexpected shape or labels in choices.npz")
    return scenarios, clusters, choices, fixed


def evaluate(choices, fixed):
    # All comparisons have shape [scenario, run, year, cluster, dilemma].
    mismatch = choices != fixed[None, None, None, :, :]
    initial = mismatch[:, :, :1, :, :]  # each run has its own simulated year-0 preferences

    agreement = (~mismatch).mean(axis=-1)
    # New: initially agreed with NN, but now disagrees.
    new_disagreement = (~initial & mismatch).mean(axis=-1)
    # Resolved: initially disagreed with NN, but now agrees.
    resolved_disagreement = (initial & ~mismatch).mean(axis=-1)
    return agreement, new_disagreement, resolved_disagreement


def plot_metric(values, scenarios, clusters, ylabel, title, path):
    years = np.arange(values.shape[2])
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11, 4), sharey=True, squeeze=False)
    for s, ax in enumerate(axes[0]):
        for i, cluster in enumerate(clusters):
            v = 100 * values[s, :, :, i]
            low, mid, high = np.percentile(v, [10, 50, 90], axis=0)
            color = COLORS.get(cluster, None)
            ax.plot(years, mid, label=cluster, color=color, linewidth=2)
            ax.fill_between(years, low, high, color=color, alpha=0.18)
        ax.set(title=scenarios[s].replace("_", " "), xlabel="Years since deployment")
        ax.grid(alpha=0.2)
        ax.set_xlim(0, years[-1])
    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(clusters), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true",
                        help="Use only the original held-out test dilemmas (459 IDs)")
    args = parser.parse_args()

    scenarios, clusters, choices, fixed = load_data()
    suffix = "_test" if args.test_only else ""
    if args.test_only:
        dilemma_file = ROOT.parent / "moral-classification" / "data" / "dilemmas.jsonl"
        with dilemma_file.open(encoding="utf-8") as f:
            test_ids = {row["id"] for line in f if line.strip()
                        if (row := json.loads(line))["split"] == "test"}
        with (DATA / "frozen_nn_predictions.json").open(encoding="utf-8") as f:
            dilemma_ids = json.load(f)["dilemma_ids"]
        select = np.array([d in test_ids for d in dilemma_ids])
        if not select.any():
            raise ValueError("No test dilemmas matched frozen predictions")
        choices = choices[..., select]
        fixed = fixed[:, select]
    agreement, new, resolved = evaluate(choices, fixed)
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    summary_path = REPORTS / f"nn_drift_summary{suffix}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "cluster", "year", "agreement_mean_pct",
                         "new_disagreement_mean_pct", "resolved_disagreement_mean_pct"])
        for s, scenario in enumerate(scenarios):
            for i, cluster in enumerate(clusters):
                for year in range(choices.shape[2]):
                    writer.writerow([
                        scenario, cluster, year,
                        round(100 * agreement[s, :, year, i].mean(), 3),
                        round(100 * new[s, :, year, i].mean(), 3),
                        round(100 * resolved[s, :, year, i].mean(), 3),
                    ])

    plot_metric(agreement, scenarios, clusters, "Agreement with frozen NN (%)",
                "Frozen neural network vs simulated preferences", FIGURES / f"nn_drift_agreement{suffix}.png")
    plot_metric(new, scenarios, clusters, "Dilemmas newly disagreeing (%)",
                "New disagreements among dilemmas aligned at year 0", FIGURES / f"nn_drift_new_disagreements{suffix}.png")

    print(f"Mean percentages across simulation runs ({fixed.shape[1]} dilemmas):")
    for s, scenario in enumerate(scenarios):
        print(f"\n{scenario}:")
        for i, cluster in enumerate(clusters):
            start = 100 * agreement[s, :, 0, i].mean()
            finish = 100 * agreement[s, :, -1, i].mean()
            newly = 100 * new[s, :, -1, i].mean()
            recovered = 100 * resolved[s, :, -1, i].mean()
            print(f"  {cluster:8s}  agreement year 0: {start:5.1f}%  "
                  f"year {choices.shape[2]-1}: {finish:5.1f}%  "
                  f"new disagreements: {newly:4.1f}%  resolved: {recovered:4.1f}%")

    print(f"\nSaved: {summary_path}")
    print(f"       {FIGURES / f'nn_drift_agreement{suffix}.png'}")
    print(f"       {FIGURES / f'nn_drift_new_disagreements{suffix}.png'}")
    print("These are results from simulated trajectories, not CORA guarantees.")


if __name__ == "__main__":
    main()
