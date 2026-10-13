"""Plot simulation results produced by sim_cluster*.R as a single 2x3 figure.

Usage:
    python plots.py FILE [FILE ...] [--theta 0.5] [--out DIR] [--name NAME]

    One or more CSVs; they are concatenated, so the per-ratio files can be
    passed directly, e.g.
        python plots.py all_results_N30k_d20_rep1000.csv --name sim_linear
        python plots.py nonlinear_N30000_d20_rep1000.csv --name sim_nonlinear

Input columns: ratio, nt, method ("LinearAdj XGB", "LinearAdjUnpen GLM",
"Debiased NN", ...), theta_hat, sd_est, covered.

Writes DIR/NAME.pdf:
    top row    - KDE of the estimates, one panel per P(T=0)
    bottom row - absolute bias, standard deviation, coverage vs P(T=0)
with a single shared legend at the bottom.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.lines import Line2D

# Compatibility shim: seaborn < 0.13 calls pd.set_option('mode.use_inf_as_na')
# which was removed in pandas 2.0+
try:
    pd.get_option('mode.use_inf_as_na')
except KeyError:
    try:
        import pandas._config.config as _pd_cfg
        _pd_cfg.register_option('mode.use_inf_as_na', False, '')
    except Exception:
        pass

import seaborn as sns

# --- Visual mappings ---
unique_methods = ["LinearAdj", "LinearAdjUnpen", "Debiased"]
method_names = {
    "LinearAdj":      r"$\hat\beta_{\text{LA}}$",
    "LinearAdjUnpen": r"$\hat\beta_{\text{unpen}}$",
    "Debiased":       r"$\hat{\beta}_{\text{debiased}}$",
}
method_colors = {
    "LinearAdj":      "#E63946",
    "LinearAdjUnpen": "#E9C46A",
    "Debiased":       "#457B9D",
}

unique_models = ["XGB", "GLM", "NN"]
model_linestyles = {
    "XGB": "-",
    "GLM": (0, (6, 2.5)),
    "NN":  (0, (7, 2, 1.5, 2)),
}
model_markers = dict(zip(unique_models, ["o", "s", "^"]))

REF_LINE = dict(color="grey", linestyle="--", linewidth=1.5)


def linestyle_for(model, models):
    return "-" if len(models) == 1 else model_linestyles[model]


def legend_handles(methods, models):
    method_handles = [
        Line2D([0], [0], color=method_colors[m], linewidth=2.5,
               label=method_names[m])
        for m in methods
    ]
    # Models → linestyle + marker, neutral grey so colour doesn't interfere
    model_handles = [
        Line2D([0], [0], color="grey", linestyle=linestyle_for(m, models),
               marker=model_markers[m], markersize=7, linewidth=2, label=m)
        for m in models
    ]
    if len(models) <= 1:
        return method_handles
    return ([Line2D([0], [0], color="none", label="— Method —")] + method_handles +
            [Line2D([0], [0], color="none", label="— Model —")] + model_handles)


def summarize(df):
    df = df.assign(err=df["est"] - df["beta_true"])
    summary = (
        df.groupby(["ratio", "method", "learner"])
        .agg(bias=("err", "mean"), std=("est", "std"), coverage=("covered", "mean"))
        .reset_index()
    )
    summary["bias"] = summary["bias"].abs()
    return summary


def draw_kde_row(axes, df, methods, models, ratios):
    beta_true = df["beta_true"].iloc[0]
    for ax, r in zip(axes, ratios):
        df_sub = df[df["ratio"] == r]
        for model in models:
            for method in methods:
                subset = df_sub[(df_sub["learner"] == model) & (df_sub["method"] == method)]
                if subset["est"].notna().sum() < 2:
                    continue
                sns.kdeplot(data=subset, x="est", color=method_colors[method],
                            linestyle=linestyle_for(model, models), linewidth=2.2, ax=ax,
                            warn_singular=False)
        ax.axvline(x=beta_true, **REF_LINE)
        ax.set_title(rf"$P(T=0)={r:g}$", fontsize=17, fontweight="bold")
        ax.set_xlabel(r"$\hat\beta$", fontsize=15)
        ax.set_ylabel("Density" if ax is axes[0] else "", fontsize=15)


def draw_summary_row(axes, summary, methods, models, ratios):
    metrics = ["bias", "std", "coverage"]
    titles = ["Absolute Bias", "Standard Deviation", "Coverage Probability"]

    for ax, metric, title in zip(axes, metrics, titles):
        for method in methods:
            for model in models:
                subset = summary[(summary["method"] == method) &
                                 (summary["learner"] == model)].sort_values("ratio")
                if subset.empty or subset[metric].isna().all():
                    continue
                ax.plot(subset["ratio"], subset[metric],
                        color=method_colors[method],
                        linestyle=linestyle_for(model, models),
                        marker=model_markers[model],
                        linewidth=2.2, markersize=7)

        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.set_xlabel(r"$P(T = 0)$", fontsize=15)
        ax.set_xscale("log")

        # Reference line for coverage
        if metric == "coverage":
            ax.axhline(0.95, **REF_LINE)
            cov_min = summary["coverage"].min()
            ax.set_ylim(min(0.7, cov_min - 0.05), 1)

        ax.set_xticks(ratios)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticklabels([f"{r:g}" for r in ratios])
        ax.set_xticks([], minor=True)

    # Bias and SD panels share the y-axis range
    lo = 0
    hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(lo, hi)
    axes[1].set_ylim(lo, hi)


def plot_combined(df, out_file):
    summary = summarize(df)
    methods = [m for m in unique_methods if m in set(df["method"])]
    models = [m for m in unique_models if m in set(df["learner"])]
    ratios = sorted(df["ratio"].unique())
    if len(ratios) != 3:
        raise ValueError(f"2x3 layout expects 3 values of P(T=0), got {ratios}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax in axes[0, 1:]:
        ax.sharey(axes[0, 0])
        ax.tick_params(labelleft=False)

    draw_kde_row(axes[0], df, methods, models, ratios)
    draw_summary_row(axes[1], summary, methods, models, ratios)

    handles = legend_handles(methods, models)
    plt.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.5)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0), fontsize=13, frameon=True, framealpha=0.9)
    plt.savefig(out_file, dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_sim_cluster_csv(paths, theta):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df[["method", "learner"]] = df["method"].str.split(" ", n=1, expand=True)
    return df.rename(columns={"theta_hat": "est"}).assign(beta_true=theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+",
                        help="result CSV(s) from sim_cluster*.R (globs allowed)")
    parser.add_argument("--theta", type=float, default=0.5, help="true beta (default: 0.5)")
    parser.add_argument("--out", default=".", help="output directory (default: .)")
    parser.add_argument("--name", default="sim_results",
                        help="output file name without extension (default: sim_results)")
    args = parser.parse_args()

    paths = sorted({p for f in args.files for p in (glob.glob(f) or [f])})
    print("Reading:\n  " + "\n  ".join(paths))
    df = load_sim_cluster_csv(paths, args.theta)
    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, f"{args.name}.pdf")
    plot_combined(df, out_file)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
