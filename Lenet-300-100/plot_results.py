import matplotlib.pyplot as plt
import pandas as pd
import glob

def plot_accuracy_curves(
        log_dir="logs",
        densities=("100.0", "51.3", "21.1", "13.5", "3.6"),
        interval=100,
        err_interval=1000,
        max_iter=20000,
        ymin=94.0
):
    """
    Plot averaged test accuracy curves across multiple trials for given densities.

    Args:
        log_dir: base logs directory (containing trial_* subfolders)
        densities: list/tuple of densities (as strings like "51.3")
        interval: binning interval for averaging (e.g. 100 iterations)
        err_interval: step interval for error bars (e.g. 1000 iterations)
        max_iter: maximum iteration to plot
    """

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("viridis", len(densities))

    for i, density in enumerate(densities):
        # --- Collect all CSVs for this density across trials
        pattern = f"{log_dir}/trial_*/pruned_{density}.csv"
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[!] No files found for density {density}")
            continue

        # --- Load and align all trials
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df = df[df["iteration"] <= max_iter]
            # Round iteration to the nearest bin
            df["iteration_bin"] = (df["iteration"] // interval) * interval
            dfs.append(df[["iteration_bin", "test_acc"]])

        # --- Concatenate and aggregate
        all_data = pd.concat(dfs, ignore_index=True)
        grouped = all_data.groupby("iteration_bin")["test_acc"]
        mean_acc = grouped.mean()
        min_acc = grouped.min()
        max_acc = grouped.max()

        # --- Plot mean curve
        color = cmap(i)
        plt.plot(mean_acc.index, mean_acc.values, label=f"{density}%", color=color, linewidth=2)

        # --- Plot error bars every err_interval
        error_points = mean_acc.index[::err_interval // interval]
        err_x = error_points
        err_y = mean_acc.loc[error_points]
        yerr_lower = err_y - min_acc.loc[error_points]
        yerr_upper = max_acc.loc[error_points] - err_y
        plt.errorbar(
            err_x, err_y,
            yerr=[yerr_lower, yerr_upper],
            fmt='o', capsize=3, alpha=0.8,
            color=color, ecolor=color, elinewidth=1.2
        )

    # --- Formatting
    plt.xlabel("Training Iterations", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Test Accuracy vs Training Iterations (Averaged Across Trials)", fontsize=13)
    plt.ylim(ymin, None)
    plt.legend(title="Pm (weights remaining)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("plots/figure_3_center_plot.png", dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    plot_accuracy_curves(
        log_dir="one-shot-training-logs/logs",
        densities=["100.0", "51.3", "21.1", "7.0", "3.6", "1.9"],
        interval=100,
        err_interval=1000,
        max_iter=20000
    )
