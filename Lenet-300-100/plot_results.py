import matplotlib.pyplot as plt
import pandas as pd
import glob

def plot_accuracy_curves(log_dir="logs"):
    plt.figure(figsize=(7,5))
    for file in sorted(glob.glob(f"{log_dir}/pruned_*.csv")):
        density = file.split("_")[-1].replace(".csv","")
        df = pd.read_csv(file)
        plt.plot(df["iteration"], df["test_acc"], label=f"{density}%")

    dense_df = pd.read_csv(f"{log_dir}/dense_training_log.csv")
    plt.plot(dense_df["iteration"], dense_df["test_acc"], label="100.0", linestyle="--")

    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Pm (weights remaining)")
    plt.title("Test Accuracy vs Training Iterations (LeNet-300-100)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_accuracy_curves()