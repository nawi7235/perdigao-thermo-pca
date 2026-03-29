# This script performs composite analysis of surface states based on principal components from PCA.
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


VAR_LABELS = {
    "tdry": "Air temperature (°C)",
    "rh": "Relative humidity (%)",
    "pres": "Surface pressure (hPa)",
    "wspd": "Wind speed (m s⁻¹)",
    "theta": "Potential temperature (K)",
    "specific_humidity": "Specific humidity (kg kg⁻¹)",
}


def main(input_file: str) -> None:
    input_path = Path(input_file).resolve()
    repo_root = input_path.parents[2]
    # Make sure results/figures and results/tables directories exist...and correct env
    fig_dir = repo_root / "results" / "figures"
    tab_dir = repo_root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    pc_file = tab_dir / "principal_components.csv"

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    pc_df = pd.read_csv(pc_file)
    pc_df["datetime"] = pd.to_datetime(pc_df["datetime"], errors="coerce")
    pc_df = pc_df.dropna(subset=["datetime"]).copy()

    merged = df.merge(pc_df, on="datetime", how="inner")

    # mean diurnal cycle
    diurnal = merged.groupby(merged["datetime"].dt.hour)[
        ["tdry", "rh", "pres", "wspd", "theta", "specific_humidity"]
    ].mean()
    diurnal.to_csv(tab_dir / "mean_diurnal_cycle.csv")

    for var in ["tdry", "rh", "wspd", "theta", "specific_humidity"]:
        plt.figure(figsize=(8, 4))
        plt.plot(diurnal.index, diurnal[var], marker="o")
        plt.xlabel("Hour of day")
        plt.ylabel(VAR_LABELS[var])
        plt.title(f"Mean diurnal cycle of {VAR_LABELS[var]}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f"diurnal_{var}.png", dpi=200)
        plt.close()

    # extreme composites based on principal component 1
    p90 = merged["PC1"].quantile(0.90)
    p10 = merged["PC1"].quantile(0.10)

    high = merged[merged["PC1"] >= p90]
    low = merged[merged["PC1"] <= p10]

    comp = pd.DataFrame({
        "high_PC1": high[["tdry", "rh", "pres", "wspd", "theta", "specific_humidity"]].mean(),
        "low_PC1": low[["tdry", "rh", "pres", "wspd", "theta", "specific_humidity"]].mean()
    })
    comp.to_csv(tab_dir / "pc1_composites.csv")

    # standardized composite so variables with different units can be compared
    comp_norm = (comp - comp.mean(axis=1).values[:, None]) / comp.std(axis=1).values[:, None]
    comp_norm.to_csv(tab_dir / "pc1_composites_standardized.csv")

    plt.figure(figsize=(9, 5))
    x = range(len(comp_norm.index))

    plt.plot(
        x,
        comp_norm["high_PC1"],
        marker="o",
        label="High values of principal component 1"
    )
    plt.plot(
        x,
        comp_norm["low_PC1"],
        marker="o",
        label="Low values of principal component 1"
    )

    plt.xticks(x, [VAR_LABELS[idx] for idx in comp_norm.index], rotation=45, ha="right")
    plt.ylabel("Standardized anomaly")
    plt.title("Standardized composite surface states for high and low values of principal component 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pc1_extreme_composites.png", dpi=200)
    plt.close()

    print("Composite analysis complete.")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to: {tab_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/processed/perdigao_surface_dataset.csv"
    )
    args = parser.parse_args()
    main(args.input_file)