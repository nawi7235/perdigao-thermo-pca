#This is an extension of the PCA analysis that focuses on nocturnal hours.
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


VAR_LABELS = {
    "tdry": "Air temperature (degC)",
    "rh": "Relative humidity (%)",
    "pres": "Surface pressure (hPa)",
    "wspd": "Wind speed (m/s)",
    "theta": "Potential temperature (K)",
    "theta_v": "Virtual potential temperature (K)",
    "specific_humidity": "Specific humidity (kg/kg)",
    "stability_proxy": "Potential temperature anomaly (K)",
}


def add_stability_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is the anomaly of potential temperature relative to the campaign's mean.
    Positive values generally indicate relatively warmer / more mixed states;
    negative values generally indicate relatively cooler / more stable states.
    """
    df = df.copy()
    df["stability_proxy"] = df["theta"] - df["theta"].mean()
    return df


def main(input_file: str) -> None:
    input_path = Path(input_file).resolve()
    repo_root = input_path.parents[2]

    fig_dir = repo_root / "results" / "figures"
    tab_dir = repo_root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    # for the diurnal patterns and nocturnal composites, keep track of the hour of day
    df["hour"] = df["datetime"].dt.hour
    df = add_stability_proxy(df)

    # Keep only nighttime / nocturnal hours
    nocturnal = df[(df["hour"] >= 18) | (df["hour"] <= 6)].copy()

    features = [
        "tdry",
        "rh",
        "pres",
        "wspd",
        "theta",
        "theta_v",
        "specific_humidity",
        "stability_proxy",
    ]

    use = nocturnal[["datetime", "hour"] + features].dropna().copy()

    X = use[features].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    pcs = pca.fit_transform(Xs)

    pc_df = pd.DataFrame(
        pcs[:, :3],
        columns=["PC1_nocturnal", "PC2_nocturnal", "PC3_nocturnal"]
    )
    pc_df["datetime"] = use["datetime"].values
    pc_df["hour"] = use["hour"].values

    #nocturnal principal components
    pc_df.to_csv(tab_dir / "principal_components_nocturnal.csv", index=False)

    # explained variance
    evr = pd.DataFrame({
        "component": np.arange(1, len(features) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    evr.to_csv(tab_dir / "pca_explained_variance_nocturnal.csv", index=False)

    #loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i}" for i in range(1, len(features) + 1)]
    )
    loadings.to_csv(tab_dir / "pca_loadings_nocturnal.csv")

    # Merge for composites
    merged = use.merge(pc_df, on=["datetime", "hour"], how="inner")

    # ---------- figure 1: nocturnal scree plot ----------
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(features) + 1),
        pca.explained_variance_ratio_,
        marker="o"
    )
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance fraction")
    plt.title("Explained variance by principal component during nocturnal hours")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_scree_nocturnal.png", dpi=200)
    plt.close()

    # ---------- figure 2: nocturnal loadings ----------
    plt.figure(figsize=(8, 5))
    plt.imshow(loadings.values[:, :3], aspect="auto")
    plt.colorbar(label="Component loading")
    plt.xticks(
        range(3),
        [
            "Principal component 1",
            "Principal component 2",
            "Principal component 3",
        ]
    )
    plt.yticks(range(len(features)), [VAR_LABELS[f] for f in features])
    plt.title("Nocturnal principal component loadings")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_loadings_nocturnal.png", dpi=200)
    plt.close()

    # ---------- figure 3: nocturnal PC1 vs hour ----------
    pc1_hour = pc_df.groupby("hour")["PC1_nocturnal"].mean()

    plt.figure(figsize=(8, 4))
    plt.plot(pc1_hour.index, pc1_hour.values, marker="o")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean principal component 1")
    plt.title("Mean nocturnal principal component 1 by hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pc1_nocturnal_by_hour.png", dpi=200)
    plt.close()

    # ---------- figure 4: nocturnal PC1 vs PC2 ----------
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        pc_df["PC1_nocturnal"],
        pc_df["PC2_nocturnal"],
        c=pc_df["hour"],
        s=8
    )
    plt.colorbar(sc, label="Hour of day")
    plt.xlabel("Nocturnal principal component 1")
    plt.ylabel("Nocturnal principal component 2")
    plt.title("Relationship between the first two nocturnal principal components")
    plt.tight_layout()
    plt.savefig(fig_dir / "pc1_vs_pc2_nocturnal.png", dpi=200)
    plt.close()

    # ---------- Nocturnal extreme composites ----------
    p90 = merged["PC1_nocturnal"].quantile(0.90)
    p10 = merged["PC1_nocturnal"].quantile(0.10)

    high = merged[merged["PC1_nocturnal"] >= p90]
    low = merged[merged["PC1_nocturnal"] <= p10]

    comp = pd.DataFrame({
        "high_nocturnal_PC1": high[
            ["tdry", "rh", "pres", "wspd", "theta", "specific_humidity", "stability_proxy"]
        ].mean(),
        "low_nocturnal_PC1": low[
            ["tdry", "rh", "pres", "wspd", "theta", "specific_humidity", "stability_proxy"]
        ].mean()
    })
    comp.to_csv(tab_dir / "pc1_composites_nocturnal.csv")

    # Standardized comparison across variables with different units
    comp_norm = (
        comp - comp.mean(axis=1).values[:, None]
    ) / comp.std(axis=1).replace(0, np.nan).values[:, None]
    comp_norm.to_csv(tab_dir / "pc1_composites_nocturnal_standardized.csv")

    plt.figure(figsize=(9, 5))
    x = range(len(comp_norm.index))

    plt.plot(
        x,
        comp_norm["high_nocturnal_PC1"],
        marker="o",
        label="High values of nocturnal principal component 1"
    )
    plt.plot(
        x,
        comp_norm["low_nocturnal_PC1"],
        marker="o",
        label="Low values of nocturnal principal component 1"
    )

    plt.xticks(x, [VAR_LABELS[idx] for idx in comp_norm.index], rotation=45, ha="right")
    plt.ylabel("Standardized anomaly")
    plt.title("Standardized nocturnal composite surface states for high and low values of principal component 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pc1_extreme_composites_nocturnal.png", dpi=200)
    plt.close()

    # ---------- get a summary table ----------
    summary = pd.DataFrame({
        "metric": [
            "number_of_nocturnal_samples",
            "pc1_explained_variance",
            "pc2_explained_variance",
            "pc3_explained_variance",
        ],
        "value": [
            len(use),
            pca.explained_variance_ratio_[0],
            pca.explained_variance_ratio_[1],
            pca.explained_variance_ratio_[2],
        ]
    })
    summary.to_csv(tab_dir / "nocturnal_pca_summary.csv", index=False)

    print("Nocturnal analysis complete.")
    print(f"Nocturnal samples used: {len(use)}")
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