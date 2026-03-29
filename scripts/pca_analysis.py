# PCA analysis of the Perdigao surface dataset to identify dominant modes of variability and relationships between variables.
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
}


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

    features = [
        "tdry",
        "rh",
        "pres",
        "wspd",
        "theta",
        "theta_v",
        "specific_humidity",
    ]

    use = df[["datetime"] + features].dropna().copy()

    X = use[features].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA()
    pcs = pca.fit_transform(Xs)

    pc_df = pd.DataFrame(
        pcs[:, :3],
        columns=["PC1", "PC2", "PC3"]
    ) # nts: only keep first 3 PCs for now since they explain most of the variance and are easier to visualize
    pc_df["datetime"] = use["datetime"].values
    pc_df["hour"] = pc_df["datetime"].dt.hour

    evr = pd.DataFrame({
        "component": np.arange(1, len(features) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    evr.to_csv(tab_dir / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i}" for i in range(1, len(features) + 1)]
    )
    loadings.to_csv(tab_dir / "pca_loadings.csv")

    pc_df.to_csv(tab_dir / "principal_components.csv", index=False)

    #figure 1: Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(features) + 1), pca.explained_variance_ratio_, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance fraction")
    plt.title("Explained variance by principal component")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_scree.png", dpi=200)
    plt.close()

    # figure 2: Loadings
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
    plt.title("Principal component loadings")
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_loadings.png", dpi=200)
    plt.close()

    # figure 3: PC1 vs PC2
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(pc_df["PC1"], pc_df["PC2"], c=pc_df["hour"], s=8)
    plt.colorbar(sc, label="Hour of day")
    plt.xlabel("Principal component 1 (dimensionless)")
    plt.ylabel("Principal component 2 (dimensionless)")
    plt.title("Relationship between the first two principal components")
    plt.tight_layout()
    plt.savefig(fig_dir / "pc1_vs_pc2.png", dpi=200)
    plt.close()

    print("PCA analysis complete.")
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