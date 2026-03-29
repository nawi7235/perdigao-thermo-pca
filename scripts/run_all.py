#This script runs the entire data processing and analysis pipeline for the Perdigao dataset. It assumes the following structure:
# - scripts/
#   - build_dataset.py
#   - pca_analysis.py
#   - composites.py
#   - nocturnal_analysis.py


from pathlib import Path
import argparse
import subprocess
import sys


def run(cmd):
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing isfs_*.nc files"
    )
    args = parser.parse_args()

    # scripts/directory
    scripts_dir = Path(__file__).resolve().parent

    # repo root
    repo_root = scripts_dir.parent

    #path to t scripts
    build_script = scripts_dir/ "build_dataset.py"
    pca_script = scripts_dir/ "pca_analysis.py"
    composites_script = scripts_dir / "composites.py"
    nocturnal_script = scripts_dir/ "nocturnal_analysis.py"

    # output file path
    processed_file = repo_root / "data" / "processed" / "perdigao_surface_dataset.csv"

    # make sure output directory exists
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        str(build_script),
        "--input-dir", str(Path(args.input_dir).resolve()),
        "--output-file", str(processed_file),
    ])

    run([
        sys.executable,
        str(pca_script),
        "--input-file", str(processed_file),
    ])

    run([
        sys.executable,
        str(composites_script),
        "--input-file", str(processed_file),
    ])

    run([
        sys.executable,
        str(nocturnal_script),
        "--input-file", str(processed_file),
    ])

    print("All analyses complete.")