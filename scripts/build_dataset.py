# This script processes the raw ISFS NetCDF files from Perdigao. It does the following:
# extracts vars, compute needed vars, handles NaNs, parses timestamps from filenames, sorts final dataset by time
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import xarray as xr


def saturation_vapor_pressure_hpa(tc: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * tc) / (tc + 243.5))


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["tdry", "rh", "pres", "u", "v", "wspd", "wdir", "rain", "batt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "rh" in df.columns:
        df.loc[(df["rh"] < 0) | (df["rh"] > 100), "rh"] = np.nan
    if "pres" in df.columns:
        df.loc[df["pres"] <= 0, "pres"] = np.nan

    T_c = df["tdry"].to_numpy()
    RH = df["rh"].to_numpy()
    p = df["pres"].to_numpy()

    es = saturation_vapor_pressure_hpa(T_c)
    e = (RH / 100.0) * es

    with np.errstate(invalid="ignore", divide="ignore"): # will err without
        ln_ratio = np.log(np.where(e > 0, e / 6.112, np.nan))
    td = (243.5 * ln_ratio) / (17.67 - ln_ratio) #derived from tetens

    w = 0.622 * e/ (p - e)
    q = w / (1.0+ w)
    T_k = T_c+ 273.15
    theta = T_k *(1000.0/p) ** 0.286
    Tv = T_k * (1.0 + 0.61 * q)
    theta_v = theta * (1.0 + 0.61 * q)

    df["e_hpa"] = e
    df["tdew"] = td
    df["mixing_ratio"] = w
    df["specific_humidity"] = q
    df["theta"] = theta
    df["tv"] = Tv
    df["theta_v"] = theta_v

    return df


def parse_filename_base_time(path: Path):
    """
    Perdigao ISFS files look like this:
    isfs_20170305.120030.nc
    """
    m = re.search(r"isfs_(\d{8})\.(\d{6})\.nc$", path.name)
    if not m:
        raise ValueError(f"Couldn't parse timestamp from filename: {path.name}")

    datestr, timestr = m.groups()
    return pd.to_datetime(datestr + timestr, format="%Y%m%d%H%M%S")


def clean_missing(ds: xr.Dataset, var: str, vals: np.ndarray) -> np.ndarray:
    missing = ds[var].attrs.get("missing_value", None)
    fillv = ds[var].attrs.get("_FillValue", None)

    if missing is not None:
        vals = np.where(vals == missing, np.nan, vals)
    if fillv is not None:
        vals = np.where(vals == fillv, np.nan, vals)
    return vals


def build_datetimes(ds: xr.Dataset, path: Path) -> pd.DatetimeIndex:
    """
    Treating the file's time var as relative seconds within the file. This should
    preserve the right 6-hour chunk timing.
    """
    if "time" not in ds:
        raise ValueError(f"No time variable in {path.name}")

    time_vals = pd.to_numeric(np.asarray(ds["time"].values), errors="coerce")
    finite_time = time_vals[np.isfinite(time_vals)]

    if finite_time.size == 0:
        raise ValueError(f"No finite time values in {path.name}")

    file_base = parse_filename_base_time(path)

    #to relative seconds within the file
    rel_seconds = time_vals - np.nanmin(time_vals)

    timestamps = file_base + pd.to_timedelta(rel_seconds, unit="s")
    timestamps = pd.to_datetime(timestamps, errors="coerce")

    if pd.Series(timestamps).notna().sum() == 0:
        raise ValueError(f"Could not construct timestamps for {path.name}")

    return pd.DatetimeIndex(timestamps)


def open_single_file(path: Path) -> pd.DataFrame:
    ds = xr.open_dataset(path, decode_times=False)

    try:
        timestamps = build_datetimes(ds, path)

        data_dict = {"datetime": pd.to_datetime(timestamps, errors="coerce")}

        for var in ["tdry", "rh", "pres", "u", "v", "wspd", "wdir", "rain", "batt"]:
            if var in ds.variables:
                vals = np.asarray(ds[var].values)
                vals = clean_missing(ds, var, vals)
                data_dict[var] = vals

        df = pd.DataFrame(data_dict)
        df["source_file"] = path.name

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        if df.empty:
            raise ValueError(f"All datetimes are NaT in {path.name}")

        return df

    finally: 
        ds.close()


def main(input_dir: str, output_file: str) -> None:
    input_path = Path(input_dir) #try glob
    files = sorted(input_path.glob("isfs_*.nc"))

    if not files:
        raise FileNotFoundError(f"No files found in {input_dir}")

    print(f"Found {len(files)} files")
    dfs = []

    for i, f in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Reading {f.name}")
        try:
            df = open_single_file(f)
            print(
                f"  success: {len(df)} rows, "
                f"{df['datetime'].min()} to {df['datetime'].max()}"
            )
            dfs.append(df)
        except Exception as exc:
            print(f"  failed: {repr(exc)}")

    if not dfs:
        raise RuntimeError("No files were successfully read.")

    data = pd.concat(dfs, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"])
    data = data.sort_values("datetime").drop_duplicates(subset=["datetime"])

    data = compute_derived(data)

    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Saved {output_path}")
    print(f"Final dataset rows: {len(data)}")
    print(f"Datetime range: {data['datetime'].min()} to {data['datetime'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/processed/perdigao_surface_dataset.csv"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_file)