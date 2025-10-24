import os
import re
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

# Base path
BASE_PATH = "/net/projects/forexd/WP1/02_ImprovedLabels/Data/region_08_buffer_500_ndvi_timeseries_ids_s1dm"

# Disturbance types
disturbance_types = ["wind", "defoliators", "bark_beetle"]

# Regex to capture year
pattern = re.compile(r"(\w+)_(\d{4})_")

# Output dirs
OUTPUT_DIR = "aggregated_timeseries_median"
FIGURES_DIR = "figures_median"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Dictionary to store results
data = {d: {} for d in disturbance_types}


def process_nc(nc_path, year, d_type):
    """Open a NetCDF file and extract the mean time series over space."""
    with xr.open_dataset(nc_path, chunks={"time": 10}) as ds:
        var = list(ds.data_vars)[0]
        arr = ds[var]

        ts = arr.mean(dim=[dim for dim in arr.dims if dim != "time"])
        ts = ts.compute().values
        time = ds["time"].values

        # get the last part before .nc
        filename = os.path.basename(nc_path)
        base = os.path.splitext(filename)[0]        # remove .nc
        last_part = base.split("_")[-1]            # take last part after "_"

        if last_part == "ids":
            key = "ids"
        elif last_part == "s1dm":
            key = "s1dm"
        else:
            return None  # skip files that don't match

        return (year, d_type, key, ts, time)



# ------------------ PROCESS DATA ------------------ #
for d_type in disturbance_types:
    d_path = os.path.join(BASE_PATH, d_type)
    if not os.path.exists(d_path):
        continue

    folders = [f for f in os.listdir(d_path) if pattern.match(f)]
   
    for folder in tqdm(folders, desc=f"Processing {d_type}"):
        match = pattern.match(folder)
        if not match:
            continue

        _, year = match.groups()
        year = int(year)
        folder_path = os.path.join(d_path, folder)

        nc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".nc")]
        nc_files = nc_files[:10]   # keep only 10 files for testing

        if len(nc_files) < 2:
            continue

        results = Parallel(n_jobs=2)(delayed(process_nc)(nc, year, d_type) for nc in nc_files)
    
        if year not in data[d_type]:
            data[d_type][year] = {"ids": [], "s1dm": []}

        # Store time + values together
        for year, d_type, key, ts, time in results:
            data[d_type][year][key].append({"time": time, "values": ts})


# ------------------ EXPORT & PLOT ------------------ #
for d_type in data:
    for year in data[d_type]:
        # Build a global weekly time axis
        all_times = []
        for key in ["ids", "s1dm"]:
            for entry in data[d_type][year][key]:
                all_times.append(pd.to_datetime(entry["time"]))
        start = min([t.min() for t in all_times])
        end = max([t.max() for t in all_times])
        weekly_index = pd.date_range(start=start, end=end, freq="W")

        # Data for aggregated + individual
        out_dict = {"time": weekly_index}
        individual_df = pd.DataFrame({"time": weekly_index})

        for key in ["ids", "s1dm"]:
            aligned_series = []
            for i, entry in enumerate(data[d_type][year][key]):
                t = pd.to_datetime(entry["time"])
                ts = entry["values"]

                df_ts = pd.DataFrame({"time": t, "val": ts}).set_index("time")
                df_ts = df_ts.reindex(weekly_index)
                aligned_series.append(df_ts["val"].values)

                # add as individual column
                individual_df[f"{key}_{i}"] = df_ts["val"].values

            if len(aligned_series) > 0:
                arrs = np.vstack(aligned_series)
                out_dict[f"{key}_median"] = np.nanmedian(arrs, axis=0)
                q25 = np.nanpercentile(arrs, 25, axis=0)
                q75 = np.nanpercentile(arrs, 75, axis=0)
                out_dict[f"{key}_q25"] = q25
                out_dict[f"{key}_q75"] = q75
            else:
                out_dict[f"{key}_median"] = np.full(len(weekly_index), np.nan)
                out_dict[f"{key}_q25"] = np.full(len(weekly_index), np.nan)
                out_dict[f"{key}_q75"] = np.full(len(weekly_index), np.nan)

        # ---- Save aggregated CSV ----
        df_out = pd.DataFrame(out_dict)
        csv_path = os.path.join(OUTPUT_DIR, f"{d_type}_{year}_aggregated.csv")
        df_out.to_csv(csv_path, index=False)

        # ---- Save individual CSV ----
        indiv_path = os.path.join(OUTPUT_DIR, f"{d_type}_{year}_allseries.csv")
        individual_df.to_csv(indiv_path, index=False)

        # ---- Save Plot ----
        fig, ax = plt.subplots(figsize=(10, 5))
        for key in ["ids", "s1dm"]:
            median = out_dict[f"{key}_median"]
            q25 = out_dict[f"{key}_q25"]
            q75 = out_dict[f"{key}_q75"]

            ax.plot(weekly_index, median, label=f"{key} median")
            ax.fill_between(weekly_index, q25, q75, alpha=0.3)

        ax.set_title(f"{d_type.capitalize()} {year}")
        ax.set_xlabel("Time (weekly)")
        ax.set_ylabel("Value")
        ax.legend()

        fig_path = os.path.join(FIGURES_DIR, f"{d_type}_{year}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# # ------------------ EXPORT & PLOT ------------------ #
# for d_type in data:
#     for year in data[d_type]:
#         # Build a global weekly time axis
#         all_times = []
#         for key in ["ids", "s1dm"]:
#             for entry in data[d_type][year][key]:
#                 all_times.append(pd.to_datetime(entry["time"]))
#         start = min([t.min() for t in all_times])
#         end = max([t.max() for t in all_times])
#         weekly_index = pd.date_range(start=start, end=end, freq="W")

#         # Data for aggregated + individual
#         out_dict = {"time": weekly_index}
#         individual_df = pd.DataFrame({"time": weekly_index})

#         for key in ["ids", "s1dm"]:
#             aligned_series = []
#             for i, entry in enumerate(data[d_type][year][key]):
#                 t = pd.to_datetime(entry["time"])
#                 ts = entry["values"]

#                 df_ts = pd.DataFrame({"time": t, "val": ts}).set_index("time")
#                 df_ts = df_ts.reindex(weekly_index)
#                 aligned_series.append(df_ts["val"].values)

#                 # add as individual column
#                 individual_df[f"{key}_{i}"] = df_ts["val"].values

#             if len(aligned_series) > 0:
#                 arrs = np.vstack(aligned_series)
#                 out_dict[f"{key}_mean"] = np.nanmean(arrs, axis=0)
#                 out_dict[f"{key}_std"] = np.nanstd(arrs, axis=0)
#             else:
#                 out_dict[f"{key}_mean"] = np.full(len(weekly_index), np.nan)
#                 out_dict[f"{key}_std"] = np.full(len(weekly_index), np.nan)

#             # out_dict[f"{key}_mean"] = np.nanmean(arrs, axis=0)
#             # out_dict[f"{key}_std"] = np.nanstd(arrs, axis=0)

#         # ---- Save aggregated CSV ----
#         df_out = pd.DataFrame(out_dict)
#         csv_path = os.path.join(OUTPUT_DIR, f"{d_type}_{year}_aggregated.csv")
#         df_out.to_csv(csv_path, index=False)

#         # ---- Save individual CSV ----
#         indiv_path = os.path.join(OUTPUT_DIR, f"{d_type}_{year}_allseries.csv")
#         individual_df.to_csv(indiv_path, index=False)

#         # ---- Save Plot ----
#         fig, ax = plt.subplots(figsize=(10, 5))
#         for key in ["ids", "s1dm"]:
#             mean = out_dict[f"{key}_mean"]
#             std = out_dict[f"{key}_std"]
#             ax.plot(weekly_index, mean, label=f"{key} mean")
#             ax.fill_between(weekly_index, mean - std, mean + std, alpha=0.3)

#         ax.set_title(f"{d_type.capitalize()} {year}")
#         ax.set_xlabel("Time (weekly)")
#         ax.set_ylabel("Value")
#         ax.legend()

#         fig_path = os.path.join(FIGURES_DIR, f"{d_type}_{year}.png")
#         plt.savefig(fig_path, dpi=150, bbox_inches="tight")
#         plt.close(fig)
