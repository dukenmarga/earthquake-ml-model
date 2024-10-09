from typing import cast

import h5py
import numpy as np
import pandas as pd
from h5py import Dataset

# This script computes the percentile of the selected waveforms

file_name = "merge.hdf5"
csv_file = "merge.csv"

# reading the csv file into a dataframe:
df = pd.read_csv(csv_file)

# filterering the dataframe
df = df[
    (df.trace_category == "earthquake_local")
    & (df.source_distance_km <= 200)
    & (df.source_magnitude > 7)
]

print(f"total events selected: {len(df)}")

# making a list of trace names for the selected data
ev_list: list[str] = df["trace_name"].to_list()

# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(file_name, "r")
delta = np.array([])
for c, evi in enumerate(ev_list):
    dataset = cast(Dataset, dtfl.get("data/" + str(evi)))

    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)
    p_arrival = cast(float, dataset.attrs["p_arrival_sample"])
    s_arrival = cast(float, dataset.attrs["s_arrival_sample"])
    delta = np.append(delta, s_arrival - p_arrival)

print(f"5 percentile: {np.percentile(delta, 5)/100:.2f} sec")
print(f"10 percentile: {np.percentile(delta, 10)/100:.2f} sec")
print(f"15 percentile: {np.percentile(delta, 15)/100:.2f} sec")
print(f"85 percentile: {np.percentile(delta, 85)/100:.2f} sec")
print(f"90 percentile: {np.percentile(delta, 90)/100:.2f} sec")
print(f"95 percentile: {np.percentile(delta, 95)/100:.2f} sec")
print(f"99 percentile: {np.percentile(delta, 99)/100:.2f} sec")
