from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from h5py import Dataset
from numpy.typing import NDArray
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client


# convert hdf5 to obspy stream
def make_stream(dataset: Dataset):
    """
    input: hdf5 dataset
    output: obspy stream
    """
    data: NDArray[np.float64] = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.delta = 0.01
    tr_E.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_E.stats.channel = f"{dataset.attrs['receiver_type']}E"
    tr_E.stats.station = dataset.attrs["receiver_code"]
    tr_E.stats.network = dataset.attrs["network_code"]

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.delta = 0.01
    tr_N.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_N.stats.channel = f"{dataset.attrs['receiver_type']}N"
    tr_N.stats.station = dataset.attrs["receiver_code"]
    tr_N.stats.network = dataset.attrs["network_code"]

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs["trace_start_time"])
    tr_Z.stats.channel = f"{dataset.attrs['receiver_type']}Z"
    tr_Z.stats.station = dataset.attrs["receiver_code"]
    tr_Z.stats.network = dataset.attrs["network_code"]

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


# Plot displacement, velocity and acceleration
# This can be used for inspection the wave data
def make_plot(tr: obspy.Trace, title: str = "Raw Data", ylab: str = "counts"):
    """
    input: trace
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()


# get trace names (event names)
def get_trace_names(csv_file: str) -> list[str]:
    # reading the metadata from csv file
    df = pd.read_csv(csv_file)

    # filterering the dataframe: source distance <= 200 km and magnitude > 5
    df = df[
        (df.trace_category == "earthquake_local")
        & (df.source_distance_km <= 200)
        & (df.source_magnitude > 5)
    ]
    print(f"total events selected: {len(df)}")

    # making a list of trace names for the selected data
    ev_list: list[str] = df["trace_name"].to_list()

    return ev_list


def convert_signal_to_acceleration(file_name: str, ev_list: list[str]) -> None:
    dtfl = h5py.File(file_name, "r")
    total = len(ev_list)

    # downloading the instrument response of the station from IRIS
    client = Client("IRIS")

    # Processing each event
    for i, evi in enumerate(ev_list):
        if i < 200:
            continue
        if i >= 210:
            break
        print(f"progress: {i+1}/{total} {evi}")

        # read data
        dataset = cast(Dataset, dtfl.get("data/" + str(evi)))

        # downloading the instrument response of the station from IRIS
        try:
            inventory = client.get_stations(
                network=dataset.attrs["network_code"],
                station=dataset.attrs["receiver_code"],
                # starttime=UTCDateTime(dataset.attrs["trace_start_time"]),
                # endtime=UTCDateTime(dataset.attrs["trace_start_time"]) + 60,
                loc="*",
                channel="*",
                level="response",
            )
        except Exception as e:
            print(f"Error:{e}")
            continue

        # converting into acceleration data
        st = make_stream(dataset)
        try:
            st = st.remove_response(inventory=inventory, output="ACC", plot=False)
        except Exception as e:
            print(f"Error: {e}")
            continue

        # get the acceleration for each component, East, North and Vertical
        e = cast(obspy.Trace, st[0])
        n = cast(obspy.Trace, st[1])
        z = cast(obspy.Trace, st[2])

        # ploting the verical component (for inspection if needed)
        # make_plot(e, title="Acceleration", ylab="m/s^2")
        # make_plot(n, title="Acceleration", ylab="m/s^2")
        # make_plot(z, title="Acceleration", ylab="m/s^2")

        # saving the data
        e.write(f"acc-simulation/{evi}.sac", format="SAC")

    dtfl.close()


# Converting the raw signale into acceleration and saving it in .sac format
if __name__ == "__main__":
    # Get event names of STEAD waves, each event name is unique
    csv_file = "merge.csv"
    ev_list = get_trace_names(csv_file)

    # Converting the raw signale into acceleration
    file_name = "merge.hdf5"
    convert_signal_to_acceleration(file_name, ev_list)
