import glob
import random
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5py import Dataset
from numpy.typing import NDArray
from obspy import read
from scipy.fft import fft, fftfreq  # type: ignore
from scipy.signal import butter, filtfilt  # type: ignore


# Newmark β method for solving differential equations
def newmark_beta(
    ground_accel: NDArray[np.float64],
    dt: float,
    m: float,
    k: float,
    zeta: float,  # damping ratio
    beta: float = 1 / 4,
    gamma: float = 1 / 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Calculate derived parameters
    c = 2 * zeta * np.sqrt(k * m)  # Damping coefficient
    num_steps = len(ground_accel)  # Number of time steps

    # Initialize response arrays
    displacement = np.zeros(num_steps)
    velocity = np.zeros(num_steps)
    acceleration = np.zeros(num_steps)

    # Initial conditions
    acceleration[0] = -ground_accel[0]

    # Precompute constants for Newmark-β
    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = (1 / (2 * beta)) - 1
    a4 = (gamma / beta) - 1
    a5 = dt * ((gamma / (2 * beta)) - 1)

    # Effective stiffness
    k_eff = k + a0 * m + a1 * c

    # Time-stepping loop
    for i in range(1, num_steps):
        # Effective force
        f_eff = (
            -m * ground_accel[i]
            + m
            * (
                a0 * displacement[i - 1]
                + a2 * velocity[i - 1]
                + a3 * acceleration[i - 1]
            )
            + c
            * (
                a1 * displacement[i - 1]
                + a4 * velocity[i - 1]
                + a5 * acceleration[i - 1]
            )
        )

        # Solve for new displacement
        displacement[i] = f_eff / k_eff

        # New acceleration
        acceleration[i] = (
            a0 * (displacement[i] - displacement[i - 1])
            - a2 * velocity[i - 1]
            - a3 * acceleration[i - 1]
        )

        # New velocity
        velocity[i] = velocity[i - 1] + dt * (
            (1 - gamma) * acceleration[i - 1] + gamma * acceleration[i]
        )

    return displacement, velocity, acceleration


def sdof_analysis(
    ground_accel: NDArray[np.float64], dt: float, m: float, k: float, damping: float
) -> tuple[NDArray[np.float64], ...]:
    displacement, velocity, acceleration = newmark_beta(
        ground_accel,
        dt,
        m,
        k,
        damping,
    )

    return displacement, velocity, acceleration


def plot_response(
    ground_motion: NDArray[np.float64],
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    acceleration: NDArray[np.float64],
) -> None:
    t = np.arange(0, 60, dt)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot ground motion acceleration
    plt.subplot(4, 1, 1)
    plt.plot(t, ground_motion, label="Ground Motion")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Ground motion")
    plt.xlim(0, 60)
    plt.grid(True)

    # Plot structure displacement (roof level)
    plt.subplot(4, 1, 2)
    plt.plot(t, displacement, label="Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("SDOF Response: Displacement")
    plt.xlim(0, 60)
    plt.grid(True)

    # Plot structure velocity (roof level)
    plt.subplot(4, 1, 3)
    plt.plot(t, velocity, label="Velocity", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("SDOF Response: Velocity")
    plt.xlim(0, 60)
    plt.grid(True)

    # Plot structure acceleration (roof level)
    plt.subplot(4, 1, 4)
    plt.plot(t, acceleration, label="Acceleration", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("SDOF Response: Acceleration")
    plt.xlim(0, 60)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def p_wave_start_time(df: pd.DataFrame, event: str) -> int:
    # get row where event name matches
    df = df[(df.trace_name == event)]
    return int(df["p_arrival_sample"].item())


# Apply a band-pass filter to focus on earthquake frequency range
def bandpass_filter(
    data: NDArray[np.float64], lowcut: float, highcut: float, fs: float, order: int = 4
) -> NDArray[np.float64]:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cast(NDArray[np.float64], butter(order, [low, high], btype="band"))
    filtered_signal = cast(NDArray[np.float64], filtfilt(b, a, data))
    return filtered_signal


# Frequency analysis (FFT)
def frequency_analysis(
    signal: NDArray[np.float64], sampling_rate: float
) -> tuple[NDArray[np.float64], float]:
    N = len(signal)
    T = 1.0 / sampling_rate
    fft_values = np.abs(fft(signal))  # type: ignore
    frequencies = fftfreq(N, T)
    return frequencies, fft_values  # type: ignore


def p_wave_freq(df: NDArray[np.float64]) -> float:
    # Apply band-pass filter to isolate earthquake frequencies (0.5 Hz to 20 Hz)
    rate = 100  # Hz -> use 0.01 second as wave step
    filtered_signal = bandpass_filter(df, lowcut=0.5, highcut=20, fs=rate)

    # Perform frequency analysis (FFT)
    frequencies, fft_values = frequency_analysis(filtered_signal, rate)

    # Find dominant frequency
    dominan_frequency = frequencies[np.argmax(fft_values)]
    return dominan_frequency


def peak_ground_acceleration(p_wave_acc: NDArray[np.float64]) -> float:
    pga = np.max(np.abs(p_wave_acc))  # Calculate Peak Ground Acceleration
    return pga


# Converting the raw signale into acceleration and saving it in .sac format
if __name__ == "__main__":
    # Structural properties
    dt = 0.01  # delta time in seconds
    # mass is based on:
    area = [36, 45, 72, 100]  # [m2]
    mass_per_area = [25, 30, 45]  # [kg/m2]
    masses: list[float] = []
    masses = [a * w for a in area for w in mass_per_area]

    # stiffness is based on the structural properties: n * 12EI/l^3, I = bh^3/12, L=4m
    ks = [
        4 * 12 * 20000 * 0.3 * 0.3**3 / 12 / (4**3) * 1000,
        6 * 12 * 20000 * 0.3 * 0.3**3 / 12 / (4**3) * 1000,
        8 * 12 * 20000 * 0.3 * 0.3**3 / 12 / (4**3) * 1000,
        4 * 12 * 20000 * 0.2 * 0.2**3 / 12 / (4**3) * 1000,
        6 * 12 * 20000 * 0.2 * 0.2**3 / 12 / (4**3) * 1000,
        8 * 12 * 20000 * 0.2 * 0.2**3 / 12 / (4**3) * 1000,
    ]  # stiffness in KN/m
    dampings = [
        0.02,
        0.03,
        0.05,
    ]  # damping ratio: 2% (steel structure), 5% (concrete structure)

    sac_files = glob.glob("acc/*.sac")

    # Read csv metadata as we need to get p-wave start time
    csv_file = "merge.csv"
    df = pd.read_csv(csv_file)

    # Read signals as we need to get segmented p-wave
    file_name = "merge.hdf5"
    dtfl = h5py.File(file_name, "r")

    # Initialize empty dataset: 6 features + 1 target
    dataset = np.empty((0, 7))

    # Loop through all sac files (waveforms)
    for i, sac_file in enumerate(sac_files):
        print(f"Processing: {sac_file} ({i+1}/{len(sac_files)})")

        # Read ground motion (acceleration in m/s^2)
        st = read(sac_file, debug_headers=True)
        ground_acc = st[0].data
        # since we have small acceleration, we want to take the effect of larger earthquake.
        # usually acceleration of 1g is pretty common in Indonesia, so we will normalize
        # waveforms by 9.8 m/s2.
        # we will have 4 types of data: 1 original and 3 amplified
        max_ground_acc = np.max(ground_acc)
        number_of_amplification = 12

        # Get p-wave start time
        trace_name = ".".join(sac_file.split("/")[-1].split(".")[:-1])
        start_time = p_wave_start_time(df, trace_name)

        # Get segmented p-wave signal from start time to 3 seconds after the start time
        time_window = 3  # seconds
        raw_signal = cast(Dataset, dtfl.get(f"data/{trace_name}"))

        for i in range(number_of_amplification):
            amplification = random.uniform(1, 12) / max_ground_acc
            data = np.array(raw_signal) * amplification
            p_wave_signal = data[
                start_time : start_time + time_window * 100, 0
            ]  # use 0th column (E component), since we analyzed only E component

            # Get dominant p-wave frequency
            dominant_p_wave_freq = p_wave_freq(p_wave_signal)

            # Get PGA values from p-wave acceleration
            p_wave_acc = (
                ground_acc[start_time : start_time + time_window * 100] * amplification
            )
            pga = peak_ground_acceleration(p_wave_acc)

            # Perform SDOF analysis for each combination of mass, stiffness, and damping
            for mass in masses:
                for k in ks:
                    for damping in dampings:
                        displacement, velocity, acceleration = sdof_analysis(
                            ground_acc * amplification, dt, mass, k, damping
                        )

                        # Get natural frequency of the structure
                        natural_freq = np.sqrt(k / mass)

                        # Plot response (for inspection)
                        # plot_response(ground_acc, displacement, velocity, acceleration)

                        # Get maximum values
                        max_displacement = max(displacement)
                        max_velocity = max(velocity)
                        max_acceleration = max(acceleration)

                        # Print maximum values
                        # print(f"Maximum displacement: {max_displacement:.6f} m")
                        # print(f"Maximum velocity: {max_velocity:.6f} m/s")
                        # print(f"Maximum acceleration: {max_acceleration:.6f} m/s²")

                        # Arrange the feature and target response
                        new_data = np.array(
                            [
                                pga,
                                dominant_p_wave_freq,
                                mass,
                                k,
                                damping,
                                natural_freq,
                                max_displacement,
                            ]
                        )
                        dataset = np.vstack([dataset, new_data])

    # Save to csv
    df = pd.DataFrame(dataset)
    df.to_csv(
        "dataset.csv",
        header=[
            "PGA",
            "PWaveFreq",
            "Mass",
            "Stiffness",
            "Damping",
            "NaturalFreq",
            "MaxDisplacement",
        ],
        index=False,
    )
