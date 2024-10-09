import glob

import numpy as np
from obspy import read

# Converting the raw signale into acceleration and saving it in .sac format
if __name__ == "__main__":

    sac_files = glob.glob("acc-simulation/*.sac")

    # Loop through all sac files (waveforms)
    amplification: list[float] = [3, 6, 9.8]
    for i, sac_file in enumerate(sac_files):
        # if i == 0:
        #     continue
        print(f"Processing: {sac_file} ({i+1}/{len(sac_files)})")

        # Read ground motion (acceleration in m/s^2)
        st = read(sac_file, debug_headers=True)
        ground_acc = st[0].data

        max_acc = np.max(ground_acc)
        a = np.array(ground_acc * amplification[i] / max_acc)
        a.tofile(f"{sac_file}.csv", sep=",")
        if i == 2:
            break

    # # Save to csv
    # df = pd.DataFrame(dataset)
    # df.to_csv(
    #     "dataset.csv",
    #     header=[
    #         "PGA",
    #         "PWaveFreq",
    #         "Mass",
    #         "Stiffness",
    #         "Damping",
    #         "NaturalFreq",
    #         "MaxDisplacement",
    #     ],
    #     index=False,
    # )
