import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython.display as display

# Loading the preprocessed data
df = pd.read_pickle("../../data/interim/01_resampled_data.pkl")

# Styling the plot

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# Plotting the data and saving the figures

labels = df["label"].unique()
participants = df["participant"].sort_values().unique()

for label in labels:
    for participant in participants:
        combined_df_plot = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        
        if len(combined_df_plot) > 0:        
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_df_plot[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_df_plot[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

            ax[0].legend(loc = "upper left", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc = "upper left", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.title(f"{label} - {participant}".title(), loc="left")
            plt.savefig(f"../../reports/figures/{label.title()}-{participant}.png")
            plt.show()