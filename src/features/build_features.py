import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Data_Transformation import LowPassFilter, PrincipalComponentAnalysis
from Temporal_Abstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
predictor_columns = list(df.columns[:6])

for col in predictor_columns:
    df[col] = df[col].interpolate()

# Finding the duration of single set    
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# Finding the duration of all the sets

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0] / 5  # Calculating the average time of per repetitions for heavy sets
duration_df.iloc[1] / 10  # Calculating the average time of per repetitions for medium sets

# Butterworth Low-Pass Filter

df_low_pass_filter = df.copy()
LowPass = LowPassFilter()

# Frequency Sampling (200ms for 1 repetition)
fs = 1000/200
cutoff = 1.3
df_low_pass_filter = LowPass.low_pass_filter(df_low_pass_filter, "acc_y", fs, cutoff, order=5)

# Looping all over the columns

for col in predictor_columns:
    df_low_pass_filter = LowPass.low_pass_filter(df_low_pass_filter, col, fs, cutoff, order=5)
    df_low_pass_filter[col] = df_low_pass_filter[col + "_lowpass"]
    del df_low_pass_filter[col + "_lowpass"]
    

# PCA (Pricipal Component Analysis)

df_PCA = df_low_pass_filter.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_PCA, predictor_columns)
df_PCA = PCA.apply_pca(df_PCA, predictor_columns, 3)

# Sum of squared attributes
df_squared = df_PCA.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyro_r = df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

# Temporal Abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyro_r"]

# Window Size
ws = int(1000/200)

# In this case the Numerical Abstraction will do the calculation of mean and std for window size 5
# By doing this we are allowing it to use the data of the different labels (for ex Bench and Ohp data might get mixed)and the calculating the
# mean and std and this step should not be done.

# for col in predictor_columns:
#     df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
#     df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
    
    
# # The earlier step was putting the data of others labels (for example, ohp into row/bench)
# # This step will create subsets of the set labels for individual sets 
# # (for ex -  a subset will have only ohp data or row data or bench data) and then perform numerical abstraction


df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)
    
    

# Discrete Fourier Transform (DFT)

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000/200) # Sampling frequency
ws = int(2800/200) # Window size (2800 miliseconds - average time for a repetition)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transform to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop = True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# Dealing with Overlapping window

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
kmeans = KMeans(n_clusters = 5, n_init = 20, random_state =0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Exporting the dataset

df_cluster.to_pickle("../../data/interim/data_features.pkl")