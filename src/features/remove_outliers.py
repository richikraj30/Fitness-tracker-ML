import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

df = pd.read_pickle('../../data/interim/01_resampled_data.pkl')

outlier_columns = list(df.columns[:6])

# Function to mark values as outliers using the IQR method.
def mark_outliers_iqr(dataset, col):
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset
# Plot outliers in case of a binary outlier score. Here, the col specifies the real data
# column and outlier_col the columns with a binary value (outlier or not).
# def plot_binary_outliers(dataset, col, outlier_col, reset_index):
#     dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
#     dataset[outlier_col] = dataset[outlier_col].astype("bool")

#     if reset_index:
#         dataset = dataset.reset_index()

#     fig, ax = plt.subplots()

#     plt.xlabel("samples")
#     plt.ylabel("value")

#     # Plot non outliers in default color
#     ax.plot(
#         dataset.index[~dataset[outlier_col]],
#         dataset[col][~dataset[outlier_col]],
#         "+",
#     )
#     # Plot data points that are outliers in red
#     ax.plot(
#         dataset.index[dataset[outlier_col]],
#         dataset[col][dataset[outlier_col]],
#         "r+",
#     )

#     plt.legend(
#         ["no outlier " + col, "outlier " + col],
#         loc="upper center",
#         ncol=2,
#         fancybox=True,
#         shadow=True,
#     )
#     plt.show()

    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    """    
def mark_outliers_chauvenet(dataset, col, C=2):

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Mark values as outliers using LOF
def mark_outliers_lof(dataset, columns, n=20):
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores 

outliers_removed_df = df.copy()


for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
        
        # Replacing the values marked as outlier with NAN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
        
        # Update the column in the original dataset
        outliers_removed_df.loc[outliers_removed_df["label"] == label, col] = dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        
        print(f"Label: {label}, Column: {col}, Outliers removed: {n_outliers}")
        

outliers_removed_df.to_pickle("../../data/interim/outliers_removed_chauvenets.pkl")