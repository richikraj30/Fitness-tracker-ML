import pandas as pd
from glob import glob


data_path = "../../data/raw/MetaMotion\\"

# Creating a function for all the preprocessing steps

files = glob("../../data/raw/MetaMotion/*.csv")

def read_and_preprocess_data(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1
    
    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")    
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set   
            acc_set += 1     # Creating a new set for each participant (It's nothing but an identifier for each participant)
            acc_df = pd.concat([acc_df, df])
            
        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
    
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]
    
    return acc_df, gyro_df

acc_df, gyro_df = read_and_preprocess_data(files)

merged_data = pd.concat([acc_df.iloc[:,:3], gyro_df], axis=1)
# Renaming columns for better readability
merged_data.columns = [
    
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "participant",
    "label",
    "category",
    "set"
]

# Accelerometer frequency : 12.500 Hz
# Gyroscope frequency : 25.000 Hz
# The NaN values are due to the different sampling frequencies of accelerometer and gyroscope. 

sampling = {
    
    "acc_x" : "mean",
    "acc_y" : "mean",
    "acc_z" : "mean",
    "gyro_x" : "mean",
    "gyro_y" : "mean",
    "gyro_z" : "mean",
    "participant" : "last", 
    "label" : "last",
    "category" : "last",
    "set" : "last"
}

# Splitting the data by days
days = [g for n, g in merged_data.groupby(pd.Grouper(freq="D"))]

#  Using resampling method
#  Resample rule - '200ms' means that we will sample the data at 200 milliseconds intervals
#  Applying the resampling operation to each day
resampled_data = pd.concat([df.resample(rule = "200ms").apply(sampling).dropna() for df in days])
resampled_data["set"] = resampled_data["set"].astype(int)

# Exporting the preprocessed data to a pickle file for further analysis
resampled_data.to_pickle("../../data/interim/01_resampled_data.pkl")

