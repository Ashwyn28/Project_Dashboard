import pandas as pd
import pipeline
from data_processing.feature_extraction import feature_extraction_without_blocking_and_labels
from data_processing.feature_extraction import feature_extraction_without_blocking

# Static data to act as test live data

gamer_data_01_day1 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer5-ppg-2000-01-01.csv")
gamer_data_01_day2 = pd.read_csv("file:///Users/ashwyn/Desktop/PPG/gamer5-ppg-2000-01-02.csv")

# parameters for feature extraction

data = pd.concat([gamer_data_01_day1, gamer_data_01_day2])
data = data.values
data = data[:, 1]
inc = 6000
isFatigued = [0, 0, 1, 1, 1, 1, 1]
features = []
csv_name = 'data_g5_live_labels.csv'

# feature extraction

data = feature_extraction_without_blocking(data, features, inc)
data.to_csv(csv_name)

