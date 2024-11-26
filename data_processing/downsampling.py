import pandas as pd
from sklearn.utils import resample

# Downsampling function 

def downsample(data):

    # data with labels == fatigued set to 0

    df_not_fatigued = data[data['fatigued'] == 0]

    # data with labels == notfatigues set to 1

    df_fatigued = data[data['fatigued'] == 1]

    # downsampling and concating data

    df_nfatigued_downsample = resample(df_not_fatigued, replace=False, n_samples=len(df_fatigued), random_state=42)
    df_downsampled = pd.concat([df_nfatigued_downsample, df_fatigued])
    data = df_downsampled
    
    return data

