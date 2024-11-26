import pipeline
import pandas as pd
from data_processing.preprocessing import scale

# live data (static data used for now)

df_live = pd.read_csv("data_g5_live_30min.csv")
df_live = df_live.drop(columns=['Unnamed: 0'])
df = scale(df_live)

# prediction of live data

def make_prediction(i, model):
    sample = [df[i]]
    pred = model.predict(sample)[0]

    return pred

 