import numpy as np
import pandas as pd
import math
import plotly.express as px
import heartpy as hp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random as rd
from data_processing.preprocessing import seperate_data_to_sleep_levels as sl

# Feature extraction function 

def feature_extraction(sleep_levels, features, inc, isFatigued):

    for k in range(0, len(sleep_levels)):

        count = 0
        start = 0
        end = inc
        Y = sleep_levels[k][start:end]

        for i in range(0, int(len(sleep_levels[k]/inc)-1)):
            fs = 100.0
            try:

                # extracting features using heartpy

                working_data, measures = hp.process(Y, fs, calc_freq=True, reject_segmentwise=False)

                # checking NANs/Infinity values in the processing 

                if (math.isnan(measures['bpm']) == False | math.isnan(measures['rmssd']) == False | math.isnan(measures['breathingrate']) == False | math.isnan(measures['ibi']) == False | math.isnan(measures['sdnn']) == False | math.isnan(measures['sdsd']) == False | math.isnan(measures['pnn20']) == False | math.isnan(measures['pnn50']) == False | math.isnan(measures['sd1']) == False | math.isnan(measures['sd2']) == False | math.isnan(measures['s']) == False | math.isnan(measures['sd1/sd2']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['hf']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['lf/hf']) == False |
                    math.isinf(measures['bpm']) == False | math.isinf(measures['rmssd']) == False | math.isinf(measures['breathingrate']) == False | math.isinf(measures['ibi']) == False | math.isinf(measures['sdnn']) == False | math.isinf(measures['sdsd']) == False | math.isinf(measures['pnn20']) == False | math.isinf(measures['pnn50']) == False | math.isinf(measures['sd1']) == False | math.isinf(measures['sd2']) == False | math.isinf(measures['s']) == False | math.isinf(measures['sd1/sd2']) == False | math.isinf(measures['lf']) == False | math.isinf(measures['hf']) == False | math.isinf(measures['lf']) == False | math.isinf(measures['lf/hf']) == False):
                    
                    features.append([measures['bpm'], measures['rmssd'], measures['breathingrate'], measures['ibi'], measures['sdnn'], measures['sdsd'], measures['pnn20'], measures['pnn50'], measures['sd1'], measures['sd2'], measures['s'], measures['sd1/sd2'], measures['lf'], measures['hf'], measures['lf/hf'], k, isFatigued[k]])

                # iterating using sleep_level segments  

                start = start + inc
                end = end + inc
                Y = sleep_levels[k][start:end]
                count = count + 1
            except:
                start = start + inc
                end = end + inc
                Y = sleep_levels[k][start:end]


        
    data = pd.DataFrame(features,columns=['bpm', 'hrv', 'br', 'ibi', 'sdnn', 'sdsd', 'pnn20', 'pnn50', 'sd1', 'sd2', 's', 'sd1/sd2', 'lf', 'hf', 'lf/hf', 'level','fatigued'])
    return data


# feature extraction without blocking and without the addition of fatigue or sleep level labels
# require to extract 'live data'

def feature_extraction_without_blocking_and_labels(data, features, inc):
          
    count = 0
    start = 0
    end = inc
    Y = data[start:end]

    for i in range(0, int(len(data/inc)-1)):
        fs = 100.0
        try:

            # extracting features using heartpy

            working_data, measures = hp.process(Y, fs, calc_freq=True, reject_segmentwise=False)

            # checking NANs/Infinity values in the processing 

            if (math.isnan(measures['bpm']) == False | math.isnan(measures['rmssd']) == False | math.isnan(measures['breathingrate']) == False | math.isnan(measures['ibi']) == False | math.isnan(measures['sdnn']) == False | math.isnan(measures['sdsd']) == False | math.isnan(measures['pnn20']) == False | math.isnan(measures['pnn50']) == False | math.isnan(measures['sd1']) == False | math.isnan(measures['sd2']) == False | math.isnan(measures['s']) == False | math.isnan(measures['sd1/sd2']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['hf']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['lf/hf']) == False):
                features.append([measures['bpm'], measures['rmssd'], measures['breathingrate'], measures['ibi'], measures['sdnn'], measures['sdsd'], measures['pnn20'], measures['pnn50'], measures['sd1'], measures['sd2'], measures['s'], measures['sd1/sd2'], measures['lf'], measures['hf'], measures['lf/hf']])
                
            start = start + inc
            end = end + inc
            Y = data[start:end]
            count = count + 1
        except:
            start = start + inc
            end = end + inc
            Y = data[start:end]

    data = pd.DataFrame(features,columns=['bpm', 'hrv', 'br', 'ibi', 'sdnn', 'sdsd', 'pnn20', 'pnn50', 'sd1', 'sd2', 's', 'sd1/sd2', 'lf', 'hf', 'lf/hf'])
    return data


def feature_extraction_without_blocking(data, features, inc):
          
    count = 0
    start = 0
    end = inc
    Y = data[start:end]

    for i in range(0, int(len(data/inc)-1)):
        fs = 100.0
        try:

            # extracting features using heartpy

            working_data, measures = hp.process(Y, fs, calc_freq=True, reject_segmentwise=False)

            # checking NANs/Infinity values in the processing 

            if (math.isnan(measures['bpm']) == False | math.isnan(measures['rmssd']) == False | math.isnan(measures['breathingrate']) == False | math.isnan(measures['ibi']) == False | math.isnan(measures['sdnn']) == False | math.isnan(measures['sdsd']) == False | math.isnan(measures['pnn20']) == False | math.isnan(measures['pnn50']) == False | math.isnan(measures['sd1']) == False | math.isnan(measures['sd2']) == False | math.isnan(measures['s']) == False | math.isnan(measures['sd1/sd2']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['hf']) == False | math.isnan(measures['lf']) == False | math.isnan(measures['lf/hf']) == False):
                features.append([measures['bpm'], measures['rmssd'], measures['breathingrate'], measures['ibi'], measures['sdnn'], measures['sdsd'], measures['pnn20'], measures['pnn50'], measures['sd1'], measures['sd2'], measures['s'], measures['sd1/sd2'], measures['lf'], measures['hf'], measures['lf/hf'], 0, Y[start][0]])
                print("TIMESTAMP = \n", Y[start][0])
            start = start + inc
            end = end + inc
            Y = data[start:end]
            count = count + 1
        except:
            start = start + inc
            end = end + inc
            Y = data[start:end]

    data = pd.DataFrame(features,columns=['bpm', 'hrv', 'br', 'ibi', 'sdnn', 'sdsd', 'pnn20', 'pnn50', 'sd1', 'sd2', 's', 'sd1/sd2', 'lf', 'hf', 'lf/hf','f/nf', 'timestamp'])
    return data