import numpy as np
import pandas as pd
import math
import plotly.express as px
import heartpy as hp
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random as rd

def load_data(data):
    data = pd.read_csv(data).values

    return data

def seperate_annotations(annotation):
    annotations = annotation.values
    count_sleep = 0
    count_reaction_time = 0
    count_diary_entry = 0
    length = len(annotation)

    for i in range(0, length):
        if annotations[:, 1][i] == 'Stanford Sleepiness Self-Assessment (1-7)':
            count_sleep = count_sleep + 1
        if annotations[:, 1][i] == 'Sleep-2-Peak Reaction Time (ms)':
            count_reaction_time = count_reaction_time + 1
        if annotations[:, 1][i] == 'Diary Entry (text)':
            count_diary_entry = count_diary_entry + 1

    stanford_sleep_levels = annotations[:][0:count_sleep]
    sleep_2_peak_reaction_time = annotations[:][count_sleep: count_sleep + count_reaction_time]
    diary_entry = annotations[:][count_sleep + count_reaction_time: count_diary_entry + count_reaction_time]

    return stanford_sleep_levels, sleep_2_peak_reaction_time, diary_entry

def seperate_data_to_sleep_levels(data1, data2, stanford_sleep_levels):

    sleep_l1 = []
    sleep_l2 = []
    sleep_l3 = []
    sleep_l4 = []
    sleep_l5 = []
    sleep_l6 = []
    sleep_l7 = []
    sleep_levels = [sleep_l1, sleep_l2, sleep_l3, sleep_l4, sleep_l5, sleep_l6, sleep_l7]
    length = len(stanford_sleep_levels)

    for i in range(0, length):

        time = stanford_sleep_levels[i, 0].split(":")[0].split("T")[1]
        level = int(stanford_sleep_levels[i, 2])
        for i in range(0, len(data1[:, 0])):
            if(data1[i, 0].split(":")[0] == time):
                sleep_levels[level - 1].append(data1[i, 1])

    for i in range(0, length):

        time = stanford_sleep_levels[i, 0].split(":")[0].split("T")[1]
        level = int(stanford_sleep_levels[i, 2])
        for i in range(0, len(data2[:, 0])):
            if(data2[i, 0].split(":")[0] == time):
                sleep_levels[level - 1].append(data2[i, 1])

    sleep_level_01 = np.array(sleep_levels[0])
    sleep_level_02 = np.array(sleep_levels[1])
    sleep_level_03 = np.array(sleep_levels[2])
    sleep_level_04 = np.array(sleep_levels[3])
    sleep_level_05 = np.array(sleep_levels[4])
    sleep_level_06 = np.array(sleep_levels[5])
    sleep_level_07 = np.array(sleep_levels[6])
    sleep_levels = [sleep_level_01, sleep_level_02, sleep_level_03, sleep_level_04, sleep_level_05, sleep_level_06, sleep_level_07]

    return sleep_levels

def scale(data):
    data = preprocessing.scale(data)
    return data