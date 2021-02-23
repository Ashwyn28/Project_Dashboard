from evaluation.dt_evaluation import get_accuracy
from ml_models import dt
from data_processing.downsampling import downsample
from data_processing.randomisation import randomise
from data_processing.feature_extraction import feature_extraction
import pandas as pd
import pathlib
from data_processing import preprocessing

###### Preprocessing 

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../project-template/data").resolve()

# load data

df_day1 = DATA_PATH.joinpath("gamer5-ppg-2000-01-01.csv")
df_day2 = DATA_PATH.joinpath("gamer5-ppg-2000-01-02.csv")

data_day1 = preprocessing.load_data(df_day1)
data_day2 = preprocessing.load_data(df_day2)

# Slice according to sleep level

df_annotations = pd.read_csv(DATA_PATH.joinpath("gamer5-annotations.csv"))

[stanford_sleep_levels, sleep_2_peak_reaction_time, diary_entry] = preprocessing.seperate_annotations(df_annotations)
data = preprocessing.seperate_data_to_sleep_levels(data_day1, data_day2, stanford_sleep_levels)
print(data)
print("seperated according to sleep levels\n")
# Extract features, removing nans

sleep_levels = data
features = []
inc = 6000
isFatigued = [0, 0, 1, 1, 1, 1, 1]

data = feature_extraction(sleep_levels, features, inc, isFatigued)

#output : dataframe of processed data

######## Modelling

#Randomise
data = randomise(data)

#Downsample
data = downsample(data)
print(data)

#Split features X and labels y & Train test split 
[X_train, X_test, y_train, y_test] = dt.tts(data)

#Normailising/scaling the data
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

#Decision Tree
tree = dt.decision_tree(X_train_scaled, y_train)

#Decision Tree Prunning
[best_alpha, alpha_results] = dt.decision_tree_prunning(tree, X_train_scaled, y_train)

tree = dt.decision_tree_pruned(data, X_train_scaled, y_train, 0.014)

#output : trained model 

####### Evaluation 

#Get accuracy 
acc = get_accuracy(tree, X_test_scaled, y_test)

#output : accuracy 

print(acc)

