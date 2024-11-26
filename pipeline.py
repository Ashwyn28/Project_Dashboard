from evaluation import dt_evaluation as evl
from ml_models import dt
from ml_models import adaboost
from ml_models import ann
from ml_models import svm
from data_processing.downsampling import downsample
from data_processing.randomisation import randomise
from data_processing.feature_extraction import feature_extraction
import pandas as pd
import pathlib
from data_processing import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

################################################################################################################

# PROCESSING

################################################################################################################

def process(path, d1, d2, annotations, inc, isFatigued, csv_name, rand, ds):

    # get relative data folder
    PATH = pathlib.Path(__file__).parent
    DATA_PATH = PATH.joinpath(path).resolve()

    # load data

    df_day1 = DATA_PATH.joinpath(d1)
    df_day2 = DATA_PATH.joinpath(d2)

    data_day1 = preprocessing.load_data(df_day1)
    data_day2 = preprocessing.load_data(df_day2)

    # Slice according to sleep level

    df_annotations = pd.read_csv(DATA_PATH.joinpath(annotations))

    [stanford_sleep_levels, sleep_2_peak_reaction_time, diary_entry] = preprocessing.seperate_annotations(df_annotations)
    data = preprocessing.seperate_data_to_sleep_levels(data_day1, data_day2, stanford_sleep_levels)

    # Extract features, removing nans

    sleep_levels = data
    features = []
    inc = inc
    isFatigued = isFatigued

    data = feature_extraction(sleep_levels, features, inc, isFatigued)

    #output : dataframe of processed data

    #Randomise
    if(rand == 1):
        data = randomise(data)

    #Downsample
    if(ds == 1):
        data = downsample(data)

    data.to_csv(csv_name)

path = "../project/data"
d1 = "gamer5-ppg-2000-01-01.csv"
d2 = "gamer5-ppg-2000-01-02.csv"
annotations = "gamer5-annotations.csv"
inc = 6000*30
isFatigued = [0, 0, 1, 1, 1, 1, 1]
csv_name = 'pipeline_data_downsampled_30min.csv'

#process(path, d1, d2, annotations, inc, isFatigued, csv_name, rand=0, ds=1)
################################################################################################################

# MODELLING

################################################################################################################

def model(learning_model, df):

    #Split features X and labels y & Train test split 
    [X_train, X_test, y_train, y_test] = dt.tts(df, 0.4)

    print("X_test = \n", X_test)

    #Normailising/scaling the data
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    if(learning_model == dt):

        #Decision Tree
        tree = dt.decision_tree(X_train_scaled, y_train)

        #Decision Tree Prunning
        [best_alpha, alpha_results] = dt.decision_tree_prunning(tree, X_train_scaled, y_train)
        print("best alpha = \n", best_alpha)
        print("alpha = \n", alpha_results)

        tree = dt.decision_tree_pruned(X_train_scaled, y_train, best_alpha)
        model_name = "Decision Tree"

    elif(learning_model == adaboost):

        #Adaboost
        tree = learning_model.adaboost(df, X_train_scaled, y_train, "SAMME", 40)
        model_name = "Adaboost Decision Tree"

    elif(learning_model == ann):

        tree = ann.neural_net(X_train_scaled, y_train, 1000, 3)

        parameter_space = {
            'hidden_layer_sizes': [(10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.0001, 0.05, 0.001],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'momentum': [0, 0.9]
        }

        tree = ann.gridSearch(X_train_scaled, y_train, tree, parameter_space)
        model_name = "Artificial Neural Network"

    elif(learning_model == svm):

        param_grid = [
        {'C': [0.5, 1, 10, 100],
        'gamma': ['scale', 1, 0, 0.1, 0.01, 0.0001],
        'kernel': ['rbf']
        }]

        tree = svm.svc(X_train_scaled, y_train, param_grid)
        model_name = "Support Vector Machine"

    #output : trained model 

    ####### Evaluation 

    #Get accuracy 
    y_pred = tree.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(tree.predict([X_test_scaled[0]]))

    #output : accuracy 

    #print("pred= \n", one_pred)
  
    #print("accuracy = \n", acc)
    #print("length test set = \n", len(X_test))

    return X_test_scaled, X_train_scaled, tree, y_pred, acc, model_name, y_test, y_train, X_test, X_train
