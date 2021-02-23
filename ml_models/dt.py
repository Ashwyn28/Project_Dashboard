import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def tts(df):

    X = df.drop(['fatigued'], axis=1).copy()
    X.head()
    y = df['fatigued'].copy()
    y.head()

    y.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test

def decision_tree(X, y):
    
    dt = DecisionTreeClassifier(random_state=42)
    dt = dt.fit(X, y)

    return dt 

def decision_tree_prunning(dt, X_train, y_train):
    path = dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    ccp_alphas = ccp_alphas[:-1]

    alpha_loop_values = []

    for ccp_alpha in ccp_alphas:
        dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        scores = cross_val_score(dt, X_train, y_train, cv=5)
        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

    alpha_results = pd.DataFrame(alpha_loop_values, columns=["alpha", "mean_accuracy", "std"])
    best_alpha = max(alpha_results.mean_accuracy)

    return best_alpha, alpha_results

def decision_tree_pruned(X, y, ccp_alpha):
    dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    dt = dt.fit(X, y)

    return dt




        