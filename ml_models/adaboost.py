from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Adaboost classifier using scikit-learn

def adaboost(df, X, y, algorithm, n_estimators):
    adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm=algorithm, n_estimators=n_estimators)
    adb = adb.fit(X, y)

    return adb

