from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# ANN classifier using scikit-learn

def neural_net(X, y, max_iter, random_state):
    
    clf = MLPClassifier(random_state=random_state, max_iter=max_iter)
    clf = clf.fit(X, y)

    return clf

# Gridsearch to perform cross validation

def gridSearch(X, y, clf, parameter_space):

    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X, y)

    return clf

