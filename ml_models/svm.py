from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# SVM using scikit-learn

def svc(X, y, param_grid):

    svm = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose = 0 )
    svm = svm.fit(X, y)

    return svm
