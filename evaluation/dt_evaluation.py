from sklearn.metrics import accuracy_score


def get_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc

def confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return tn, fp, fn, tp

def ac_fatigued(tp, fn):
    return tp/(tp+fn)

def ac_nfatigued(tn, fp):
    return tn/(tn+fp)

