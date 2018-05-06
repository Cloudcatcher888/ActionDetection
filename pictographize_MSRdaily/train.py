from sklearn import svm

from getfeature import getArmAngleHist
from preproc import get_trajectory


def trainArmAng():
    X = []
    y = []
    for a in range(1, 21):
        for s in range(1, 11):
             for e in range(1, 3):
                 traj = get_trajectory(a, s, e)
                 if traj is not None:
                    X.append(getArmAngleHist(traj))
                    y.append(a)
    clf = svm.SVC(kernel='poly')
    clf.fit(X, y)
    return clf
