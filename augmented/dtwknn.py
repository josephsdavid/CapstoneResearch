import numpy as np
from dtw import dtw
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


x = np.load("x.npy")
x.shape
# 20002, 4500, timestamps are columns

y = np.load("y.npy")
y.shape

# 20002,


def normalDTWDistance(x,y):
    res = dtw(x, y, keep_internals = True)
    return(res.normalizedDistance)


class NearestNeighbours:
    def __init__(self, distance):
        self.distance = distance

    def distances(self, x, y):
        res = []
        for i in range(0,x.shape[0]):
            res.append(self.distance(y, x[:,i]))
        return np.asarray(res)

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        preds = []
        for i in range(0, x.shape[0]):
            distances = self.distances(self.x_train, i)
            min_index = np.argmin(distances)
            preds.append(self.y_train[min_index])
        return preds


clf = NearestNeighbours(normalDTWDistance)


x_train, x_test, y_train, y_test = train_test_split(x[:250,:],y[:250], test_size =0.5)
x_train.shape

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

accuracy_score(y_test, preds)
