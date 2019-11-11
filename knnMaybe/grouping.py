import csv
import numpy as np
import re
import basic_preprocess as bp



a1norm, a2norm = bp.gensensors("walknormal.csv")
a1limp, a2limp = bp.gensensors("walklimp.csv")

norms = [a1norm, a2norm, a1limp, a2limp]
a1norms, a2norms, a1limps, a2limps = (n[:4500] for n in norms)

#lnorm = [[i, 1] for i in a1norms]
#rnorm = [[i, 1] for i in a2norms]
#llimp = [[i, 1] for i in a1limps]
#rlimp = [[i, 1] for i in a2limps]

lnorm = a1norms
rnorm = a2norms
llimp = a1limps
rlimp = a2limps


def chunks(l, n):
    n = max(1, n)
    return (list(l[i:i+n]) for i in range(0, len(l), n))

shorts = [lnorm, rnorm, llimp, rlimp]

n1, n2, l1, l2 = (list(chunks(s, 1500)) for s in shorts)


groups =  [n1, n2, l1, l2]
x  = np.vstack(list(map(np.vstack, groups)))
x.shape

def genSplitClasses(x):
    length = x.shape[0]
    split = int(length/2)
    res = np.ones(length)
    res[:split] = 0
    return(res)

# 1 is limping 0 is not limping
y = genSplitClasses(x)
y

x = np.asmatrix(x)
x.shape

from dtw import dtw

def normalDTWDistance(x,y):
    res = dtw(x, y, keep_internals = True)
    return(res.normalizedDistance)


# for fun
from scipy.spatial.distance import pdist, squareform

dist = pdist(x, normalDTWDistance)

from scipy.cluster import hierarchy


cl = hierarchy.linkage(dist)
import matplotlib.pyplot as plt
plt.figure()
hierarchy.dendrogram(cl, leaf_font_size = 8.)



# using a u in neighbors to be e l i t e
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

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold

clf = NearestNeighbours(normalDTWDistance)
kf = RepeatedKFold(n_splits = 8, n_repeats = 10)

scores = []
for ids, (train_index, test_index) in enumerate(kf.split(x)):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    scores.append(accuracy_score(y_test, preds))

print(scores)
sum(scores)/len(scores)




clf.fit(x_train, y_train)
preds = nn.predict(x_test)


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, preds)

