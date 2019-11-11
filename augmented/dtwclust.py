import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from dtw import *
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

x = np.load("x.npy")
y = np.load("y.npy")

x_train, x_test, y_train, y_test = train_test_split(x[:250,:], y[:250], test_size = 0.5)





x.shape

dtw(x[1,:], x[0,:], keep_internals = True)

alignment.plot("twoway", offset = -5)

def dtwdist(x,y):
    res = dtw(x, y, keep_internals = True)
    return(res.distance)

dist = pdist(x_train, dtwdist)
dist2 = squareform(pdist)

# https://qiita.com/takeshikondo/items/3aae12df9063c539b0ea
class KMedoids():
  def __init__(self, n_cluster, max_iter=300):
    self.n_cluster = n_cluster
    self.max_iter = max_iter

  def fit_predict(self, D):

    m, n = D.shape

    initial_medoids = np.random.choice(range(m), self.n_cluster, replace=False)
    tmp_D = D[:, initial_medoids]

    labels = np.argmin(tmp_D, axis=1)

    results = pd.DataFrame([range(m), labels]).T
    results.columns = ['id', 'label']

    col_names = ['x_' + str(i + 1) for i in range(m)]
    results = pd.concat([results, pd.DataFrame(D, columns=col_names)], axis=1)

    before_medoids = initial_medoids
    new_medoids = []

    loop = 0
    while ((len(set(before_medoids).intersection(set(new_medoids))) != self.n_cluster)
           and (loop < self.max_iter) ):

      if loop > 0:
        before_medoids = new_medoids.copy()
        new_medoids = []

      for i in range(self.n_cluster):
        tmp = results[results['label'] == i].copy()

        tmp['distance'] = np.sum(tmp.loc[:, ['x_' + str(id + 1)
                                            for id in tmp['id']]].values, axis=1)
        tmp = tmp.reset_index(drop=True)
        new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

      new_medoids = sorted(new_medoids)
      tmp_D = D[:, new_medoids]

      clustaling_labels = np.argmin(tmp_D, axis=1)
      results['label'] = clustaling_labels

      loop += 1

    results = results.loc[:, ['id', 'label']]
    results['flag_medoid'] = 0

    for medoid in new_medoids:
      results.loc[results['id'] == medoid, 'flag_medoid'] = 1

    tmp_D = pd.DataFrame(tmp_D, columns=['medoid_distance'+str(i)
                                         for i in range(self.n_cluster)])
    results = pd.concat([results, tmp_D], axis=1)

    self.results = results
    self.cluster_centers_ = new_medoids

    return results['label'].values

km = KMedoids(2, 300)
pred = km.fit_predict(dist2)

print ('cluster predict: ', pred)
