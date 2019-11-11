import n2d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
matplotlib.use('agg')

x1 = np.load('pcsNormalAugmented.npy')
y1 = np.zeros(x1.shape[-1])
x2 = np.load('pcsLimpAugmented.npy')
y2 = np.ones(x2.shape[-1])
x3 = np.load('originalNormalPcs.npy')
y3 = np.zeros(x3.shape[-1])
x4 = np.load('originalLimpPcs.npy')
y4 = np.ones(x4.shape[-1])

x = np.hstack([x1, x2, x3, x4]).T
y = np.hstack([y1, y2, y3, y4]).T

y.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.5)

x_set = np.vstack([x_train, x_test])
y_set = np.hstack([y_train, y_test])

x_set = np.load("x.npy")
y_set = np.load("y.npy")


ndim = 10
model = n2d.n2d(x_set, nclust = ndim)

model.preTrainEncoder(weights= "weights/thisMightWork-1000-ae_weights.h5")

manifoldGMM = n2d.UmapGMM(nclust = 2)

model.predict(manifoldGMM)

model.assess(y_set.T)
# acc .97  nmi .82 ari .89

model.visualize(y_set, None, dataset= "augmented", nclust = 2)

np.save("x.npy", x_set)
np.save("y.npy", y_set)




