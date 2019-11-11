import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)



x1 = np.load("originalNormal.npy")
x2 = np.load("limpNormal.npy")

pcnormal = pca.fit_transform(StandardScaler().fit_transform(x1))
pclimp = pca.fit_transform(StandardScaler().fit_transform(x2))

np.save("originalNormalPcs.npy", pcnormal)
np.save("originalLimpPcs.npy", pclimp)


