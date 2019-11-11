import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)



#nl = np.load("limpAugmented.npy")
#
#x = StandardScaler().fit_transform(nl)
#
#np.save("limpAugmentedScaled.npy", x)

#nA = np.load("normalAugmented.npy")
#nA.shape
#
#x = StandardScaler().fit_transform(nA)
#
#np.save("normalAugmentedScaled.npy", x)

#x = np.load("normalAugmentedScaled.npy")
x = np.load("limpAugmentedScaled.npy")



pcs = []
for i in range(0, int(x.shape[-1]/12)):
    res = pca.fit_transform(x[:,(12*i):(12*(i+1))])
    pcs.append(res)


#np.save("pcsNormalAugmented.npy", np.hstack(pcs))
np.save("pcsLimpAugmented.npy", np.hstack(pcs))

