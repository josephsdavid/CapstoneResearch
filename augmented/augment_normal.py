import re
import dataAugmenters as da
import numpy as np
import csv


def readCSV(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile)
        result = {}
        for row in reader:
            for column, value in row.items():
                result.setdefault(column, []).append(value)
    return(result)


walking = readCSV("../data/walknormal.csv")
walking.keys()



def getIndex (regex, obj, col):
    l = obj[col]
    idx = [i for i, item in enumerate(l) if regex.match(item)]
    return(idx)
def makeDict(regex, obj, col):
    idx = getIndex(regex, obj, col)
    res = {key:[value[i] for i in idx] for key, value in obj.items()}
    return(res)


sensors = [re.compile("1"), re.compile("2")]

sensor1, sensor2 = (makeDict(f,walking, 'SensorId') for f in sensors)

accGyro = [' AccX (g)', ' AccY (g)', ' AccZ (g)',
           ' GyroX (deg/s)', ' GyroY (deg/s)', ' GyroZ (deg/s)']

set1 = []
set2 = []
for ag in accGyro:
    set1.append(np.asarray(sensor1[ag], dtype = float))
    set2.append(np.asarray(sensor2[ag], dtype = float))


leftArray = np.vstack(set1)
rightArray = np.vstack(set2)

leftArray, rightArray = (l[:,:4500] for l in [leftArray, rightArray])

original = np.vstack([leftArray, rightArray]).T
np.save("originalNormal.npy", original)

import matplotlib.pyplot as plt
plt.plot(original)
plt.title("Normal Walking")
plt.show()

fig = plt.figure()
for ii in range(8):
    ax = fig.add_subplot(2,4, ii+1)
    ax.plot(da.Scaling(da.Permute(original), sigma = 0.2))


normalA = []
for i in range(0, 10000):
    normalA.append(
        da.Permute(da.Jitter(da.Scaling(original)))
    )

normalAugmented = np.hstack(normalA)

np.save("normalAugmented.npy", normalAugmented)
