import csv
import numpy as np
import re
import pandas as pd


def readCSV(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile)
        result = {}
        for row in reader:
            for column, value in row.items():
                result.setdefault(column, []).append(value)
    return(result)

def getIndex (regex, obj, col):
    l = obj[col]
    idx = [i for i, item in enumerate(l) if regex.match(item)]
    return(idx)
def makeDict(regex, obj, col):
    idx = getIndex(regex, obj, col)
    res = {key:[value[i] for i in idx] for key, value in obj.items()}
    return(res)

def square(x):
    return(float(x) ** 2)
def norm(obj, l3):
    x, y, z = (obj[i] for i in l3)
    return([square(x[i]) + square(y[i]) + square(z[i]) for i in range(0, len(x))])

def gensensors(path):
    d = readCSV(path)
    one = re.compile("1")
    two = re.compile("2")
    sens = [one, two]
    sens1, sens2 = (makeDict(f, d, 'SensorId') for f in sens)
    accs = [' AccX (g)', ' AccY (g)', ' AccZ (g)']
    a1norm, a2norm = (norm(s, accs) for s in [sens1, sens2])
    return(a1norm, a2norm)


# for now to prove that we can do anything we are just going to use the norm of
# the acceleration vectors





