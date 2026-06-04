import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



syn = np.load("./synthetic/stocks/path_ada_irr.npy")

d = syn.shape[2]

fig, ax = plt.subplots(1, d, figsize=(20, 5))

idx = np.random.choice(syn.shape[0],10)


for sample in syn[idx]:
    for i in range(d):
        ax[i].plot(sample[:,i])
plt.show()
