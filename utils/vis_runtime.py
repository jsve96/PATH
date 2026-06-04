import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


runtimes = pd.read_csv("./results/runtimes.csv")
methods = runtimes.iloc[:,0]
runtimes = runtimes.iloc[:,1:].mean(axis=1)

ds_acc = pd.read_csv("./results/main_table_mean.csv").iloc[:,1:].mean(axis=1)

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

for i, method in enumerate(methods):
    if method == "PATH (Ours)":
        ax.scatter(runtimes[i], ds_acc[i], label=method, marker="*", color="blue", s=120)
        ax.annotate(
        method,
        xy=(runtimes[i], ds_acc[i]),
        xytext=(70, -0.5),          # move label left
        textcoords="offset points",
        ha="right",              # align text to the right of label box
        va="center",
        fontsize=10,
    )
    else:
        ax.scatter(runtimes[i], ds_acc[i], label=method, marker="x", color="black",linewidth=2.5)
        if method == 'CSPD-OU':
            ax.annotate(
                method,
                xy=(runtimes[i], ds_acc[i]),
                xytext=(50, -0.5),          # move label left
                textcoords="offset points",
                ha="right",              # align text to the right of label box
                va="center",
                fontsize=10,
            )
        else:
            ax.annotate(
                method,
                xy=(runtimes[i], ds_acc[i]),
                xytext=(-5, -0.5),          # move label left
                textcoords="offset points",
                ha="right",              # align text to the right of label box
                va="center",
                fontsize=10,
            )

ax.set_xscale("log")
ax.set_ylabel("Discriminative Score")
ax.set_xlabel("Training Time (s)")
ax.grid(linewidth=0.5)
ax.set_ylim(0,0.52)
ax.hlines(0.5,30,ax.get_xlim()[1]+4000, ls='--',color='black')
ax.set_xlim(30,ax.get_xlim()[1])
fig.savefig('./results/img/runtime.pdf',bbox_inches='tight')

plt.show()
