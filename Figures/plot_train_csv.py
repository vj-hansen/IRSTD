"""
Plot csv data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 14})

FILE_DD_V2 = "../train_04_04/run-train_04_04-tag-Loss_total_loss.csv"
FILE_DD_V1 = "../tensorboard_data/train_21_03/21_03_run--tag-Loss_total_loss.csv"
DF_DD_v2 = pd.read_csv(FILE_DD_V2, delimiter=",")
DF_DD_v1 = pd.read_csv(FILE_DD_V1, delimiter=",")
Y_DD_v2 = DF_DD_v2["Value"]
X_DD_v2 = DF_DD_v2["Step"]
X_DD_v1 = DF_DD_v1["Step"]
Y_DD_v1 = DF_DD_v1["Value"]

plt.plot(X_DD_v1, Y_DD_v1, color=("#000000"), linewidth=1)
plt.plot(X_DD_v2, Y_DD_v2, color=("#FF0000"), linewidth=1)
# plt.axhline(y=0.3, color='#FF0000', linestyle='--',linewidth=0.5)
# plt.axhline(y=0.15, color='#000000', linestyle='--',linewidth=0.5)


plt.xlabel("Step")
plt.ylabel(r"Loss ($L$)")
plt.ylim(0, 3)
plt.xlim(0, 4500)
plt.yticks([0, 0.15, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
plt.xticks(np.arange(0, 4500, step=500))
plt.grid("on", alpha=0.5)
plt.legend(["Original ResNet50", "Modified ResNet50"])
plt.tight_layout()


plt.savefig("train_loss_CN_ResNet50.pdf")
plt.show()
