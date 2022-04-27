import numpy as np
import seaborn as sns
import os
from PIL import Image
import cv2
import matplotlib.pylab as plt

sns.set_theme()

image_np = Image.open("only_tgt_ipi.jpg").convert("L")
image_np = np.array(image_np)

ax = sns.heatmap(
    image_np,
    annot=True,
    cmap="Blues",
    cbar=False,
    fmt="d",
    xticklabels=False,
    yticklabels=False,
    linewidths=0.0,
)

# ax = sns.heatmap(image_np, cmap='CMRmap', cbar=False, xticklabels=False, yticklabels=False, linewidths=0.5)

plt.tight_layout()
plt.savefig("heatmap_tgt.pdf")
plt.show()
