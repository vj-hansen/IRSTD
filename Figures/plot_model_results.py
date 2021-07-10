"""
Plot csv data.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

df = pd.read_csv("../testing-models.csv", delimiter=';')
df = df.sort_values(by=['speed'])
arr = df.to_numpy()

models = arr[:6, 0]
niou = arr[:6, 1]
fb = arr[:6, 2]
speed = arr[:6, 3]
mcc = arr[:6, 4]

labels = [models[0], models[1], models[2], models[3], models[4], models[5]]
fb_vals = [fb[0], fb[1], fb[2], fb[3], fb[4], fb[5]]
niou__vals = [niou[0], niou[1], niou[2], niou[3], niou[4], niou[5]]
mcc_vals = [mcc[0], mcc[1], mcc[2], mcc[3], mcc[4], mcc[5]]
speed_vals = [speed[0], speed[1], speed[2], speed[3], speed[4], speed[5]]

x = np.arange(len(labels))
WIDTH = 0.35

fig, ax = plt.subplots()

# plt.scatter(fb_vals, mcc_vals, c=mcc, s=100, cmap='Greys', edgecolor='black')
# for i in range(6):
# 	if models[i] == 'DD-v2-05' or models[i] == 'DD-v1-03':
# 		fl = -0.02
# 		gn = -0.06
# 	else:
# 		fl = -0.04
# 		gn = 0.02
# 	plt.annotate(models[i], (fb_vals[i], mcc_vals[i]),
# 				xytext = (fb_vals[i]+fl, mcc_vals[i]+gn), size=12,
# 				horizontalalignment='center', verticalalignment='bottom')

# rects1 = ax.barh(x - WIDTH/2, niou__vals, WIDTH, label='nIoU', color=('#0F76AF'))
rects1 = ax.barh(x, speed_vals, WIDTH, label='FPS', color=('#0000FF'), alpha=0.6, edgecolor='black')
# rects2 = ax.barh(x + WIDTH/2, fb_vals, WIDTH, label='fb', color=('#D64349'))

ax.invert_yaxis()
# plt.xlabel(r"$F_\beta$")
plt.xlabel('FPS')
# plt.ylabel('MCC')

ax.set_yticks(x)
ax.set_yticklabels(labels)
# plt.legend()
# plt.xticks(np.arange(0.4, 1, step=0.1))
# plt.yticks(np.arange(-0.1, 1, step=0.1))
# plt.grid(alpha=0.3)
ax.bar_label(rects1, label_type='edge', padding=6)
# ax.bar_label(rects2)
plt.xlim(0, 10)
# plt.ylim(-0.1, 1)

plt.tight_layout()
plt.savefig('test_res_fps.pdf')
plt.show()
