"""
Create a radar chart to illustrate the performance of the data-driven methods
"""

from math import pi
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

df = pd.DataFrame({
	'group': 		['DD-v2-03','DD-v1-03'],
	'nIoU': 		[0.772, 0.774],
	r"$F_\beta$": 	[0.900, 0.908],
	'MCC': 			[0.842, 0.817],
	'TPR': 			[0.88, 0.90]
})

categories	= list(df)[1:]
N 			= len(categories)

angles 	= [n / float(N) * 2 * pi for n in range(N)]
angles 	+= angles[:1]

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)

ax.set_rlabel_position(0)
plt.yticks([0.7, 0.8, 0.9], ["0.7","0.8","0.9"], color="grey", size=10)
plt.ylim(0.6,1)

values	= df.loc[0].drop('group').values.flatten().tolist()
values 	+= values[:1]

ax.plot(angles,
		values,
		color		= '#FF0000',
		linewidth	= 2,
		label 	 	= "DD-v2-03")

values 	= df.loc[1].drop('group').values.flatten().tolist()
values 	+= values[:1]

ax.plot(angles,
		values,
		color 		= '#000000',
		linewidth 	= 2,
		label 		= "DD-v1-03")

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('dd_radar_chart.pdf')
plt.show()
