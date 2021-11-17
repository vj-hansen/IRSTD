import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 16})

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
a = np.random.rand(20, 20)

fig, ax = plt.subplots()
plt.imshow(a, cmap='binary')

ax.xaxis.set_label_position('top')

ax.xaxis.set_ticks_position('top')
plt.ylabel(r"$y$", fontsize = 18)
plt.xlabel(r"$x$", fontsize = 18)
plt.xlim(0, 10)
plt.ylim(10, 0)
plt.tight_layout()
plt.savefig('miscDiagram-left_hand.pdf')
plt.show()

