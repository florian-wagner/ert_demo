import numpy as np
import matplotlib.pylab as plt
from ipywidgets import interact
t = np.arange(0.0, 1.0, 0.01)


def plot_pseudosepths(skip, max_sep, spacing=1):
    # number of electrodes
    configs_l = []
    N = 48
    for a in range(1, N - skip - 2 - skip):
        b = a + skip + 1
        #print(a, b)
        for i in range(1, max_sep + 1):
            m = b + i * skip
            n = m + skip + 1
            if n <= N:
                #print(a, b, m, n)
                configs_l.append([a, b, m, n])

    configs = (np.array(configs_l) - 1) * spacing

    x = np.mean(configs, axis=1)
    y = -(np.max(configs, axis=1) - np.min(configs, axis=1)) * 0.195

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(0, 70)
    ax.set_ylim(-20, 0)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
