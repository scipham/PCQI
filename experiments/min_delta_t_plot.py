import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Manually collected data:
quench_strength = np.array([0.05,0.1, 0.14, 0.17])
plot_quench_strength = np.array([0.05,0.1, 0.14, 0.17, 0.2, 0.3, 0.4])
min_delta_t = np.array([0.001, 2e-4, 8e-5, 2e-5])

fit_func = lambda x, a, b: a*x + b

popt, pcov = curve_fit(fit_func, quench_strength, np.log(min_delta_t))
fit_delta_t = np.exp(fit_func(plot_quench_strength, popt[0], popt[1]))

fig, ax = plt.subplots()

ax.plot(plot_quench_strength, fit_delta_t,label="fit",color="blue")
ax.scatter(quench_strength, min_delta_t, marker="x", label="data", color="red", s=56)

ax.set_yscale("log")
ax.legend()
ax.set_xlim((0.0, 0.42))
ax.set_ylim((6e-7, 1e-2))
ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
ax.set_xlabel(r'Quench strength in transverse field ($h$)')
ax.set_ylabel(r'Minimum Timestep $\Delta t$ required')

plt.show()