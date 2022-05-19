import numpy as np
import matplotlib.pyplot as plt

dt = 0.0005
end_of_time = 0.1
time = np.arange(0,end_of_time,dt)

reg1e4 = np.loadtxt('experiments/TD_results/reg1e4.txt', dtype = complex)
reg1e8 = np.loadtxt('experiments/TD_results/reg1e8.txt', dtype = complex)
reg1e7 = np.loadtxt('experiments/TD_results/reg1e7.txt', dtype = complex)
reg1e6 = np.loadtxt('experiments/TD_results/reg1e6.txt', dtype = complex)
netket = np.load('experiments/TD_results/netket.npy')
qutip = np.load('experiments/TD_results/qutip.npy')


plt.plot(time, reg1e4, label = '1e-4')
plt.plot(time, reg1e8, label = '1e-8')
plt.plot(time, reg1e7, label = '1e-7')
plt.plot(time, reg1e6, label = '1e-6')
plt.plot(time, netket/10, label = 'netket')
plt.plot(time, qutip/10, label = 'qutip')

plt.legend()
plt.grid()

plt.xlabel('time in (a.u.))')
plt.ylabel('per spin expectation value of $S_x$')

plt.show()