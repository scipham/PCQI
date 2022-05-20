import numpy as np
import matplotlib.pyplot as plt

dt = 0.0005
end_of_time = 0.35
time = np.arange(0,end_of_time,dt)

reg1e4 = np.loadtxt('experiments/TD_results_eot0.35c-1/reg7001e4c-1.txt', dtype = complex)
#reg1e8 = np.loadtxt('experiments/TD_results_eot0.35c-1/reg7001e8.txt', dtype = complex)
#reg1e7 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e7.txt', dtype = complex)
#reg1e6 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e6.txt', dtype = complex)
reg1e5 = np.loadtxt('experiments/TD_results_eot0.35c-1/reg7005e4c-1.txt', dtype = complex)
reg1e45 = np.loadtxt('experiments/TD_results_eot0.35c-1/reg7005e5c-1.txt', dtype = complex)
reg1e55 = np.loadtxt('experiments/TD_results_eot0.35c-1/reg7001e5c-1.txt', dtype = complex)
netket = np.load('experiments/TD_results_eot0.35c/netket.npy')
qutip = np.load('experiments/TD_results_eot0.35c/qutip.npy')


plt.plot(time[:-1], reg1e4, label = '1e-4')
plt.plot(time[:-1], reg1e45, label = '5e-4')
plt.plot(time[:-1], reg1e55, label = '5e-5')
plt.plot(time[:-1], reg1e5, label = '1e-5')
#plt.plot(time, netket/10, label = 'netket')
plt.plot(time, qutip/10, label = 'qutip')

plt.ylim(0.15,0.35)

plt.legend()
plt.grid()

plt.xlabel('time in (a.u.))')
plt.ylabel('per spin expectation value of $S_x$')

plt.show()