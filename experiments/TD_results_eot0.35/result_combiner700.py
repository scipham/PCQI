import numpy as np
import matplotlib.pyplot as plt

dt = 0.0005
end_of_time = 0.35
time = np.arange(0,end_of_time,dt)

reg1e4 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e4.txt', dtype = complex)
reg1e8 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e8.txt', dtype = complex)
reg1e7 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e7.txt', dtype = complex)
reg1e6 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e6.txt', dtype = complex)
reg1e525 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e525.txt', dtype = complex)
reg1e55 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e55.txt', dtype = complex)
reg1e575 = np.loadtxt('experiments/TD_results_eot0.35/reg7001e575.txt', dtype = complex)
netket = np.load('experiments/TD_results_eot0.35/netket.npy')
qutip = np.load('experiments/TD_results_eot0.35/qutip.npy')


plt.plot(time[:-1], reg1e4, label = '1e-4')
plt.plot(time[:-1], reg1e8, label = '1e-8')
plt.plot(time[:-1], reg1e7, label = '1e-7')
plt.plot(time[:-1], reg1e6, label = '1e-6')
plt.plot(time[:-1], reg1e525, label = '1e-525')
plt.plot(time[:-1], reg1e55, label = '1e-55')
plt.plot(time[:-1], reg1e575, label = '1e-575')
#plt.plot(time, netket/10, label = 'netket')
plt.plot(time, qutip/10, label = 'qutip')

plt.legend()
plt.grid()

plt.ylim(0.15,0.35)

plt.xlabel('time in (a.u.))')
plt.ylabel('per spin expectation value of $S_x$')

plt.show()