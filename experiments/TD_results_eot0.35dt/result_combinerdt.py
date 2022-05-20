import numpy as np
import matplotlib.pyplot as plt

dt = [0.01,0.001,0.005,0.0005]
end_of_time = 0.35

reg1e4dt01 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e4dt01.txt', dtype = complex)
reg1e4dt001 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e4dt001.txt', dtype = complex)
reg1e4dt005 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e4dt005.txt', dtype = complex)
reg1e4dt0005 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e4dt0005.txt', dtype = complex)
reg1e5dt01 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e5dt01.txt', dtype = complex)
reg1e5dt001 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e5dt001.txt', dtype = complex)
reg1e5dt005 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e5dt005.txt', dtype = complex)
reg1e5dt0005 = np.loadtxt('experiments/TD_results_eot0.35dt/reg1e5dt0005.txt', dtype = complex)

netket = np.load('experiments/TD_results_eot0.35c/netket.npy')
qutip = np.load('experiments/TD_results_eot0.35c/qutip.npy')

data = [reg1e4dt01,reg1e4dt001,reg1e4dt005,reg1e4dt0005,reg1e5dt01,reg1e5dt001,reg1e5dt005,reg1e5dt0005]
labels = ['reg = 1e-4, dt = 0.01','reg = 1e-4, dt = 0.001','reg = 1e-4, dt = 0.005','reg = 1e-4, dt = 0.0005',\
          'reg = 1e-5, dt = 0.01','reg = 1e-5, dt = 0.001','reg = 1e-5, dt = 0.005','reg = 1e-5, dt = 0.0005',]

for i, dat in enumerate(data):

    time = np.arange(0,end_of_time,dt[i%4])
    plt.plot(time[:-1], dat, label = labels[i])


plt.ylim(0.15,0.35)

plt.legend()
plt.grid()

plt.xlabel('time in (a.u.))')
plt.ylabel('per spin expectation value of $S_x$')

plt.show()