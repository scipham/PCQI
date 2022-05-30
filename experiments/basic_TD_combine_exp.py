#--------Always include this heading in experiments
import sys, os
from time import time
sys.path.append(os.path.join(sys.path[0],'..'))

from TFIM import *
from NQS_RBM import *
from TD_NQS_RBM import *
from PLOTTING import *
from netket_f.netket import Netket

np.random.seed(12)

EXPERIMENTS_PATH = sys.path[0]
RESULTS_PATH = EXPERIMENTS_PATH + "/data/" 
#-------------------------------------------------------

init_H = TFIM(h=0.5, g=0.5)
target_H = TFIM(h=0.55, g=0.55)
#init_H = TFIM(h=4, g=4)
#target_H = TFIM(h=2, g=2)

N_V = 10
N_H = 40 

init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.4, 
                     "epochs": 70}

evol_params = {'target_H': target_H,
            'delta_t': 0.5*0.2e-3, #0.2e-3,
            'end_of_time':0.24,
            'kContrastDiv': 8000,
            'reg_mode':'diag_shift',
            'reg_strength':1e-5,
            'val_fraction':0.1}


Sx_TD_exp_dict = {}
with open(RESULTS_PATH + "temp_result_1_copy.pickle",'rb') as f:
    temp_time = np.arange(0.0, 0.24, 0.2e-3)
    Sx_TD_exp_dict['dt = 2.0e-4'] = (temp_time, pickle.load(f)[4])#(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

with open(RESULTS_PATH + "temp_result_1_x0.5_copy.pickle",'rb') as f:
    temp_time = np.arange(0.0, 0.24, 0.5*0.2e-3)
    Sx_TD_exp_dict['dt = 1.0e-4'] = (temp_time, pickle.load(f)[4]) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)


time = np.arange(0,evol_params['end_of_time'], evol_params['delta_t'] )

#plot_time_dependent_exp_vals(time, time_evol_output[3],  time_evol_output[4]) 
#plot_time_evolution_errors(time, time_evol_output[6])

#Below everything, including Netket comparison:

N_chain = N_V

h_init = init_H.h
h_quench = target_H.h

dt = evol_params['delta_t']
end_of_time = evol_params['end_of_time'] #2 + 1e-8

'''
netket_obj = Netket(N_chain = N_chain)

netket_obj.calc_ground_state(h_init=h_init)
Sx_approx = netket_obj.quench_evolve(h_quench=h_quench, dt=dt, end_of_time=end_of_time)
Sx_exact = netket_obj.quench_evolve_exact(dt=dt, end_of_time=end_of_time)

with open(RESULTS_PATH + "temp_result_1_netket_qutip.npy", 'wb') as f:
    np.save(f, Sx_approx)
    np.save(f, Sx_exact)
'''

Sx_approx, Sx_exact = None, None
with open(RESULTS_PATH + "temp_result_1_netket_qutip.npy", 'rb') as f:
    Sx_approx = np.load(f)[:-1]
    Sx_exact = np.load(f)
    print(Sx_approx, Sx_exact)

plot_netket_quench(time, N_V, Sx_approx, Sx_exact, Sx_TD_exp_dict)
