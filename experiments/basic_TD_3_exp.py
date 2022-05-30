#--------Always include this heading in experiments
from pickle import NEXT_BUFFER
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


N_V = 10
N_H = 40 

init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.4, 
                     "epochs": 70}

evol_params = {'target_H': target_H,
            'delta_t': 1e-6,
            'end_of_time':0.008,
            'kContrastDiv': 9000,
            'reg_mode':'trunc_spec',
            'reg_strength':1e-5,
            'val_fraction':0.1}

td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = N_V,
                            Nh = N_H,
                            init_mode="ground_state",
                            init_state_params=init_state_params)

time_evol_output = td_nqs_model.evolute_quench(**evol_params, required_paulis = [[f"X{s}" for s in range(N_V)],[f"Z{s}" for s in range(N_V)]])

with open(RESULTS_PATH+"temp_result_3_sprint.pickle",'wb') as f:
    pickle.dump(time_evol_output,f)

#time_evol_output = None
#with open(RESULTS_PATH + "temp_result_5_copy.pickle",'rb') as f:
#    time_evol_output = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

time = np.arange(0,evol_params['end_of_time'], evol_params['delta_t'] )

#plot_time_dependent_exp_vals(time, time_evol_output[3],  time_evol_output[4]) plot_time_evolution_errors(time, time_evol_output[6])

#Below everything, including Netket comparison:

N_chain = N_V

h_init = init_H.h
h_quench = target_H.h

dt = evol_params['delta_t']
end_of_time = evol_params['end_of_time'] #2 + 1e-8

netket_obj = Netket(N_chain = N_chain)

netket_obj.calc_ground_state(h_init=h_init)
Sx_approx = netket_obj.quench_evolve(h_quench=h_quench, dt=dt, end_of_time=end_of_time)
Sx_exact = netket_obj.quench_evolve_exact(dt=dt, end_of_time=end_of_time)

plot_netket_quench(time, N_V, Sx_approx, Sx_exact, time_evol_output[4])
