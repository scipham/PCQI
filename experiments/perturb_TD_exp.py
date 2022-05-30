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

N_V = 10
N_H = 40 

init_state_params = {"kContrastDiv": 6000, 
                     "lrate": 0.4, 
                     "epochs": 70}

evol_params = {'amplitude': 0.3,
            'ang_freq':34.0,
            'offset': 0.12,
            'decay_length': 0.03,
            'delta_t': 0.4e-3,
            'end_of_time':0.25,
            'kContrastDiv': 14000,
            'reg_mode':'diag_shift',
            'reg_strength':1e-4,
            'val_fraction':0.1}

td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = N_V,
                            Nh = N_H,
                            init_mode="ground_state",
                            init_state_params=init_state_params)

time_evol_output = td_nqs_model.evolute_periodic_perturb(**evol_params, required_paulis = [[f"X{s}" for s in range(N_V)],[f"Z{s}" for s in range(N_V)]])

with open(RESULTS_PATH+"temp_pp_result_r6.pickle",'wb') as f:
    pickle.dump(time_evol_output,f)

#Plot first the raw perturbation itself:
time = np.arange(0,evol_params['end_of_time'], evol_params['delta_t'] )
amplitude=evol_params['amplitude']
ang_freq=evol_params['ang_freq']
offset=evol_params['offset']
decay_length=evol_params['decay_length']
dt=evol_params['delta_t']
end_of_time=evol_params['end_of_time']
raw_perturb = amplitude*np.sin(ang_freq*time)*np.exp(-(time-offset)**2 / (2*decay_length)) + init_H.h
plt.plot(time, raw_perturb)
plt.show()

#plot_time_dependent_exp_vals(time, time_evol_output[3],  time_evol_output[4]) plot_time_evolution_errors(time, time_evol_output[6])

#Below everything, including Netket comparison:

N_chain = N_V

h_init = init_H.h

dt = evol_params['delta_t']
end_of_time = evol_params['end_of_time'] #2 + 1e-8

netket_obj = Netket(N_chain = N_chain)

netket_obj.calc_ground_state(h_init=h_init)
Sx_approx, Sx_exact = netket_obj.periodic_perturb_evolve(amplitude=evol_params['amplitude'],ang_freq=evol_params['ang_freq'], offset=evol_params['offset'],decay_length=evol_params['decay_length'], dt=dt, end_of_time=end_of_time)

#with open(RESULTS_PATH+"temp_netket_pp_result.pickle",'wb') as f:
#    pickle.dump((Sx_approx, Sx_exact),f)

nqs_results = None
#with open(RESULTS_PATH + "temp_netket_pp_result.pickle",'rb') as f:
#    nqs_results = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)



plot_netket_quench(time, N_V, Sx_approx, Sx_exact)#, time_evol_output[4])

