#--------Always include this heading in experiments
import sys, os
from time import time
sys.path.append(os.path.join(sys.path[0],'..'))

from TFIM import *
from NQS_RBM import *
from TD_NQS_RBM import *
from PLOTTING import *

np.random.seed(12)

EXPERIMENTS_PATH = sys.path[0]
RESULTS_PATH = EXPERIMENTS_PATH + "/data/" 
#-------------------------------------------------------

init_H = TFIM(h=1/2, g=1/2)
target_H = TFIM(h=1, g=1)

delta_t = 0.005
end_of_time = 0.3


td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = 10,
                            Nh = 40,
                            init_mode="ground_state",
                            init_state_params={"kContrastDiv": 6000, "lrate": 0.4, "epochs": 70})

time_evol_output = td_nqs_model.evolute_quench(target_H=target_H,
                            delta_t=delta_t,
                            end_of_time=end_of_time,
                            kContrastDiv=6000,
                            reg_strength=0.0005)

plt.plot(np.arange(0,end_of_time,delta_t),time_evol_output[2]) #Pauli-x-mean
plt.plot(np.arange(0,end_of_time,delta_t),time_evol_output[0]) #Energies
plt.show()

#with open(RESULTS_PATH+"temp_result.pickle",'wb') as f:
#    pickle.dump(time_evol_output,f)

#time_evol_output = None
#with open(RESULTS_PATH + "temp_result.pickle",'rb') as f:
#    time_evol_output = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)

time = np.arange(0,end_of_time, delta_t )
plot_time_dependent_exp_vals(time, time_evol_output[0],  time_evol_output[1]) 
