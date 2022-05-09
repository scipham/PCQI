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

hamilt = TFIM(h=4.0, g=4.0)

nqs_model = NQS_RBM(hamilt = hamilt,
                    Nv = 10,
                    Nh = 40)

nqs_results = nqs_model.get_RBM_GS(kContrastDiv = 10000,
                              lrate = 0.1,
                              epochs = 100,
                              reg_mode = 'diag_shift',
                              reg_strength=0.004)


#with open(RESULTS_PATH+"temp_result.pickle",'wb') as f:
#    pickle.dump(nqs_results,f)

#nqs_results = None
#with open(RESULTS_PATH + "temp_result.pickle",'rb') as f:
#    nqs_results = pickle.load(f) #(Convergence, Percentage, aRBM, cRBM, WRBM, E_exact_per_site)


Convergence,Percentage, aRBM, cRBM, WRBM, EexactPerSite = nqs_results

#
# Displaying analytics
#
plot_time_independent_convergence(Convergence, Percentage, EexactPerSite)