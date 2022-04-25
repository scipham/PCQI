
import matplotlib.pyplot as plt
import numpy as np

from TFIM import TFIM
from TD_NQS_RBM import TD_NQS_RBM


init_H = TFIM(h=1/2, g=1/2)
target_H = TFIM(h=1, g=1)

delta_t = 0.05
end_of_time = 0.25


td_nqs_model = TD_NQS_RBM(init_H = init_H,
                            Nv = 10,
                            Nh = 40,
                            init_mode="ground_state",
                            init_state_params={"kContrastDiv": 6000, "lrate": 0.2, "epochs": 70})

energies, paulixvals = td_nqs_model.evolute_quench(target_H=target_H,
                            delta_t=delta_t,
                            end_of_time=end_of_time,
                            time_lrate=0.1, 
                            kContrastDiv=6000)

print(paulixvals)
plt.plot(np.arange(0,end_of_time,delta_t),paulixvals)
plt.plot(np.arange(0,end_of_time,delta_t),energies)
plt.show()