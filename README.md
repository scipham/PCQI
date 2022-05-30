# Course Project PCQI: Time-Dependent processes with Neural Quantum States

Course Project for Physics and Classical/Quantum Information 2022: 
Time-Dependent processes with Neural Quantum States: Computing Time Evolutions and Out-of-time correlation (OTOC) of Neural Quantum States (NQS) for the Transverse Field Ising Model (TFIM).

## Authors
Pim Veefkind (Studentnummer = 2324822)

Thomas Rothe (Studentnummer = 1930443)

## Simulation Data

The raw data from simulations of the report (and beyond) can be found in the subfolder: ./experiments/data/

## Usage of simulation code

The main folder of this readme file contains the core code components for the NQS Time Evolution, OTOC computation and plotting routines. The code can only be run by so-called experiment scripts (or experiments).
Note that 3.7 <= python version <= 3.9 is required!

### Run Prepared NQS simulation experiments:

 Prepared and modifiable experiment scripts can be found in the experiments folder. Those include the scripts used for simulating cases presented in figures of the accompanying report. The scripts can be run as usual with any python environment within the required python version range.

For example:
 ```bash
python ./experiments/basic_TD_exp.py
 ```
 
### Run custom NQS simulation experiments from scratch:
Create an own experiment (script) in the experiments subfolder and always include the following header:

 ```python
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
```

Subsequently one can create an hamiltonian object (only 1D TFIM currently supported):
```python
H = TFIM(h=0.5, g=0.5)
```

1. For time independent simulations / ground state calculations:
Create a NQS_RBM object and calculate, for example, the ground state:
```
nqs_model = NQS_RBM(...)
nqs_results = nqs_model.get_RBM_GS(...)

```

2. For time dependent simulations / time evolutions:
Create an TD_NQS_RBM object and run a time evolution with an output variable:
```
td_nqs_model = TD_NQS_RBM(...)
time_evol_output = td_nqs_model.evolute_quench(...)
```

3. For OTOC compuations:
Create an OTOC object and run the otoc computation:
```python
otoc_calculator = OTOC(...)
otoc_output = otoc_calculator.compute_efficient(...)
```


Optionally, you can include in any of these cases a NetKet and qutip computation within the same scripts!!

For example, you can get the time evoluted expectation value of mean sigma_x per spin:
```python
netket_obj = Netket(...)
netket_obj.calc_ground_state(...)
Sx_netket = netket_obj.quench_evolve(....)
Sx_qutip = netket_obj.quench_evolve_exact(...)
```


For further ideas / how-to's on including NetKet and Qutip: see the readily prepared experiments in the experiments folder.
