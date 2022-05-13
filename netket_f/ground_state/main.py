from initial_ground_state_scripts.nqsrbm import NQSRBM

J = 1
B = 0.5
Nv = 10
Nh = 40
kContrastDiv = 6000
lrate = 0.4
epochs = 65

calculate_initial_ground_state = True

if calculate_initial_ground_state:
    NQSrun = NQSRBM(J,B,Nv,Nh,kContrastDiv,lrate,epochs)