import numpy as np
import pickle

from .metropolissampler import MetropolisSamp
from .weightupdatesmoother import WeightUpdateSmoothed

def NQSRBM(J,B,Nv,Nh,kContrastDiv,lrate,epochs):
    # Service message
    print("""\
        Neural Quantum State of the transverse field Ising model:
        Ising model parameters J, B: %f, %f
        Number of visible spins: %i
        Number of hidden spins: %i
        Monte Carlo sequence size: %i
        Learning Rate: %f
        Epochs: %i
        """ %(J,B,Nv,Nh,kContrastDiv,lrate,epochs))
    #
    # Initializing weights with real values between -1 and 1
    # The system is VERY sensitive to initial conditions. 
    # E.g. it will not converge if all weights are negative.
    #
    W0 = (0.2)*(2*np.random.rand(Nh,Nv)-1. +np.random.rand(Nh,Nv)*1j)
    a0 = (0.1)*(2*np.random.rand(Nv)-1. + np.random.rand(Nv)*1j)
    c0 = (0.1)*(2*np.random.rand(Nh)-1. + np.random.rand(Nh)*1j)
    #W0 = np.random.normal(size=(Nh,Nv))/1e4
    #a0 = np.random.normal(size=(Nv))/10
    #c0 = np.random.normal(size=(Nh))/1e4
    W0 = np.real(W0)
    a0 = np.real(a0)
    c0 = np.real(c0)
    #
    # Initialing visible spins with either 0 or 1
    #
    V0 = np.random.choice([0,1],Nv)
    #
    Magnetization = np.sum(V0)-Nv/2
    #while Magnetization > 0:
    #    site = np.random.randint(Nv)
    #    print('Flip-site', site)
    #    if V0[site] > 0 :
    #        V0[site] = - V0[site] + 1 
    #    Magnetization = np.sum(V0)-Nv/2
    #while Magnetization < 0:
    #    site = np.random.randint(Nv)
    #    print('Flip-site', site)
    #    if V0[site] == 0:
    #        V0[site] = - V0[site] + 1 
    #    Magnetization = np.sum(V0)-Nv/2
    #Magnetization = np.sum(V0)-Nv/2
    print('Magnetization Initial state: ', Magnetization)
    #
    # Learning/Variational Minimization cycle
    #
    V = np.copy(V0)
    W = np.copy(W0)
    a = np.copy(a0)
    c = np.copy(c0)
    #
    # The transverse field Ising model happens to
    # be exactly solvable through other means.
    # We secretly know the exact GS energy:
    #
    g = B/J
    FreeFermionModes = np.sqrt(1 + g**2-2*g*np.cos(2*np.pi*np.arange(Nv)/Nv)) 
    EexactPerSite = -J*np.sum(FreeFermionModes)/Nv #Number of modes on each site * energy of occupation = interaction energy
    #
    # Variable Initialization for plotting results
    #
    Convergence = np.array([[1,1]])
    Percentage = np.array([0])
    prct = 0
    #
    for ep in range(epochs):
        #
        Vensemble,prct = MetropolisSamp(W,a,c,V,kContrastDiv) #Get  representative samples
        W,a,c,EExpVal = WeightUpdateSmoothed(J,B,W,a,c,Vensemble,lrate,ep) #Update paramters by fixed paramter gradients on ensemble
        EVarPerSite = np.real(EExpVal)/Nv
        Convergence = np.append(Convergence,np.array([[ep,EVarPerSite]]),axis=0)
        Percentage = np.append(Percentage,np.array([prct]),axis=0)
        #lrate = lrate * 0.95 
        print('\rEpoch %i/%i: Variational Energy: %f, Exact Energy: %f ' %(ep+1,epochs,EVarPerSite,EexactPerSite), end='')
        if not np.abs(EVarPerSite) < 10e6:
            print('\nNumerical Runaway: discontinuing...')
            break
        #print('Weights updated: Started learning epoch %i out of %i\n' %(ep+1,epochs))
    
    WRBM = np.copy(W)
    aRBM = np.copy(a)
    cRBM = np.copy(c)
    sampler = 'MetroSmoothed'
    filename = f'NQSdata_J{J:01}_h{B:01}_{sampler}_Cycles{kContrastDiv}_Epochs{epochs}.pickle'
    print('\nFile = ', filename)
    results = (Convergence, Percentage, aRBM, cRBM, WRBM, EexactPerSite)
    with open(filename,'wb') as f:
        pickle.dump(results,f)
        
    return results