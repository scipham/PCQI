import numpy as np

from .lnrmbwavefunction import LnRMBWavefunction

def Elocal(J,B,W,a,c,V):
    #
    # Computing the wavefunction for state V
    #
    L = V.shape[0]
    LnPsi = LnRMBWavefunction(W,a,c,V)
    LnPsiBar = np.conj(LnPsi)
    #
    # Computing the energy for state V
    # First the Ising term
    #
    Vshift = np.array([V[(i+1)%L] for i in range(L)])
    One = np.ones(L)
    ElocalJ = -J*(np.sum((2*V-One)*(2*Vshift-One)))
    #
    # Next the magnetic term -B\sum_i \sigma^x_i
    # Because this is not diagonal on the
    # states, we compute 
    # <V|EB|Psi> instead
    # The action of Sigma^x_i is
    # to flip the spin on site i:
    # i.e. map V[i] to -V[i]+1
    #
    EBlocalPsi = 0
    for i in range(L):
        V[i] = -V[i]+1
        EBlocalPsi = EBlocalPsi - B*np.exp(LnRMBWavefunction(W,a,c,V)-LnPsi) #Compare flipped with unflipped (sigma_x applied)
        V[i] = -V[i]+1
    
    ElocalPsi = ElocalJ + EBlocalPsi
    
    return ElocalPsi