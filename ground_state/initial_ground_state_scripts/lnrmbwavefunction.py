import numpy as np

def LnRMBWavefunction(W,a,c,V):
    #
    # Golden rule of numerics: avoid exponentials.
    # Use ln's instead.
    #
    Wsummed = 0
    LnPreFactor = 0
    L = V.shape[0]
    for s in range(L):
        Wsummed = Wsummed + W[:,s]*V[s]
        LnPreFactor = LnPreFactor - a[s]*V[s]
    
    # Difference between bits 0 and 1 and spins -1 and 1
    LnPrePreFactor = np.sum(a)/2 + np.sum(c)/2+np.sum(W)/4
    AngleFactor = np.prod(1+np.exp(-c - Wsummed))
    LnPsiRMB = LnPrePreFactor + LnPreFactor + np.log(AngleFactor)
    return LnPsiRMB