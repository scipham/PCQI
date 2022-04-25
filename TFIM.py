from dataclasses import dataclass, field

@dataclass
class TFIM:
    h: float #Transverse field strength
    g: float #Ratio of transverse field / interaction strengths
    J: str = field(init=False) #Implied interaction strength
    
    def __post_init__(self):
        self.J = self.h / self.g