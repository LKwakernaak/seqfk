from pathlib import Path

import numpy as np

pardir = Path('Parameterization/')

stiffiles = [i for i in pardir.rglob("Stiffness-C-*")]
eqfiles = [i for i in pardir.rglob("Equilibrium-MD-*")]


LD = lookup_dict = {
    0: "A",
    1: "C",
    2: "G",
    3: "T"
}

RLD = reverse_lookup_dict = {
    "A" : 0,
    "C" : 1,
    "G" : 2,
    "T" : 3
}

pairs = [
           "AA",
           "AC",
           "AG",
           "AT",
           "CA",
           "CC",
           "CG",
           "CT",
           "GA",
           "GC",
           "GG",
           "GT",
           "TA",
           "TC",
           "TG",
           "TT"
]

ALPHA = 1.84 # The number of superhelical turns
LENGTH = 147
RADIUS = 4.19 # The superhelical radius in nm
PITCH = 2.59 # The superhelical pitch in nm
R_EFF = np.sqrt(RADIUS**2 + (PITCH/2/np.pi)**2) # The effective radius
GAMMA = 0.0796

def find_pairstring(string):
    return str(string).split(".")[-2].split('-')[-1]

stiffness = np.empty([4, 4, 6, 6]) # ACGT, ACGT, xyzXYZ xyzXYZ
equilibrium = np.empty([4, 4, 6])  # ACGT, ACGT xyzXYZ

for string in stiffiles:
    pair = find_pairstring(string)
    try:
        stiffness[RLD[pair[0]], RLD[pair[1]], :, :] = np.roll(np.genfromtxt(string), 3, [0, 1])
    except KeyError:
        pass # Non-ACGT

for string in eqfiles:
    pair = find_pairstring(string)
    try:
        equilibrium[RLD[pair[0]], RLD[pair[1]], :] = np.roll(np.genfromtxt(string), 3)
    except KeyError:
        pass # Non-ACGT

