import os
from pathlib import Path

import numpy as np

file_d = os.path.dirname(__file__)

pardir = Path(file_d).joinpath('Parameterization/')

stiffiles = [i for i in pardir.rglob("Stiffness-MD-*")]
eqfiles = [i for i in pardir.rglob("Equilibrium-C-*")]

assert len(stiffiles) > 0
assert len(eqfiles) > 0


def reverse_dict(dict):
    return {value: key for (key, value) in dict.items()}


LD = lookup_dict = {
    0: "A",
    1: "T",
    2: "C",
    3: "G"
}

RLD = reverse_lookup_dict = reverse_dict(LD)

pair_lookup = {
    "AA": 0,
    "AT": 1,
    "AC": 2,
    "AG": 3,
    "TA": 4,
    "TT": 5,
    "TC": 6,
    "TG": 7,
    "CA": 8,
    "CT": 9,
    "CC": 10,
    "CG": 11,
    "GA": 12,
    "GT": 13,
    "GC": 14,
    "GG": 15
}

reverse_pair_lookup = reverse_dict(pair_lookup)

ALPHA = 1.84  # The number of superhelical turns
LENGTH = 147
RADIUS = 4.19  # The superhelical radius in nm
PITCH = 2.59  # The superhelical pitch in nm
R_EFF = np.sqrt(RADIUS ** 2 + (PITCH / 2 / np.pi) ** 2)  # The effective radius
GAMMA = 0.0796


def find_pairstring(string):
    return str(string).split(".")[-2].split('-')[-1]


stiffness = np.empty([4, 4, 6, 6])  # ATCG, ATCG, xyzXYZ xyzXYZ so position first and rotation after
equilibrium = np.empty([4, 4, 6])  # ATCG, ATCG xyzXYZ

for string in stiffiles:
    pair = find_pairstring(string)
    try:
        stiffness[RLD[pair[0]], RLD[pair[1]], :, :] = np.roll(np.genfromtxt(string), 3, [0, 1])  # files are XYZxyz
    except KeyError:
        pass  # Non-ACGT

for string in eqfiles:
    pair = find_pairstring(string)
    try:
        equilibrium[RLD[pair[0]], RLD[pair[1]], :] = np.roll(np.genfromtxt(string), 3)  # files are XYZxyz
    except KeyError:
        pass # Non-ACGT


