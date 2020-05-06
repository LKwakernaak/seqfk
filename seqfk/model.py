import numba as nb
import numpy as np

from parameters import equilibrium, stiffness, ALPHA, LENGTH, RADIUS, PITCH, R_EFF, GAMMA

# ALPHA = 1.84 # The number of superhelical turns
# LENGTH = 147
# RADIUS = 41.9 # The superhelical radius
# PITCH = 25.9 # The superhelical pitch
# R_EFF = np.sqrt(RADIUS**2 + (PITCH/2/np.pi)**2) # The effective radius

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

pair_lookup = {
    "AA": 0,
    "AC": 1,
    "AG": 2,
    "AT": 3,
    "CA": 4,
    "CC": 5,
    "CG": 6,
    "CT": 7,
    "GA": 8,
    "GC": 9,
    "GG": 10,
    "GT": 11,
    "TA": 12,
    "TC": 13,
    "TG": 14,
    "TT": 15
}

q_roll = np.array([
    0.012410451,    #AA
    0.012372536,    #AC
    0.079562987,    #AG
    0.019409417,    #AT
    0.083496463,    #CA
    0.063703201,    #CC
    0.095824007,    #CG
    0.079562987,    #CT
    0.03372236,     #GA
    0.0053117746,   #GC
    0.063703201,    #GG
    0.012372536,    #GT
    0.058653564,    #TA 
    0.03372236,     #TC
    0.083496463,    #TG
    0.012410451     #TT
])

Q_roll = np.array([
    126.98464,      #AA
    148.42141,      #AC
    143.15931,      #AG
    123.91326,      #AT
    73.527282,      #CA
    126.98464,      #CC
    113.06128,      #CG
    97.396194,      #CT
    97.396194,      #GA
    123.91326,      #GC
    130.1586,       #GG
    83.019248,      #GT
    113.06128,      #TA
    143.15931,      #TC
    146.67053,      #TG
    130.1586        #TT
])

q_tilt = np.array([
    -0.024820902,   #AA
    -0.0017675051,  #AC
    -0.030057128,   #AG
    0,              #AT
    0.0088826025,   #CA
    0.0017695334,   #CC
    0,              #CG
    0.030057128,    #CT
    -0.026622916,   #GA
    0,              #GC
    -0.0017695334,  #GG
    -0.0017695334,  #GT
    0,              #TA
    0.026622916,    #TC
    -0.0088826025,  #TG
    0.024820902     #TT
])

Q_tilt = np.array([
    207.73324,      #AA
    216.86174,      #AC
    221.16218,      #AG
    200.28179,      #AT
    129.10674,      #CA
    207.73324,      #CC
    210.62471,      #CG
    146.17762,      #CT
    146.17762,      #GA
    200.28179,      #GC
    225.01953,      #GG
    150.88272,      #GT
    210.62471,      #TA
    221.16218,      #TC
    214.38125,      #TG
    225.01953       #TT
])

nb.njit()
def apply(f_vector, sequence):
    out = np.empty(len(sequence)-1, dtype=f_vector.dtype)
    for i in range(len(sequence)-1):
        pair = sequence[i]+ sequence[i+1]*4
        out[i] = f_vector[i]
    return out

def encode(string, table=RLD):
    array = np.empty(len(string), dtype=np.uint8)
    # print(string)
    for i,letter in enumerate(string):
        array[i] = table[letter]
    return array

def decode(array, table=LD):
    string = ''
    # print(array)
    for i in array:
        string += table[i]
    return string

def encode_pair(string, table=RLD):
    output = table[string[0]]*4 + table[string[1]]
    return output

def calc_tangent_positions(s):
    return np.sqrt(
        (RADIUS * np.cos(s/R_EFF))**2 +
        (RADIUS * np.sin(s/R_EFF))**2 +
        (s*PITCH/2/np.pi/R_EFF)**2)


# class Sequence():
#     def __init__(self, N=LENGTH):
#         random_sequence = np.random.choice(np.arange(4, dtype=np.uint8), (N))
#         self.sequence = random_sequence # A:0 C:1 G:2 T:3
#         self.positions = zuiddam_positions(random_sequence)
#         self.rest_positions = self.positions.copy()
#
#     def calc_E(self):
#         pass
#
#     def calc_E_roll(self):
#         self.E_roll = 1/2 * Q
#
#     def update(self):
#         self.positions = metropolis_positions(self.positions, self.sequence)


@nb.njit()
def equilibrium_chain_positions(sequence):
    positions = np.empty(len(sequence), dtype=np.float32)
    positions[0] = 0
    for i in range(1, len(sequence)):
        a, b = sequence[i-1: i+1]
        positions[i] = equilibrium[a, b, 2] + positions[i-1]

    return positions

@nb.njit()
def zuiddam_positions(sequence):
    """
    Return the position in nm along the superhelix as a function of the dna sequence index.
    These positions are the resting position as defined by Martijn Zuiddam.

    :param sequence: 1D numpy array or constant
    :return: Positions in same dimensions as sequence
    """
    positions = np.empty(len(sequence), dtype=np.float32)
    for i in range(len(sequence)):
        positions[i] = (i-0.5)*(2*np.pi*R_EFF*ALPHA)/(LENGTH - 1)
    return positions

@nb.njit()
def zuiddam_floating_index(positions):
    """
    Return the zuiddam position "P(s)" which is a floating point index of the bases.
    The inverse of zuiddam_positions(sequence)

    :param positions: 1D numpy array or constant
    :return: A floating point index in the same dimensions as positions
    """
    index = np.empty(len(positions), dtype=np.float32)
    for i in range(len(index)):
        index[i] = positions[i] * (LENGTH - 1)/(2*np.pi * R_EFF * ALPHA) + 1/2
    return index

@nb.njit()
def calc_chain_force(positions, sequence):
    force = np.empty_like(positions)
    for i in range(len(positions)-1):
        dx = positions[i+1] - positions[i]
        a, b = sequence[i:i+2]
        stretch = dx - equilibrium[a, b, 2]
        springconstant = stiffness[a, b, 2, 2]
        force[i] = springconstant*stretch
    return force

@nb.njit()
def calc_chain_potential(positions, sequence):
    potential = np.empty(len(positions)-1)
    for i in range(len(positions) - 1):
        dx = positions[i + 1] - positions[i]
        a, b = sequence[i:i + 2]
        stretch = dx - equilibrium[a, b, 2]
        springconstant = stiffness[a, b, 2, 2]
        potential[i] = 1/2*springconstant * stretch**2
    return potential

def calc_roll_force(positions, sequence):
    roll_angle = np.cos(2*np.pi*positions/10 - 147*np.pi/10)
    for i in range(len(positions)):
        a,b = sequence[i:i+2]
        u = roll_angle - equilibrium[a, b, 5]
        springconstant = stiffness[a, b, 5, 5]
        torque = u*springconstant


average_tilt_eq = np.average(equilibrium[:,:,3])
average_tilt_stiff = np.average(stiffness[:,:,3,3])
@nb.njit()
def effective_tilt_potential(positions):
    positions = zuiddam_floating_index(positions)
    tilt_angle = np.sin(2*np.pi*positions/10 - 147*np.pi/10)*GAMMA
    return 1/2 * average_tilt_stiff*(tilt_angle-average_roll_eq)**2

average_roll_eq = np.average(equilibrium[:,:,4])
average_roll_stiff = np.average(stiffness[:,:,4,4])
@nb.njit()
def effective_roll_potential(positions):
    positions = zuiddam_floating_index(positions)
    roll_angle = np.cos(2*np.pi*positions/10 - 147*np.pi/10)*GAMMA
    return 1/2 * average_roll_stiff*(roll_angle-average_roll_eq)**2

average_twist_eq = np.average(equilibrium[:,:,5])
average_twist_stiff = np.average(stiffness[:,:,5,5])
@nb.njit()
def effective_twist_potential(positions):
    positions = zuiddam_floating_index(positions)
    twist_angle = np.ones_like(positions)*2*np.pi/10.17
    return 1/2 * average_twist_stiff*(twist_angle-average_roll_eq)**2



def parametric_position(position_along_superhelix):
    return np.asarray([
        RADIUS * np.cos(position_along_superhelix / R_EFF),
        RADIUS * np.sin(position_along_superhelix / R_EFF),
        - (PITCH/2/np.pi/R_EFF) * position_along_superhelix
    ]).T

def frenetserret(position_along_superhelix):
    """
    Returns the frenetserret vectors along the superhelix.

    :param position_along_superhelix: the s position along the superhelix in nm
    :return: A len(S) x [x,y,z] x [t,n,b] shape
    """
    s = position_along_superhelix
    tangent = np.asarray([
        -RADIUS/R_EFF * np.sin(s/R_EFF),
        RADIUS/R_EFF * np.cos(s/R_EFF),
        - (PITCH/2/np.pi/R_EFF)*np.ones_like(s)
    ]).T

    normal = np.asarray([
        np.cos(s/R_EFF),
        np.sin(s/R_EFF),
        np.zeros_like(s)
    ]).T

    binormal = np.cross(tangent, normal)

    return np.dstack([tangent, normal, binormal])

def rotation_along_superhelix(position_along_superhelix):
    index = zuiddam_floating_index(position_along_superhelix)

    x = np.asarray([
        np.zeros_like(index),
        np.cos(np.pi * 2 / 10 * index - 147 * np.pi / 10),
        -np.sin(np.pi * 2 / 10 * index - 147 * np.pi / 10)
    ])

    y = np.asarray([
        np.zeros_like(index),
        -np.sin(np.pi*2/10*index - 147*np.pi/10),
        -np.cos(np.pi*2/10*index - 147*np.pi/10)
    ])

    z = np.asarray([
        np.ones_like(index),
        np.zeros_like(index),
        np.zeros_like(index)
    ])

    return np.transpose(np.dstack([x.T,y.T,z.T]), (0,2,1))

def rotation_matrix_along_superhelix(position_along_superhelix):
    return rotation_along_superhelix(position_along_superhelix) @ \
           frenetserret(position_along_superhelix)


def is_rotation(array):
    weight = np.linalg.det(array)

    return (np.isclose(np.linalg.inv(array), np.transpose(array, (0,2,1))) * np.isclose(weight, 1)[:, None, None])


if __name__ == '__main__':
    def diff(A, B):
        for i, letter in enumerate(A):
            if B[i] != letter:
                return False
        return True

    s = "ACGTCGTACGTCGGGTATAAAT"
    print(s)
    encoded_s = encode(s)
    print("encoded_s", encoded_s)
    decoded_s = decode(encoded_s)
    print("decoded_s", decoded_s)
    print("Matches", diff(s, decoded_s))

    s = Sequence()
    print(calc_chain_force(s.sequence, s.positions))

    print(equilibrium_chain_positions(s.sequence))
    print(zuiddam_positions(s.sequence))

    rot = rotation_along_superhelix(s.sequence)


