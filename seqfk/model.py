import numba as nb
import numpy as np

from parameters import equilibrium, stiffness, ALPHA, LENGTH, R_EFF, GAMMA, LD, RLD


@nb.njit()
def average_angle(a, b):
    diff = ((a - b + np.pi * 3) % (np.pi * 2)) - np.pi
    angle = (np.pi * 2 + b + (diff / 2)) % (np.pi * 2)
    return angle


def encode(string, table=RLD):
    """
    Encode the string data DNA to an array.
    :param string: string representation:  AACGTG
    :param table: The dictionary used to translate representations
    :return: number representation:  [0,0,2,3,1,3]
    """
    array = np.empty(len(string), dtype=np.uint8)
    # print(string)
    for i, letter in enumerate(string):
        array[i] = table[letter]
    return array


def decode(array, table=LD):
    """
    Decode the array DNA to string DNA
    :param array: number representation:  [0,0,2,3,1,3]
    :param table: The dictionary used to translate representations
    :return: string representation:  AACGTG
    """
    string = ''
    # print(array)
    for i in array:
        string += table[i]
    return string


def encode_pair(string, table=RLD):
    """
    Encode a nucleosome pair to number representation
    :param string: string representation:  GC
    :param table: translation table
    :return: number representation: 14
    """
    output = table[string[0]] * 4 + table[string[1]]
    return output


@nb.njit()
def zuiddam_positions(indices):
    """
    Return the position in nm along the superhelix as a function of the dna sequence index.
    These positions are the resting position as defined by Martijn Zuiddam.

    :param indices: 1D numpy array of the floating indices
    :return: Positions in same dimensions as sequence
    """
    positions = np.empty(len(indices), dtype=np.float32)
    for i in range(len(indices)):
        positions[i] = indices[i] * (2 * np.pi * R_EFF * ALPHA) / (LENGTH - 1)
    return positions


@nb.njit()
def zuiddam_floating_index(positions):
    """
    Return the zuiddam position "P(s)" which is a floating point index of the bases.
    The inverse of zuiddam_positions(sequence)

    :param positions: 1D numpy array or constant
    :return: A floating point index in the same dimensions as positions
    """
    index = np.empty(len(positions))
    for i in range(len(index)):
        index[i] = positions[i] * (LENGTH - 1) / (2 * np.pi * R_EFF * ALPHA)
    return index


@nb.njit()
def q_roll(floating_index):
    """
    Calculate the roll as a function of the floating index
    :param floating_index: Array of indices
    :return: Roll of indices
    """
    roll = np.empty(len(floating_index), dtype=np.float64)
    for i in nb.prange(len(floating_index)):
        roll[i] = GAMMA * np.cos(2 * np.pi * floating_index[i] / 10 - 147 * np.pi / 10)
    return roll


@nb.njit()
def q_tilt(floating_index):
    """
    Calculate the tilt as a function of the floating index
    :param floating_index: Array of indices
    :return: Tilt of indices
    """
    tilt = np.empty(len(floating_index), dtype=np.float64)
    for i in nb.prange(len(floating_index)):
        tilt[i] = GAMMA * np.sin(2 * np.pi * floating_index[i] / 10 - 147 * np.pi / 10)
    return tilt


@nb.njit()
def rise_potential(floating_index, sequence):
    """
    Calculate the sequence dependent rise potential as a function of the floating index and given sequence
    :param floating_index: Array of indices
    :param sequence: Number representation sequence
    :return: Rise potential of DNA
    """
    potential = np.empty(len(floating_index) - 1)
    positions = zuiddam_positions(floating_index)
    for i in range(len(positions) - 1):
        dx = positions[i + 1] - positions[i]
        a, b = sequence[i:i + 2]
        stretch = dx - equilibrium[a, b, 2]
        springconstant = stiffness[a, b, 2, 2]
        potential[i] = 1 / 2 * springconstant * stretch ** 2
    return potential


@nb.njit()
def roll_potential(floating_index, sequence):
    """
    Calculate the sequence dependent roll potential as a function of the floating index and given sequence
    :param floating_index: Array of indices
    :param sequence: Number representation sequence
    :return: Roll potential of DNA
    """
    roll_angle = q_roll((floating_index[:-1] + floating_index[1:]) / 2)
    potential = np.empty(len(floating_index) - 1)

    for i in range(len(floating_index) - 1):
        a, b = sequence[i:i + 2]
        u = roll_angle[i] - equilibrium[a, b, 4]  # is roll angle actually an angle?
        springconstant = stiffness[a, b, 4, 4]
        potential[i] = .5 * springconstant * u ** 2

    return potential


@nb.njit()
def tilt_potential(floating_index, sequence):
    """
    Calculate the sequence dependent tilt potential as a function of the floating index and given sequence
    :param floating_index: Array of indices
    :param sequence: Number representation sequence
    :return: Tilt potential of DNA
    """

    tilt_angle = q_tilt((floating_index[:-1] + floating_index[1:]) / 2)
    potential = np.empty(len(floating_index) - 1)

    for i in nb.prange(len(floating_index) - 1):
        a, b = sequence[i:i + 2]
        u = tilt_angle[i] - equilibrium[a, b, 3]
        springconstant = stiffness[a, b, 3, 3]
        potential[i] = .5 * springconstant * u ** 2

    return potential


average_tilt_eq = np.average(equilibrium[:, :, 3])
average_tilt_stiff = np.average(stiffness[:, :, 3, 3])


@nb.njit()
def averaged_tilt_potential(floating_index):
    """
    Calculate the averaged tilt potential as a function of the floating index
    :param floating_index: Array of indices
    :return: Average tilt potential of DNA
    """
    tilt_angle = np.sin(2 * np.pi * floating_index / 10 - 147 * np.pi / 10) * GAMMA
    return 1 / 2 * average_tilt_stiff * (tilt_angle - average_roll_eq) ** 2


average_roll_eq = np.average(equilibrium[:, :, 4])
average_roll_stiff = np.average(stiffness[:, :, 4, 4])


@nb.njit()
def averaged_roll_potential(floating_index):
    """
    Calculate the averaged roll potential as a function of the floating index
    :param floating_index: Array of indices
    :return: Average roll potential of DNA
    """
    roll_angle = np.cos(2 * np.pi * floating_index / 10 - 147 * np.pi / 10) * GAMMA
    return 1 / 2 * average_roll_stiff * (roll_angle - average_roll_eq) ** 2


average_twist_eq = np.average(equilibrium[:, :, 5])
average_twist_stiff = np.average(stiffness[:, :, 5, 5])


@nb.njit()
def averaged_twist_potential(floating_index):
    """
    Calculate the averaged twist potential as a function of the floating index
    :param floating_index: Array of indices
    :return: Average twist potential of DNA
    """
    twist_angle = np.ones_like(floating_index) * 2 * np.pi / 10.17
    return 1 / 2 * average_twist_stiff * (twist_angle - average_roll_eq) ** 2


average_rise_eq = np.average(equilibrium[:, :, 2])
average_rise_stiff = np.average(stiffness[:, :, 2, 2])


@nb.njit()
def averaged_rise_potential(floating_index):
    """
    Calculate the averaged rise potential as a function of the floating index
    :param floating_index: Array of indices
    :return: Average rise potential of DNA
    """
    potential = np.empty(len(floating_index) - 1)
    positions = zuiddam_positions(floating_index)
    for i in range(len(floating_index) - 1):
        dx = positions[i + 1] - positions[i]
        stretch = dx - average_rise_eq
        springconstant = average_rise_stiff
        potential[i] = 1 / 2 * springconstant * stretch ** 2
    return potential


@nb.njit()
def collision_energy(floating_index):
    E = 0.
    for i in range(len(floating_index) - 1):
        if (floating_index[i + 1] - floating_index[i]) < 0:
            E += np.infty
    return E


@nb.njit()
def tension_energy(floating_index, deformation, force_constant=average_rise_stiff):
    """
    Calculate the tension energy when: two general elements or handles are connected to the ends of the dna are moved by deformation/2 away from the dna.
    :param floating_index: Indices of the DNA
    :param deformation: The distance with which the handles are moved
    :param springconstant: The springconstant of the handles
    :return: Energy due to handle stretch
    """
    # d = deformation/2 # stretch the string on both sides with half of the deformation

    positions = zuiddam_positions(floating_index)

    normal_ends = zuiddam_positions([0, 146])

    dx_1 = (normal_ends[0] - positions[0]) - deformation / 2
    dx_2 = (normal_ends[-1] - positions[-1]) + deformation / 2

    # constant force
    dE_1 = force_constant * dx_1 ** 2
    dE_2 = force_constant * dx_2 ** 2

    return dE_1 + dE_2
