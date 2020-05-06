import numba as nb
import numpy as np

import model as m


# from model import effective_roll_potential, effective_tilt_potential, calc_chain_potential

@nb.njit(fastmath=True)
def positions_perturbation(positions, sequence, i, stepsize, temp):

    size = len(positions)

    perturbation = (np.random.random_sample() - 0.5) * stepsize

    if i == 0:
        pos_slice = positions[:2]
        seq_slice = sequence[:2]
        perturbed = pos_slice.copy()
        perturbed[0] += perturbation
    elif i == size - 1:
        pos_slice = positions[-2:]
        seq_slice = sequence[-2:]
        perturbed = pos_slice.copy()
        perturbed[1] += perturbation
    else:
        pos_slice = positions[i - 1:i + 2]
        seq_slice = sequence[i - 1:i + 2]
        perturbed = pos_slice.copy()
        perturbed[1] += perturbation

    delta_E = np.sum(m.effective_roll_potential(perturbed) - m.effective_roll_potential(pos_slice))

    delta_E += np.sum(m.effective_tilt_potential(perturbed) - \
                      m.effective_tilt_potential(pos_slice))

    delta_E += np.sum(m.calc_chain_potential(perturbed, seq_slice) - \
                      m.calc_chain_potential(pos_slice, seq_slice))

    if delta_E < 0:
        pass

    elif np.random.rand() > np.exp(-1 * delta_E / temp):
        return False

    positions[i] += perturbation
    return True

@nb.njit(fastmath=True)
def sequence_perturbation(positions, sequence, i, temp):

    size = len(positions)
    perturbation = np.random.randint(4)

    if i == 0:
        pos_slice = positions[:2]
        seq_slice = sequence[:2]
        perturbed = np.empty_like(seq_slice)
        perturbed[:] = seq_slice
        perturbed[0] += perturbation
        perturbed[0] %= 4
    elif i == size - 1:
        pos_slice = positions[-2:]
        seq_slice = sequence[-2:]
        perturbed = np.empty_like(seq_slice)
        perturbed[:] = seq_slice
        perturbed[1] += perturbation
        perturbed[1] %= 4
    else:
        pos_slice = positions[i - 1:i + 2]
        seq_slice = sequence[i - 1:i + 2]
        perturbed = np.empty_like(seq_slice)
        perturbed[:] = seq_slice
        perturbed[1] += perturbation
        perturbed[1] %= 4

    delta_E = np.sum(m.calc_chain_potential(pos_slice, perturbed) - \
                     m.calc_chain_potential(pos_slice, seq_slice))

    if delta_E < 0:
        # print('no')
        pass

    elif np.random.rand() > np.exp(-1 * delta_E / temp):
        # print('yes')
        return False

    sequence[i] = (sequence[i] + perturbation) % 4
    return True

@nb.njit(fastmath=True)
def metropolis_positions(positions, sequence, steps=100, temp=1):
    """
    A sequential algorithm for updating our sequence
    :return:
    """
    size = len(positions)
    step = 0

    stepsize = np.sqrt(temp) / 15

    for step in range(steps):

        i = np.random.randint(size)

        while not positions_perturbation(positions, sequence, i, stepsize, temp):
            pass

    return True

@nb.njit(fastmath=True)
def metropolis_sequence(positions, sequence, steps=100, temp=1):
    size = len(positions)

    for step in range(steps):

        i = np.random.randint(size)

        while not sequence_perturbation(positions, sequence, i, temp):
            pass

    return True

@nb.njit()
def metropolis_sequence_positions(positions, sequence, steps=100, temp=1, p_mutation=0.2):
    for i in range(steps):
        if np.random.rand() < p_mutation:
            metropolis_sequence(positions, sequence, steps=1, temp=temp)
        else:
            metropolis_positions(positions, sequence, steps=1, temp=temp)

@nb.njit(parallel=True, fastmath=True)
def checkerboard_positions(positions, sequence, steps=1, temp=1):
    """
    The checkerboard_positions version of the metropolis_positions algorithm.

    :param positions:
    :param sequence:
    :param steps:
    :param temp:
    :return:
    """

    size = len(positions)
    stepsize = np.sqrt(temp)/15

    for step in range(steps):
        for zero_one in (0, 1):
            for i in nb.prange(size//2):
                i = zero_one+i*2

                while not positions_perturbation(positions, sequence, i, stepsize, temp):
                    pass


    return True

@nb.njit(parallel=True, fastmath=True)
def checkerboard_sequence(positions, sequence, steps=1, temp=1):

    size = len(positions)
    stepsize = np.sqrt(temp) / 15

    for step in range(steps):
        for zero_one in (0, 1):
            for i in nb.prange(size // 2):
                i = zero_one + i * 2

                while not sequence_perturbation(positions, sequence, i, temp):
                    pass

@nb.njit(fastmath=True)
def checkerboard_sequence_positions(positions, sequence, steps=1, temp=1, p_mutation=0.2):
    size = len(positions)
    stepsize = np.sqrt(temp) / 15

    for step in range(steps):
        for zero_one in (0, 1):
            for i in nb.prange(zero_one, size, 2):
                i += zero_one

                if np.random.rand() < p_mutation:
                    while not sequence_perturbation(positions, sequence, i, temp):
                        pass

                else:
                    while not positions_perturbation(positions, sequence, i, stepsize, temp):
                        pass