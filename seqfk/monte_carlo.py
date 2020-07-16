import numba as nb
import numpy as np


@nb.njit()
def get_local(positions, sequence, i):
    """
    Get a local slice of positions and sequence around i
    :param positions: The positions along the superhelical in floating index "units"
    :param sequence: The sequence of the superhelical DNA
    :param i: The position at which to slice
    :return: A slice of 2 or 3 positions, the 2 or 3 long sequence, the local index of the element to mutate.
    """
    if i == 0:
        return positions[:2].copy(), sequence[:2].copy(), 0

    elif i == len(positions):
        return positions[-2:].copy(), sequence[-2:].copy(), 1

    else:
        return positions[i - 1:i + 2].copy(), \
               sequence[i - 1:i + 2].copy(), \
               1


@nb.njit()
def flat_chance(p):
    return np.random.rand() < p


@nb.njit()
def boltzmann_chance(E, temp):
    return np.random.rand() < np.exp(-E / temp)


@nb.njit()
def random_sequence_mutation(sequence, i):
    mutation = np.random.randint(1, 4)
    sequence_mutation(sequence, i, mutation)


@nb.njit()
def sequence_mutation(sequence, i, mutation):
    sequence[i] += mutation
    sequence[i] %= 4


@nb.njit()
def random_positions_mutation(positions, i, temp):
    mutation = (0.5 - np.random.rand()) * np.sqrt(temp) / 150
    positions_mutation(positions, i, mutation)


@nb.njit()
def positions_mutation(positions, i, mutation):
    positions[i] += mutation


@nb.njit()
def duplicate(local_p, local_s):
    local_p_new, local_s_new = local_p.copy(), local_s.copy()  # make a copy of the subsequence to test a mutation on
    return local_p_new, local_s_new


@nb.njit()
def even(size):
    for i in range(size):
        if i % 2 == 0:
            yield i


@nb.njit()
def odd(size):
    for i in range(size):
        if i % 2 != 0:
            yield i


class MonteCarloAlgorithm():
    message = 'setting up Monte-Carlo'

    def __init__(self,
                 sequencepotentials=('rise',),
                 averagedpotentials=('roll', 'tilt'),
                 tension_deformation=0):
        self._sequencepotentials = sequencepotentials
        self._averagedpotentials = averagedpotentials
        self._tension = tension_deformation
        self._setup_energy_func()
        print(self.message)

    def _setup_energy_func(self):
        """
        Generate a jitted energy function based on the potentials requested.
        :return: Energy function.
        """
        funcs = []
        if 'rise' in self._sequencepotentials:
            funcs.append('np.sum(m.rise_potential(positions, sequence))')
        if 'tilt' in self._sequencepotentials:
            funcs.append('np.sum(m.tilt_potential(positions, sequence))')
        if 'roll' in self._sequencepotentials:
            funcs.append('np.sum(m.roll_potential(positions, sequence))')

        if 'rise' in self._averagedpotentials:
            funcs.append('np.sum(m.averaged_rise_potential(positions))')
        if 'tilt' in self._averagedpotentials:
            funcs.append('np.sum(m.averaged_tilt_potential(positions))')
        if 'roll' in self._averagedpotentials:
            funcs.append('np.sum(m.averaged_roll_potential(positions))')

        if self._tension is not None:
            funcs.append('m.tension_energy(positions, {})'.format(self._tension))
            funcs.append('m.collision_energy(positions)')

        body = '+'.join(funcs)
        expression = 'lambda positions, sequence: ' + body
        self.energy = nb.njit()(eval(expression))
        return self.energy

    @staticmethod
    def energy(positions, sequence):  # placeholder for the energy function
        pass

    @staticmethod
    def get_element(steps, size):  # placeholder
        """
        Generate the elements to mutate.
        :param steps: How many steps to generate mutations for.
        :param size: The size of the DNA for which to generate the mutations.
        :return:
        """
        pass

    def __call__(self, positions, sequence, steps=1000, temp=1, p=0.2):
        delta_E = 0
        size = len(positions)

        for i in self.get_element(steps, size):
            delta_E += self._mutate(positions, sequence, i, temp, p)

        return delta_E

    def _mutate(self, positions, sequence, i, temp, p_mut):
        """
        The mutation function for the Metropolis Algorithm
        :param positions: The positions along the superhelical in floating index "units"
        :param sequence: The sequence of the superhelical DNA
        :param i: The position at which to slice
        :param temp: The temperature at which to run the simulation
        :param p_mut: The chance to mutate the sequence.
        :return: The change in energy due to the mutation.
        """

        mut_seq = flat_chance(p_mut)  # whether to mutate the sequence or the positions
        local_p, local_s, local_i = get_local(positions, sequence,
                                              i)  # a small window concerning only the elements that will be edited
        tries = 0  # keep track of the number of tries for debugging
        # while True:
        local_p_new, local_s_new = duplicate(local_p, local_s)

        if mut_seq:
            random_sequence_mutation(local_s_new, local_i)
        else:
            random_positions_mutation(local_p_new, local_i, temp)

        dE = self.energy(local_p_new, local_s_new) - self.energy(local_p, local_s)

        if dE < 0:  # always accept lowering in energy
            pass
        elif boltzmann_chance(dE, temp):  # or accept given the energy difference and the temperature
            pass
        else:
            return 0

            # tries += 1
        sequence[i] = local_s_new[local_i]  # when successfull copy the modifation
        positions[i] = local_p_new[local_i]

        return dE


class Metropolis(MonteCarloAlgorithm):
    message = 'setting up metropolis'

    @staticmethod
    @nb.njit()
    def get_element(steps, size):
        for step in range(steps):
            yield np.random.randint(size)


@nb.njit()
def parallel_slices(positions, sequence):
    """
    Generate slices/views of all the indices in order.
    :param positions: The positions along the superhelical in floating index "units"
    :param sequence: The sequence of the superhelical DNA
    :return: An array of get_local output
    """
    size = len(positions)
    slices = []
    for i in range(size):
        slices.append((get_local(positions, sequence, i)))
    return slices


@nb.njit()
def checkerboard_inner(slice, p_mutation, temp, energy):
    """
    The inner function to run in the _outer function of the checkerboard mc.
    Different modifications can be performed in parallel but this function allows all permutations to be saved.

    :param slice:  local_p, local_s, local_index. Slices of either odd or even locations
    :param p_mutation: chance of mutating the sequence
    :param temp: temperature at which to run the simulation
    :param energy: Energy calculation function
    :return: mut_seq, dE, i, mutation
    """
    mut_seq = flat_chance(p_mutation)  # whether to mutate the sequence or the positions
    local_p, local_s, local_index = slice
    tries = 0  # keep track of the number of tries for debugging
    # while True:
    local_p_new, local_s_new = duplicate(local_p, local_s)

    if mut_seq:
        mutation = np.random.randint(1, 4)
        sequence_mutation(local_s_new, local_index, mutation)
    else:
        mutation = (0.5 - np.random.rand()) * np.sqrt(temp) / 150
        positions_mutation(local_p_new, local_index, mutation)

    dE = energy(local_p_new, local_s_new) - energy(local_p, local_s)

    if dE < 0:  # always accept lowering in energy
        pass
    elif boltzmann_chance(dE, temp):  # or accept given the energy difference and the temperature
        pass
    else:  # if no suitable mutation is found, skip location
        mutation = 0
        dE = 0

        # tries += 1

    return mut_seq, dE, mutation


@nb.njit(parallel=True)
def checkerboard_outer(positions, sequence, temp, p_mutation, energy):
    """
    Generate a mutation on all positions and the entire sequence using the metropolis algorithm.
    :param positions: The positions along the superhelical in floating index "units"
    :param sequence: The sequence of the superhelical DNA
    :param p_mutation: chance of mutating the sequence
    :param temp: temperature at which to run the simulation
    :param energy: Energy calculation function
    :return:
        (
        mut_seq,      # True when the mutation modifies the sequence
        dE,         # Change in energy due to mutation
        mutation    # The value used for mutation.
        )
    """
    slices = parallel_slices(positions, sequence)

    size = len(positions)

    dE = np.empty(size, dtype=np.float32)
    mutation = np.empty(size, dtype=np.float32)
    mut_seq = np.empty(size, dtype=np.uint8)

    # odd
    for i in nb.prange(size // 2):
        index = i * 2 + 1
        _mut_seq, _dE, _mutation = checkerboard_inner(slices[index], p_mutation, temp, energy)
        mut_seq[index] = _mut_seq
        dE[index] = _dE
        mutation[index] = _mutation

    # even
    for i in nb.prange(size // 2 + size % 2):
        index = i * 2
        _mut_seq, _dE, _mutation = checkerboard_inner(slices[index], p_mutation, temp, energy)
        mut_seq[index] = _mut_seq
        dE[index] = _dE
        mutation[index] = _mutation

    return mut_seq, dE, mutation


class Checkerboard(MonteCarloAlgorithm):
    message = 'setting up checkerboard'

    def __call__(self, positions, sequence, steps=1000, temp=1, p_mutation=0.2):
        """
        Mimic the behaviour of the Metropolis class by caching intermediate parallel mutations.
        :return: The change in Energy due to the mutation.
        """

        if not hasattr(self, 'counter'):
            self.counter = 0
            # generate a set of len(positions) mutations. One for every position.
            self.generator = self._mutate(positions, sequence, temp, p_mutation, self.energy)

        while True:
            delta_E = 0
            dE, mut_seq, mutation, i = next(self.generator)  # generate mutations
            if mut_seq:  # sequence mutation
                sequence[i] += int(mutation)
                sequence[i] %= 4
            else:  # sequence mutations
                positions[i] += mutation
            delta_E += dE
            self.counter += 1
            if i == len(sequence) - 1:  # end of generator. Get a new one
                self.generator = self._mutate(positions, sequence, temp, p_mutation, self.energy)
            if self.counter % steps == 0:
                return delta_E

    @staticmethod
    def _mutate(positions, sequence, temp, p_mutation, energy):
        """
        Generator function that yields mutations.
        """
        # Generate and store len(positions) mutations
        mut_seq, dE, mutation = checkerboard_outer(positions, sequence, temp, p_mutation, energy)

        # Unpack the mutations
        for i in range(len(mut_seq)):
            yield dE[i], mut_seq[i], mutation[i], i


class DoubleEnergyMetropolis(Metropolis):
    def _setup_energy_func(self):
        """
        Set up the energy functions used to compare in the case of a position or sequence mutation.
        """
        m1 = Metropolis(
            sequencepotentials=('rise'),
            averagedpotentials=('tilt', 'roll'))
        m2 = Metropolis(
            sequencepotentials=('rise'),
            averagedpotentials=())

        self.energy_positions = m1.energy
        self.energy_sequence = m2.energy
        self.energy = m1.energy

    def __call__(self, positions, sequence, steps=1000, temp=1, p_mut=0.2):
        """
        :param positions: The positions along the superhelical in floating index "units"
        :param sequence: The sequence of the superhelical DNA
        :param steps: The number of mutation to make
        :param i: The position at which to slice
        :param temp: The temperature at which to run the simulation
        :param p_mut: The chance to mutate the sequence.
        :return: The change in energy due to the mutation.
        :return: The change in energy due to the mutation.
        """
        delta_E = 0
        size = len(positions)

        for i in self.get_element(steps, size):
            mut_seq = flat_chance(p_mut)  # whether to mutate the sequence or the positions
            local_p, local_s, local_i = get_local(positions, sequence,
                                                  i)  # a small window concerning only the elements that will be edited

            tries = 0  # keep track of the number of tries for debugging

            while True:
                local_p_new, local_s_new = local_p.copy(), local_s.copy()  # make a copy of the subsequence to test a mutation on

                if mut_seq:
                    random_sequence_mutation(local_s_new, local_i)
                    dE = self.energy_sequence(local_p_new, local_s_new) - self.energy_sequence(local_p, local_s)
                else:
                    random_positions_mutation(local_p_new, local_i, temp)
                    dE = self.energy_positions(local_p_new, local_s_new) - self.energy_positions(local_p, local_s)

                if dE < 0:  # always accept lowering in energy
                    break
                elif boltzmann_chance(dE, temp):  # or accept given the energy difference and the temperature
                    break

                tries += 1

            sequence[i] = local_s_new[local_i]  # when successfull copy the modifation
            positions[i] = local_p_new[local_i]

            delta_E += dE  # keep track of the total energy change

        return delta_E


if __name__ == '__main__':
    from sequence import Sequence

    s = Sequence()
    sequence = s.sequence
    positions = s.positions

    print('testing get_local')

    print(positions[:3], sequence[:3])
    print(get_local(positions, sequence, 0))
    print(positions[9:12], sequence[9:12])
    print(get_local(positions, sequence, 10))
    print(positions[-2:], sequence[-2:])
    print(get_local(positions, sequence, len(positions)))
    print('')

    metropolis = Metropolis()
    metropolis(positions, sequence, steps=1, temp=1)

    print('big one')

    metropolis = Metropolis(sequencepotentials=('roll', 'tilt'), averagedpotentials=())

    p1 = positions.copy()
    E_1 = metropolis.energy(positions, sequence)
    delta_E = metropolis(positions, sequence, steps=int(1e5), temp=1)
    E_2 = metropolis.energy(positions, sequence)
    p2 = positions.copy()

    dp = np.sqrt(np.sum(np.square(p2 - p1)))

    print('delta_E', delta_E)
    print(E_2 - E_1, dp)

    print('big two')

    metropolis = DoubleEnergyMetropolis()

    p1 = positions.copy()
    E_1 = metropolis.energy(positions, sequence)
    delta_E = metropolis(positions, sequence, steps=int(1e5), temp=1)
    E_2 = metropolis.energy(positions, sequence)
    p2 = positions.copy()

    dp = np.average(p2 - p1)

    print('delta_E', delta_E)
    print(E_2 - E_1, dp)
