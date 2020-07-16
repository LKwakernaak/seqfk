import numpy as np

import model
import monte_carlo


class Sequence():
    """
    Object that stores a sequence with positions with an update method to update the positions and sequence
    through a Markov Chain Monte Carlo Method.
    """

    def __init__(self, N=None, selected_iterator=None, p_mutation=None,
                 sequencepotentials=('rise'),
                 averagedpotentials=('tilt', 'roll'),
                 positions_perturbation_function=None,
                 sequence_perturbation_function=None,
                 tension_deformation=0,
                 **kwargs):
        self.positions_perturbation_function = positions_perturbation_function
        self.sequence_perturbation_function = sequence_perturbation_function

        if N is None:
            N = model.LENGTH

        if selected_iterator is None:
            selected_iterator = 'metropolis_sp'

        self.selected = selected_iterator

        if p_mutation is None:
            p_mutation = 0.2

        random_sequence = np.random.choice(np.arange(4, dtype=np.uint8), (N))
        self.sequence = random_sequence
        if tension_deformation is None:
            self.positions = np.linspace(0, N - 1, N)
        else:
            self.positions = np.linspace(0, N - 1, N)
        self.rest_positions = self.positions.copy()
        self.p_mutation = p_mutation

        self.sequencepotentials = sequencepotentials
        self.averagedpotentials = averagedpotentials

        self.tension_deformation = tension_deformation

        self._setup_settr()
        self._update_MC_func()
        self.calc_energy()

    def _setup_settr(self):
        def __setattr__(self, key, value):
            super().__setattr__(key, value)
            self._update_MC_func()

    def update(self, steps=100, temp=1):
        # print('updating')
        if self.selected.endswith('sp'):
            dE = self.MC_func(self.positions, self.sequence, steps, temp, self.p_mutation)
        else:
            dE = self.MC_func(self.positions, self.sequence, steps, temp)

        self.energy += dE

    @property
    def rise_potential(self):
        if 'rise' in self.averagedpotentials:
            return model.averaged_rise_potential(self.positions)
        elif 'rise' in self.sequencepotentials:
            return model.rise_potential(self.positions, self.sequence)
        else:
            return 0

    @property
    def tilt_potential(self):
        if 'tilt' in self.averagedpotentials:
            return model.averaged_tilt_potential(self.positions)
        elif 'tilt' in self.sequencepotentials:
            return model.tilt_potential(self.positions, self.sequence)
        else:
            return 0

    @property
    def roll_potential(self):
        if 'roll' in self.averagedpotentials:
            return model.averaged_roll_potential(self.positions)
        elif 'roll' in self.sequencepotentials:
            return model.roll_potential(self.positions, self.sequence)
        else:
            return 0

    @property
    def tension_energy(self):
        if self.tension_deformation is None:
            return 0
        else:
            return model.tension_energy(self.positions, self.tension_deformation)

    def calc_energy(self):
        self.energy = np.sum(self.rise_potential) + np.sum(
            [self.tilt_potential, self.roll_potential]) + self.tension_energy

    @property
    def MC_func(self):
        return self._MC_func

    def _update_MC_func(self):
        print('creating algorithm')
        algorithm, sp = str(self.selected).split('_')

        if algorithm == 'metropolis':
            self._MC_func = monte_carlo.Metropolis(sequencepotentials=self.sequencepotentials,
                                                   averagedpotentials=self.averagedpotentials,
                                                   tension_deformation=self.tension_deformation)
        if algorithm == 'checkerboard':
            self._MC_func = monte_carlo.Checkerboard(sequencepotentials=self.sequencepotentials,
                                                     averagedpotentials=self.averagedpotentials,
                                                     tension_deformation=self.tension_deformation)

        if algorithm == 'old':
            self._MC_func = monte_carlo.DoubleEnergyMetropolis()


if __name__ == '__main__':
    s = Sequence()

    import model as m

    print(s.energy)
    print(s.energy)

    import monte_carlo as mc

    energy_func = mc.create_energy_function(s.sequencepotentials, s.averagedpotentials)

    print(energy_func(s.positions, s.sequence))

    import matplotlib.pyplot as plt

    plt.plot(m.rise_potential(s.positions, s.sequence))

    s.update(steps=1e6, temp=1)

    plt.plot(m.rise_potential(s.positions, s.sequence))

    print('bla')
    print(s.energy)
    print(energy_func(s.positions, s.sequence))
