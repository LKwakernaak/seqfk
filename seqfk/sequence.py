import numpy as np

import model
import monte_carlo


class Sequence():
    def __init__(self, N=model.LENGTH):
        random_sequence = np.random.choice(np.arange(4, dtype=np.uint8), (N))
        self.sequence = random_sequence # A:0 C:1 G:2 T:3
        self.positions = model.zuiddam_positions(random_sequence)
        self.rest_positions = self.positions.copy()

    def update(self, *args, **kwargs):
        # self.positions = monte_carlo.metropolis_positions(self.positions, self.sequence, *args, **kwargs)
        # self.positions = monte_carlo.checkerboard_positions(self.positions, self.sequence, *args, **kwargs)
        # self.sequence = monte_carlo.metropolis_sequence(self.positions, self.sequence, *args, **kwargs)
        monte_carlo.metropolis_sequence_positions(self.positions, self.sequence, *args, **kwargs)
        # monte_carlo.checkerboard_sequence_positions(self.positions, self.sequence, *args, **kwargs)

    @property
    def chain_potential(self):
        return model.calc_chain_potential(self.positions, self.sequence)

    @property
    def tilt_potential(self):
        return model.effective_tilt_potential(self.positions)

    @property
    def roll_potential(self):
        return model.effective_roll_potential(self.positions)

    @property
    def energy(self):
        return np.sum(self.chain_potential) + np.sum([self.tilt_potential, self.roll_potential])