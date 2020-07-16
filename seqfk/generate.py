import numpy as np
from progress.bar import Bar

from data import OutputWriter, get_timestamp, write_callfile
from sequence import Sequence


def main(sequence=None,
         prep_steps=1000,
         prep_temp=None,
         samples=int(1e7),
         temp=1. / 3,
         save_every=10,
         seq_length=None,
         selected_iterator=None,
         p_mutation=None,
         sequencepotentials=('tilt', 'roll'),
         averagedpotentials=(),
         writer=OutputWriter,
         tension_deformation=0,
         hist_bounds=None,
         **kwargs):
    """
    Top level function for generating monte carlo data

    :param sequence: Sequence object fed in to start. If None, a random sequence is generated
    :param prep_steps: Number of steps to run before logging data
    :param prep_temp: Temperature at which to prepare
    :param samples: The number of samples to log
    :param temp: The temperature at which to sample in fraction of room temp 300 K
    :param save_every: Save every save_every sample when generating data
    :param seq_length: The length of the sequence 147 by default
    :param selected_iterator: The iterator to use. If none, creates MC iterator
    :param p_mutation: The chance of mutating the sequence during a step
    :param sequencepotentials: Which sequence-dependent-potentials to use. Tuple containing 'rise', 'tilt' and or 'roll'
    :param averagedpotentials: Which averaged potentials to use. Tuple containing 'rise', 'tilt' and or 'roll'
    :param writer: What writer class to use for logging
    :param tension_deformation: The deformation due to tension. Ignored if 0
    :param hist_bounds: The bin edges used for the histogram calculation in the writer class
    :param kwargs: Further key word arguments passed to the writer class, sequence creator class and stored in the call_file
    :return:
    """
    if prep_temp is None:
        prep_temp = temp

    timestamp = get_timestamp()

    # store a file with the function call for later reference
    write_callfile(timestamp=timestamp,
                   prep_steps=prep_steps,
                   prep_temp=prep_temp,
                   steps=samples,
                   temp=temp,
                   save_every=save_every,
                   seq_length=seq_length,
                   selected_iterator=selected_iterator,
                   p_mutation=p_mutation,
                   sequencepotentials=sequencepotentials,
                   averagedpotentials=averagedpotentials,
                   tension_deformation=tension_deformation,
                   hist_bounds=hist_bounds,
                   **kwargs
                   )

    s = Sequence(N=seq_length, selected_iterator=selected_iterator, p_mutation=p_mutation,
                 sequencepotentials=sequencepotentials,
                 averagedpotentials=averagedpotentials,
                 tension_deformation=tension_deformation,
                 **kwargs)

    if sequence is not None:
        s.positions = sequence.positions
        s.sequence = sequence.sequence

    if prep_steps > 0:  # pre-calculation
        s.update(steps=prep_steps, temp=prep_temp)

    interrupt_size = int(np.ceil(samples / 1000))  # how often to interrupt to update loading bar

    s_prev = s.sequence.copy()

    i = 0
    total = int(samples)
    try:
        with writer(timestamp=timestamp, hist_bounds=hist_bounds, **kwargs) as o:
            # with Liveplotter() as l:
            with Bar('Progress', max=samples, suffix='%(index)d/%(max)d [%(elapsed)d / %(eta)d / %(eta_td)s]') as bar:
                while True:
                    ds = s_prev - s.sequence
                    if i % interrupt_size == 0:
                        s.calc_energy()
                        bar.next(interrupt_size)
                        ds = s_prev - s.sequence
                        # print(s.energy)

                    o.write_line(s)

                    if i >= total - 1:
                        break

                    dE = s.update(steps=save_every, temp=temp)
                    i += 1
    except KeyboardInterrupt:
        print('Did', i, 'iterations')

    return s


if __name__ == '__main__':
    main(temp=1,
         samples=1000,
         p_mutation=0)
