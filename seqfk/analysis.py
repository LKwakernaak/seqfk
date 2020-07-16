import os
from pathlib import Path

import dask
import numba as nb
import numpy as np

from data import DataSet, delayed_load, get_callfiles

file_d = os.path.dirname(__file__)
cwd = Path(file_d)
results_path = cwd.joinpath('results/')

if not results_path.exists():
    results_path.mkdir()


@nb.stencil
def _to_pairs(encoded_sequence):
    """
    Hidden sub-method of to_pairs that leaves a tail element.
    """
    return encoded_sequence[0, 0] * 4 + encoded_sequence[0, 0 + 1]


@nb.njit()
def to_pairs(encoded_sequence):
    """
    Stencilled function that translates the sequence to sequence pairs
    :param encoded_sequence:
    :return:
    """
    return _to_pairs(encoded_sequence)[:, :-1]


@nb.njit()
def histogram(sequences):
    """
    Calculate the histogram of the sequences
    :param sequences: Array containing number encoded sequences
    :return: Histogram with implied bins of the sequence locations.
    """
    sequences = np.asarray(sequences)
    shape = sequences.shape
    output = np.zeros((16, shape[1]))

    for sequence_index in range(shape[0]):
        for i, value in enumerate(sequences[sequence_index, :]):
            output[value, i] += 1

    return output


def sum_hists(histograms):
    """
    Sum like-sized histograms into one histogram.
    :param histograms: Iterable of histograms
    :return: Single histogram
    """
    h = np.zeros_like(histograms[0])
    for i in histograms:
        h += i
    return h


def store_histograms(path, array):
    array = np.asarray(array).T
    list = array.tolist()
    output = [("{}\t" * len(list[i]) + '\n').format(*list[i]) for i in range(len(list))]
    with open(path, 'w') as f:
        f.writelines(output)


def main():
    """
    Store all Sequence histograms of all datasets found in data.
    """

    callfiles = get_callfiles(None)
    data = [DataSet(c) for c in callfiles]

    for d in data:
        if d.analysed or len(d) < 2:
            continue

        print('analyzing', d.path)
        delayed_loaded_files = delayed_load(d.sequences)
        delayed_pairs = [dask.delayed(to_pairs)(sequences) for sequences in delayed_loaded_files]
        delayed_hists = [dask.delayed(histogram)(pairs) for pairs in delayed_pairs]

        from dask.diagnostics import ProgressBar

        with ProgressBar():
            hist = dask.delayed(sum_hists)(delayed_hists).compute()

        hist /= np.sum(hist[:, 0])

        name = str(d.callfile).split('.')[0] + '.dat'
        output_path = results_path.joinpath(name)
        store_histograms(output_path, hist)


if __name__ == '__main__':
    main()
