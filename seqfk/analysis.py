import dask.array as da
import dask.dataframe as dd

from inout import get_file_paths, full_path
from model import encode
import numba as nb
import numpy as np

import matplotlib.pyplot as plt

def load():
    files = full_path(get_file_paths())
    file = files[-1]

    df = dd.read_csv(file)
    return df


def encode_sequence(sequences):
    return sequences.map(encode, meta=(None, 'u1'))

@nb.stencil
def _to_pairs(encoded_sequence):
    return encoded_sequence[0,0]*4 + encoded_sequence[0,0+1]

@nb.njit()
def to_pairs(encoded_sequence):
    return _to_pairs(encoded_sequence)[:, :-1]

def histogram(sequences):
    # sequences = np.vstack(sequences)
    keys = np.arange(16)
    output = {
        int(i):np.sum(sequences==i, axis=0) for i in keys
    }
    return output

if __name__ == '__main__':
    from inout import get_sequences

    seq = get_sequences()
    pairs = seq.map_blocks(to_pairs, dtype='uint8')

    print(pairs.compute)

    hist = histogram(pairs)


    size = len(pairs)

    plt.plot((hist[9] / size).compute(), label='GC')
    plt.plot((hist[0] + hist[12] + hist[15]).compute() / size, label='AA+TA+TC')
    plt.legend()
    plt.xlabel("Basepairs")
    plt.ylabel('Frequency')
    plt.show()

    #sequences = sequences.map(encode, meta=(None, 'u1'))
    # files = full_path(get_file_paths())
    # file = files[-1]

    # df = dd.read_csv(file)

    # sequences = df.sequence.map(encode, meta=(None, 'u1'))

    # print(sequences.head())

    # pairs = sequences.apply(to_pairs, meta=(None, 'u1'))

    # print(pairs.head())

    # pair_array = da.stack(pairs)


    # histogram(pair_array)

    # hist = histogram(pair_array).compute()

    # hist = np.apply_along_axis(np.bincount, 1, pair_array).visualize('graph.png')

    # print(hist)

    # size = len(pairs)

    # plt.plot(hist[9] / size, label='GC')
    # plt.plot((hist[0] + hist[12] + hist[15]) / size, label='AA+TA+TC')
    # plt.legend()
    # plt.xlabel("Basepairs")
    # plt.ylabel('Frequency')
    # plt.show()