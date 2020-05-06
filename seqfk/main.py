import matplotlib.pyplot as plt
import numpy as np

from sequence import Sequence
from viewer import potential_plot

from tools.grace import GracefullKiller

from inout import OutputWriter

def histogram(sequences):
    # sequences = np.vstack(sequences)

    keys = np.unique(sequences)

    output = {
        int(i):np.sum(sequences==i, axis=0) for i in keys
    }

    # A = np.sum(sequences == 0, axis=1)
    # C = np.sum(sequences == 1, axis=1)
    # G = np.sum(sequences == 2, axis=1)
    # T = np.sum(sequences == 3, axis=1)

    return output


def hist_plot(sequences):
    hist = histogram(sequences)
    for key in hist.keys():
        plt.plot(hist[key], label=key)
    # plt.plot(A, label='A')
    # plt.plot(C, label='C')
    # plt.plot(G, label='G')
    # plt.plot(T, label='T')
    plt.legend()
    plt.show()

# np.lib.stride_tricks.as_strided(np.vstack(sequences), (50000, 146, 2), (147,1,1))
def pair_view(sequence, n=2):
    sequence = np.asarray(sequence)
    strides = sequence.strides
    shape = sequence.shape

    if len(shape) == 1:
        return np.lib.stride_tricks.as_strided(sequence, (shape[0]+1-n, n), (strides[0], strides[0]))

    elif len(shape) == 2:
        return np.lib.stride_tricks.as_strided(sequence, (shape[0], shape[1]+1-n, n), (strides[0], strides[-1], strides[-1]))

def codons(sequence):
    return pair_view(sequence, n=3)

def encode_pairs(pairs, base=4):
    shape = pairs.shape
    indices = shape[-1]
    weights = np.eye(indices)*(base**np.arange(indices-1,-1, -1))

    weights = np.asarray(weights, dtype=np.int8)

    return np.einsum('kl,ijk->ij', weights, pairs)

def large_vstack(s):
    empty = np.empty([len(s), len(s[0])], dtype=s[0].dtype)
    for i in range(len(s)):
        empty[i] = s[i]

    return empty


if __name__ == '__main__':
    print('starting')

    s = Sequence()
    # v1 = Viewer(s)

    # potential_plot(s)

    old = s.sequence.copy()

    energies = []


    print(s.sequence)

    sequences = []

    s.update(temp=10, steps=1000)

    # killer = GracefullKiller()

    i = 0
    total = int(1e8)
    try:
        with OutputWriter() as o:
            while True:
                o.write_line(s)
                # energies.append(s.energy)
                # sequences.append(s.sequence.copy())
                s.update(temp=0.1, steps=10)
                if i%10000 == 0:
                    print(i, "{:05.2f} %".format(100*i/total))
                i += 1
                if i >= total: break
    except KeyboardInterrupt:
        print('Did', i, 'iterations')

    # plt.plot(energies)
    # plt.semilogx()
    # plt.show(block=True)


    # sequences_stack = large_vstack(sequences)

    # print(s.sequence)

    # print(np.isclose(old, new))

    # potential_plot(s)
    # hist_plot(sequences)

    # pairs = pair_view(sequences_stack)
    # encoded_pairs = encode_pairs(pairs)

    # hist_plot(encoded)
    # hist = histogram(encoded_pairs)
    #
    # size = len(pairs)
    #
    #
    # plt.plot(hist[9]/size, label='GC')
    # plt.plot((hist[0] + hist[12] + hist[15])/size, label='AA+TA+TC')
    # plt.legend()
    # plt.xlabel("Basepairs")
    # plt.ylabel('Frequency')
    # plt.show()
