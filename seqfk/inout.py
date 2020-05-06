import os
from pathlib import Path
import arrow

import numpy as np

from sequence import Sequence
from model import encode, decode

cwd = Path(os.getcwd())
data_path = cwd.joinpath('data/')

import dask.array as da
import dask

if not data_path.exists():
    data_path.mkdir()

datetime_format = 'YYYY-MM-DD_HH-mm-ss'

def get_timestamp():
    stamp = arrow.now()
    return stamp.format(datetime_format)

def arr_to_string(a):
    return str(a.tolist())[1:-1]

def get_file_paths():
    return os.listdir(data_path)

def full_path(paths):
    return [data_path.joinpath(i) for i in paths]

def get_unique_files():
    return full_path(
        ['_'.join(i.split('_')[:-1])
         for i in get_file_paths()]
    )

def get_unique_datetimes(paths):
    return {
        '_'.join(i.split('_')[:2])
        for i in paths
    }

def get_blob_paths(paths, filetype=None):
    if filetype is not None:
        filetype = '_{}_'.format(filetype)
    return [
        str(i)+filetype+"*.*" for i in get_unique_datetimes(paths)
    ]


class OutputWriter:
    def __init__(self, store_sequence=True, store_energy=True, store_position=False):
        self.store_sequence = store_sequence
        self.store_energy = store_energy
        self.store_position = store_position

        self._get_timestamp()
        self._sequences = []
        self._positions = []
        self._energies = []

    def _get_timestamp(self):
        self.timestamp = get_timestamp()

    def __enter__(self):
        t = self.timestamp
        self.i = 0
        # self.file = open(data_path.joinpath(t+'SeqFK.csv'), 'w+')
        # self.file.write('energy,sequence\n')
        # self.sequences = open(data_path.joinpath(t + 'sequences.csv'), 'w+')
        # self.positions = open(data_path.joinpath(t + 'positions.csv'), 'w+')
        # self.energies = open(data_path.joinpath(t + 'energies.csv'), 'w+')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (len(self._energies) >0) or (len(self._sequences)>0) or (len(self._positions)>0):
            self._write_lines()
        # self.file.close()
        # self.sequences.close()
        # self.positions.close()
        # self.energies.close()

    def write_line(self, s:Sequence):
        if self.store_sequence:
            self._sequences.append(s.sequence)
        if self.store_position:
            self._positions.append(arr_to_string(s.positions))
        if self.store_energy:
            self._energies.append(str(s.energy))

        if len(self._sequences) >= 10000:
            self._write_lines()

    @staticmethod
    def combine(strings):
        return ','.join([str(i) for i in strings])

    def get_fileformat(self):
        output = str(self.timestamp) + '_{}_' + str(self.i)
        self.i += 1
        return output

    def _write_lines(self):
        # self.file.writelines(
        #     '{}\n'.format(i) for i in
        #             [self.combine(i) for i in zip(self._energies, self._sequences)]
        # )
        # self.sequences.writelines('{}\n'.format(i) for i in self._sequences)
        # self.positions.writelines('{}\n'.format(i) for i in self._positions)
        # self.energies.writelines('{}\n'.format(i) for i in self._energies)

        f = self.get_fileformat()

        # dask.to_npy_stack only stores int as int64 and saves dtype in seperate file
        # we could manually store DNA using only 2 bits per base but that would cost
        # overhead as neither numpy nor dask support smaller units than bytes

        if self.store_sequence:
            np.save(data_path.joinpath(f.format('sequences')), self._sequences)
            self._sequences.clear()
        if self.store_position:
            np.save(data_path.joinpath(f.format('positions')), self._positions)
            self._positions.clear()
        if self.store_energy:
            np.save(data_path.joinpath(f.format('energies')), self._energies)
            self._energies.clear()

def get_sequences():
    paths=get_file_paths()
    blobs = get_blob_paths(paths, 'sequences')
    blobs.sort()
    files = data_path.glob(blobs[-1])

    # open

    chunks = [dask.delayed(np.load)(i) for i in files]
    chunks = [da.from_delayed(i, i.shape.compute(), i.dtype.compute()) for i in chunks]

    a = da.concatenate(chunks, axis=0)

    return a

if __name__ == '__main__':
    import dask.array as da
    import dask

    paths = get_file_paths()
    blobs = get_blob_paths(paths, 'sequences')
    blobs.sort()

    files = data_path.glob(blobs[-1])

    # open

    chunks = [dask.delayed(np.load)(i) for i in files]
    chunks = [da.from_delayed(i, i.shape.compute(), i.dtype.compute()) for i in chunks]

    a = da.concatenate(chunks, axis=0)




