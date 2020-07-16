import os
from pathlib import Path

import arrow
import dask
import dask.array as da
import numba as nb
import numpy as np

from sequence import Sequence

file_d = os.path.dirname(__file__)
cwd = Path(file_d)
data_path = cwd.joinpath('data/')
results_path = cwd.joinpath('results/')

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
        str(i) + filetype + "*.npy" for i in get_unique_datetimes(paths)
    ]


def write_callfile(timestamp, **kwargs):
    """
    Writes keywords from kwargs to a .call file
    """
    with open(str(data_path.joinpath(timestamp)) + '.call', 'w') as f:
        f.write('SeqFK log file parameters\n')
        for key in kwargs:
            if type(kwargs[key]) == type(np.empty(0)):
                f.write(str(key) + " = " + str(kwargs[key].tolist()) + '\n')
            else:
                f.write(str(key) + " = " + str(kwargs[key]).replace('\n', '') + '\n')


def read_callfile(path):
    """
    Reads a .call file to a dictionary
    """
    with open(path, 'r') as f:
        f.readline()
        it = f.readlines()
        kwargs = {}
        for line in it:
            line = line.strip().replace('\n', '')
            key, value = line.split('=')
            key = key.lstrip().rstrip()
            value = value.lstrip().rstrip().replace(' ', '')
            try:
                kwargs[key] = eval(value)
            except NameError:
                kwargs[key] = value
            except SyntaxError:
                kwargs[key] = value
        return kwargs


def get_callfiles(globstring=None):
    """
    Find paths of callfiles usign either the globstring or the current directory.
    """
    if globstring is None:
        paths = get_file_paths()
        globbed = [i for i in paths if i.endswith('.call')]
    else:
        globbed = [i for i in data_path.rglob(globstring) if str(i).endswith('.call')]
    return globbed


class DataSet:
    """
    High level object to group together different files into one dataset.
    """

    def __init__(self, path=None, index=None, globstring=None):
        if path is not None:
            self.path = Path(path)
        else:
            self.path = path
        self.index = index
        self.globstring = globstring
        self.datasets = None

        self._find_callfile()
        self.kwargs = read_callfile(data_path.joinpath(self.callfile))

        self._find_data_paths()

        self._find_results()

    def __len__(self):
        return len(self.datapaths)

    def _find_callfile(self):
        if str(self.path).endswith('.call'):
            self.callfile = self.path
            return
        elif self.globstring is not None:
            globbed = get_callfiles(self.globstring)
            if len(globbed) == 1:
                self.callfile = globbed[0]
                return
            elif len(globbed) > 1:
                self.datasets = [DataSet(path=i) for i in globbed]
        elif self.index is None:
            self.index = -1

        paths = get_file_paths()
        paths = [i for i in paths if i.endswith('.call')]
        paths.sort()
        self.callfile = paths[self.index]

    def _find_data_paths(self):
        self.datapaths = []
        if self.datasets is not None:
            [self.datapaths.extend(i.datapaths) for i in self.datasets]
        else:
            globstring = str(self.callfile).rsplit('.', maxsplit=1)[0] + '*'
            [self.datapaths.append(i) for i in data_path.rglob(globstring) if not str(i).endswith('.call')]
            self.datapaths = sorted(self.datapaths, key=lambda i: int(str(i).split('.')[-2].split('_')[-1]))

    def _find_results(self):
        self.result_files = []
        if self.datasets is not None:
            [self.result_files.extend(i) for i in self.datapaths]
        else:
            globstring = str(self.callfile).rsplit('.', maxsplit=1)[0] + '*.dat'
            [self.result_files.append(i) for i in results_path.rglob(globstring)]

    @property
    def analysed(self):
        return len(self.result_files) > 0

    @property
    def sequences(self):
        return [i for i in self.datapaths if 'sequences' in str(i)]

    @property
    def energies(self):
        return [i for i in self.datapaths if 'energies' in str(i)]

    @property
    def positions(self):
        return [i for i in self.datapaths if 'positions' in str(i)]

    @property
    def basic_positions(self):
        return [i for i in self.datapaths if 'basic_positions' in str(i)]

    @property
    def dx_histograms(self):
        return [i for i in self.datapaths if 'histograms' in str(i)]


class OutputWriter:
    """
    Base class for storing files.
    """

    def __init__(self, timestamp=None, store_sequence=True, store_energy=True, store_positions=False,
                 store_basic_positions=True, **kwargs):
        self.store_sequence = store_sequence
        self.store_energy = store_energy
        self.store_positions = store_positions
        self.store_basic_positions = store_basic_positions

        if timestamp is None:
            self._get_timestamp()
        else:
            self.timestamp = timestamp

        self._sequences = []
        self._positions = []
        self._energies = []
        self._basic_positions = []

        import threading, queue

        self.joblist = queue.Queue(maxsize=10)

        def writer():
            for filename, buffer in iter(self.joblist.get, None):
                # print('writing',data_path.joinpath(filename))
                if filename.endswith('.npy'):
                    np.save(data_path.joinpath(filename), buffer)
                elif filename.endswith('.seq'):
                    save_sequences(data_path.joinpath(filename), buffer)
                else:
                    raise NameError

        self.thread = threading.Thread(target=writer, name='writerthread')
        self.thread.setDaemon(True)
        self.thread.start()

        print(f"Started writing {self.timestamp}")

    def _get_timestamp(self):
        self.timestamp = get_timestamp()

    def __enter__(self):
        t = self.timestamp
        self.i = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (len(self._energies) > 0) or (len(self._sequences) > 0) or (len(self._positions) > 0):
            self._write_lines()

        self.joblist.put(None)
        self.thread.join(3.0)

    def write_line(self, s: Sequence):
        if self.store_sequence:
            self._sequences.append(s.sequence.copy())
        if self.store_positions:
            self._positions.append(s.positions.copy())
        if self.store_energy:
            self._energies.append(s.energy.copy())
        if self.store_basic_positions:
            self._basic_positions.append(
                min_max_average(s)
            )

        if len(self._sequences) >= 100000:
            self._write_lines()

    @staticmethod
    def combine(strings):
        return ','.join([str(i) for i in strings])

    def get_fileformat(self):
        output = str(self.timestamp) + '_{}_' + str(self.i) + "{}"
        self.i += 1
        return output

    def _write_lines(self):
        f = self.get_fileformat()

        # dask.to_npy_stack only stores int as int64 and saves dtype in seperate file
        # we could manually store DNA using only 2 bits per base but that would cost
        # overhead as neither numpy nor dask support smaller units than bytes

        if self.store_sequence:
            fname = f.format('sequences', '.seq')
            buffer = self._sequences.copy()
            # print('put', fname, len(buffer))
            self.joblist.put(
                (fname, buffer)
            )
            self._sequences.clear()
        if self.store_positions:
            fname = f.format('positions', '.npy')
            buffer = self._positions.copy()
            # print('put', fname, len(buffer))
            self.joblist.put(
                (fname, buffer)
            )
            self._positions.clear()
        if self.store_energy:
            fname = f.format('energies', '.npy')
            buffer = self._energies.copy()
            # print('put', fname, len(buffer))
            self.joblist.put(
                (fname, buffer)
            )
            self._energies.clear()
        if self._basic_positions:
            fname = f.format('basic_positions', '.npy')
            buffer = self._basic_positions.copy()
            # print('put', fname, len(buffer))
            self.joblist.put(
                (fname, buffer)
            )
            self._basic_positions.clear()


class StoredxWriter(OutputWriter):
    def __init__(self, **kwargs):
        self._dx_hists = []
        self.hist_edges = kwargs['hist_bounds']
        super(StoredxWriter, self).__init__(**kwargs)

    def _write_lines(self):
        super(StoredxWriter, self)._write_lines()

        f = self.get_fileformat()

        fname = f.format('histograms', '.npy')

        # merge the histograms
        shape = len(self._dx_hists), len(self.hist_edges) - 1
        buffer = np.empty(shape, dtype=np.float32)
        for i, (h, bins) in enumerate(self._dx_hists):
            buffer[i, :] = h

        # print('put', fname, len(buffer))
        self.joblist.put(
            (fname, buffer)
        )
        self._dx_hists.clear()

    def write_line(self, s: Sequence):
        if self.store_sequence:
            self._sequences.append((s.sequence))
        if self.store_energy:
            self._energies.append(s.energy)

        self._dx_hists.append(np.histogram(np.diff(s.positions), bins=self.hist_edges))

        if len(self._sequences) >= 100000:
            self._write_lines()


def min_max_average(s):
    return np.min(s.positions), np.max(s.positions), np.average(s.positions)


def sequence_pack(sequence):
    sequence = np.asarray(sequence)
    assert len(sequence.shape) == 1
    unpacked = np.unpackbits(sequence).reshape((-1, 8))
    assert np.all(unpacked[:, 0:6] == 0)
    without_zeros = unpacked[:, 6:]
    return np.packbits(without_zeros.flatten())


def pack_sequences(array):
    return np.apply_along_axis(sequence_pack, 1, array)


def sequence_unpack(sequence, length=147):
    sequence = np.asarray(sequence)
    assert len(sequence.shape) == 1
    unpacked = np.unpackbits(sequence)
    reshaped = unpacked.reshape((-1, 2))
    # with_zeros = np.zeros_like(reshaped.shape[0], 8)
    # with_zeros[:,6:] = reshaped
    with_zeros = np.pad(reshaped, pad_width=([0, 0], [6, 0]), mode='constant', constant_values=0)
    return np.packbits(with_zeros.flatten())[:length]


def unpack_sequences(array, length=147):
    return np.apply_along_axis(sequence_unpack, 1, array, length=length)


def _sequence_load(filename, length=147):
    file = np.load(filename)
    if np.max(file) == 3:
        return file
    else:
        return unpack_sequences(file, length)


def get_sequences():
    paths = get_file_paths()
    blobs = get_blob_paths(paths, 'sequences')
    blobs.sort()
    files = sorted(data_path.glob(blobs[-1]), key=lambda i: int(str(i).split('.')[-2].split('_')[-1]))

    # open
    # look for length in callfile
    length = 147
    #
    loadfunc = lambda x: _sequence_load(x, length=length)

    chunks = [dask.delayed(loadfunc)(i) for i in files]
    chunks = [da.from_delayed(i, i.shape.compute(), i.dtype.compute()) for i in chunks]

    a = da.concatenate(chunks, axis=0)

    return a


def get_energies():
    paths = get_file_paths()
    blobs = get_blob_paths(paths, 'energies')
    blobs.sort()
    files = sorted(data_path.glob(blobs[-1]), key=lambda i: int(str(i).split('.')[-2].split('_')[-1]))

    chunks = [dask.delayed(np.load)(i) for i in files]
    chunks = [da.from_delayed(i, i.shape.compute(), i.dtype.compute()) for i in chunks]

    a = da.concatenate(chunks, axis=0)

    return a


@nb.njit()
def _compress(array):
    binary_list = []
    shape = array.shape

    output = np.uint8(0)
    for i in range(shape[0]):
        output *= 0
        for j in range(shape[1]):
            output += int(array[i, j] << 2 * (3 - j % 4))
            if (j + 1) % 4 == 0 or j >= shape[1] - 1:
                binary_list.append(output)
                output *= 0
    return binary_list


def sequence_to_binary(array):
    """
    Makes a binary repr of the sequences stored in array.
    The first two bytes are the shape of the uncompressed data. The

    :param array:
    :return:
    """
    array = np.asarray(array)
    shape = array.shape
    binary_list = []

    binary_list.append(np.asarray(shape[0], dtype=np.uint16).tobytes())
    binary_list.append(np.asarray(shape[1], dtype=np.uint16).tobytes())

    compressed = _compress(array)
    binary_list.extend([i.to_bytes(1, 'little') for i in compressed])

    return binary_list


@nb.njit()
def _decompress_binary_list(binary_list, shape):
    output = np.empty(shape, dtype=np.uint8)

    mask3 = 3
    mask2 = mask3 << 2
    mask1 = mask2 << 2
    mask0 = mask1 << 2
    mask = [mask0, mask1, mask2, mask3]

    major_index = 0

    for i in range(2, len(binary_list)):
        # index = (i-2)*4
        for sub_index in np.arange(4):
            x = (major_index) // int(4 * np.ceil(shape[1] / 4))
            y = major_index % int(4 * np.ceil(shape[1] / 4))

            if y >= shape[1]:
                major_index += 1
                continue
            if x >= shape[0]:
                continue
            bit = (mask[sub_index] & binary_list[i]) >> (6 - 2 * sub_index)
            output[x, y] = bit
            major_index += 1
    return output


@nb.njit()
def _decompress_binary(binary, shape=(1000, 147)):
    output = np.empty(shape, dtype=np.uint8)

    mask3 = 3
    mask2 = mask3 << 2
    mask1 = mask2 << 2
    mask0 = mask1 << 2
    mask = [mask0, mask1, mask2, mask3]

    major_index = 0

    for byte in binary:

        for sub_index in np.arange(4):
            x = major_index // int(4 * np.ceil(shape[1] / 4))
            y = major_index % int(4 * np.ceil(shape[1] / 4))

            if y >= shape[1]:
                major_index += 1
                continue
            if x >= shape[0]:
                continue
            bit = (mask[sub_index] & byte) >> (6 - 2 * sub_index)
            output[x, y] = bit
            major_index += 1
    return output


def binary_to_sequence(binary_list):
    # shape = np.frombuffer(binary_list[0], dtype=np.uint16), np.frombuffer(binary_list[1], dtype=np.uint16)
    shape = int.from_bytes(binary_list[0], 'little'), int.from_bytes(binary_list[1], 'little')
    output = _decompress_binary_list(np.frombuffer(b''.join(binary_list[1:]), dtype=np.uint8),
                                     shape)  # [int(shape[0]), int(shape[1])])
    return output


def read_header(file):
    x = int.from_bytes(file.read(2), 'little')
    y = int.from_bytes(file.read(2), 'little')
    return x, y


def read_sequences(file):
    return file.read()


def save_sequences(filename, sequences):
    b = sequence_to_binary(sequences)
    # sequences_new = binary_to_sequence(b)

    with open(filename, 'wb') as f:
        f.writelines(b)


def load_sequences(filename):
    with open(filename, 'rb') as f:
        shape = read_header(f)
        return _decompress_binary(f.read(), shape)


def delayed_load(paths, loadfunc=load_sequences):
    delayed_loaded_files = [dask.delayed(loadfunc)(path) for path in paths]
    # delayed_pairs = [dask.delayed(to_pairs)(sequences) for sequences in delayed_loaded_files]
    # delayed_hists = [dask.delayed(histogram)(pairs) for pairs in delayed_pairs]
    return delayed_loaded_files


if __name__ == '__main__':
    s = np.random.choice(np.arange(4, dtype=np.uint8), 147)
    o = sequence_pack(s)
    s_new = sequence_unpack(o)

    assert np.allclose(s, s_new)

    sequences = np.random.choice(np.arange(4, dtype=np.uint8), (1000, 147))
    packed = pack_sequences(sequences)
    sequences_new = unpack_sequences(packed)

    assert np.allclose(sequences, sequences_new)

    d = DataSet(index=-1)
    print(d.datapaths)

    # s = np.random.choice(np.arange(4, dtype=np.uint8), (2, 10))
    # b = sequence_to_binary(sequences)
    # # sequences_new = binary_to_sequence(b)
    #
    # with open('test.seq', 'wb') as f:
    #     f.writelines(b)
    save_sequences('test.seq', sequences)

    sequences = load_sequences('test.seq')

    assert np.allclose(sequences, sequences_new)

    # import dask.array as da
    # import dask
    #
    # paths = get_file_paths()
    # blobs = get_blob_paths(paths, 'sequences')
    # blobs.sort()
    #
    # files = sorted(data_path.glob(blobs[-1]), key=lambda i: int(str(i).split('.')[-2].split('_')[-1]))
    #
    #
    # # open
    #
    # chunks = [dask.delayed(np.load)(i) for i in files[:]]
    # chunks = [da.from_delayed(i, i.shape.compute(), i.dtype.compute()) for i in chunks]
    #
    # a = da.concatenate(chunks, axis=0)

    callfiles = get_callfiles(None)
    data = [DataSet(c) for c in callfiles]
    sizes = [(d.callfile, len(d)) for d in data]
    print(sizes)
