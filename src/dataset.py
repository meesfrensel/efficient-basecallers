"""
Classes for handling data.

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/5db4957496079d19deacb01c9f4f4957f7257f49/src/classes.py
"""

import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, Sampler
import uuid

from constants import S2S_PAD
from normalization import med_mad, normalize_signal_from_read_data
from read import read_fast5
from utils import read_metadata

class BaseNanoporeDataset(Dataset):
    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        token_pad (int): value used for padding all the sequences
    """

    def __init__(self, data_dir, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None, token_pad = S2S_PAD):
        super(BaseNanoporeDataset, self).__init__()
        
        self.data_dir = data_dir
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        self.files_list = self._find_files()
        self.num_samples_per_file = self._get_samples_per_file()
        self.total_num_samples = np.sum(np.array(self.num_samples_per_file))
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self._split_train_validation()
        self._get_samplers()
        
        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

        self.token_pad = token_pad
    
    def __len__(self):
        """Number of samples
        """
        return self.total_num_samples
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.
        
        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))

    def _find_files(self):
        """Finds list of files to read
        """
        l = list()
        for r, d, f in os.walk(self.data_dir):
            for file in f:
                if file.endswith('.npz'):
                    rel_dir = os.path.relpath(r, self.data_dir)
                    l.append(os.path.join(rel_dir, file))
        l = sorted(l)
        print("Found " + str(len(l)) + " files")
        return l
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        l = list()
        for f in self.files_list:
            metadata = read_metadata(os.path.join(self.data_dir, f))
            l.append(metadata[0][1][0]) # [array_num, shape, first elem shape]
        return l
    
    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        num_validation_files = len(self.files_list) - num_train_files
        
        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)
            
        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])
        
        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))
        
        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))
                
        return None
    
    def _get_samplers(self):
        """Add samplers
        """
        self.train_sampler = IdxSampler(self.train_idxs, data_source = self)
        self.validation_sampler = IdxSampler(self.validation_idxs, data_source = self)
        return None
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        x = arr['x']
        y = arr['y']
        return self.process({'x':x, 'y':y})
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx]}
    
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        if y.dtype != 'U1':
            y = y.astype('U1')
        y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y
    
    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        y[y == str(self.token_pad)] = ''
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences


class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """
    def __init__(self, idxs, *args, **kwargs):
        super(IdxSampler, self).__init__(*args, **kwargs)
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)


class BaseFast5Dataset(Dataset):
    """Base dataset class that iterates over fast5 files for basecalling

    Attributes:
        data_dir (str): dir where the fast5 files are
        fast5_list (str): file with a list of files to be processed
        recursive (bool): if the data_dir should be searched recursively
        buffer_size (int): number of fast5 files to read 
    """

    def __init__(self, 
        data_dir = None, 
        fast5_list = None, 
        recursive = True, 
        buffer_size = 100,
        window_size = 2000,
        window_overlap = 400,
        trim_signal = True,
        ):
        """
        Args:
            data_dir (str): dir where the fast5 files are
            fast5_list (str): file with a list of files to be processed
            recursive (bool): if the data_dir should be searched recursively
            buffer_size (int): number of fast5 files to read 

        data_dir and fast5_list are exclusive
        """
        
        super(BaseFast5Dataset, self).__init__()
    
        self.data_dir = data_dir
        self.recursive = recursive
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.trim_signal = trim_signal

        if fast5_list is None:
            self.data_files = self.find_all_fast5_files()
        else:
            self.data_files = self.read_fast5_list(fast5_list)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        return self.process_reads(self.data_files[idx])

        
    def find_all_fast5_files(self):
        """Find all fast5 files in a dir recursively
        """
        # find all the files that we have to process
        files_list = list()
        for path in Path(self.data_dir).rglob('*.fast5'):
            files_list.append(str(path))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def read_fast5_list(self, fast5_list):
        """Read a text file with the reads to be processed
        """

        if isinstance(fast5_list, list):
            return self.buffer_list(fast5_list, self.buffer_size)

        files_list = list()
        with open(fast5_list, 'r') as f:
            for line in f:
                files_list.append(line.strip('\n'))
        files_list = self.buffer_list(files_list, self.buffer_size)
        return files_list

    def buffer_list(self, files_list, buffer_size):
        buffered_list = list()
        for i in range(0, len(files_list), buffer_size):
            buffered_list.append(files_list[i:i+buffer_size])
        return buffered_list

    def trim(self, signal, window_size=40, threshold_factor=2.4, min_elements=3):
        """

        from: https://github.com/nanoporetech/bonito/blob/master/bonito/fast5.py
        """

        min_trim = 10
        signal = signal[min_trim:]

        med, mad = med_mad(signal[-(window_size*100):])

        threshold = med + mad * threshold_factor
        num_windows = len(signal) // window_size

        seen_peak = False

        for pos in range(num_windows):
            start = pos * window_size
            end = start + window_size
            window = signal[start:end]
            if len(window[window > threshold]) > min_elements or seen_peak:
                seen_peak = True
                if window[-1] > threshold:
                    continue
                return min(end + min_trim, len(signal)), len(signal)

        return min_trim, len(signal)

    def chunk(self, signal, chunksize, overlap):
        """
        Convert a read into overlapping chunks before calling

        The first N datapoints will be cut out so that the window ends perfectly
        with the number of datapoints of the read.
        """
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)

        T = signal.shape[0]
        if chunksize == 0:
            chunks = signal[None, :]
        elif T < chunksize:
            chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
        else:
            stub = (T - overlap) % (chunksize - overlap)
            chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
        
        return chunks.unsqueeze(1)
    
    def normalize(self, read_data):
        return normalize_signal_from_read_data(read_data)

    def process_reads(self, read_list):
        """
        Args:
            read_list (list): list of files to be processed

        Returns:
            two arrays, the first one with the normalzized chunked data,
            the second one with the read ids of each chunk.
        """
        chunks_list = list()
        id_list = list()
        l_list = list()
        file_list = list()

        for read_file in read_list:
            reads_data = read_fast5(read_file)

            for read_id in reads_data.keys():
                read_data = reads_data[read_id]
                norm_signal = self.normalize(read_data)

                if self.trim_signal:
                    trim, _ = self.trim(norm_signal[:8000])
                    norm_signal = norm_signal[trim:]

                chunks = self.chunk(norm_signal, self.window_size, self.window_overlap)
                num_chunks = chunks.shape[0]

                uuid_fields = uuid.UUID(read_id).fields
                id_arr = np.zeros((num_chunks, 6), dtype = np.int64)
                for i, uf in enumerate(uuid_fields):
                    id_arr[:, i] = uf

                id_list.append(id_arr)
                l_list.append(np.full((num_chunks,), len(norm_signal)))
                chunks_list.append(chunks)
                file_list.append(list(read_file.encode()))

        out = {
            'x': torch.vstack(chunks_list).squeeze(1), 
            'id': np.vstack(id_list),
            'len': np.concatenate(l_list),
            'file': np.vstack(file_list)
        }
        return out
