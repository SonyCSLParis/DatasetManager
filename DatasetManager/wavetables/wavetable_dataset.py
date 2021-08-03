import copy
import glob
import os
import pickle
import random
import re
import shutil
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchaudio

from DatasetManager.wavetables.wavetable_helper import WavetableIteratorGenerator


class WavetableDataset(data.Dataset):
    """
    Wavetable Dataset (Serum)
    Wavetables are sequences of single-cycle waveforms of length 2058 samples.
    """
    def __init__(self, iterator_gen, num_frames, transformations):
        """
        All transformations
        {
            'time_shift': True,
            'time_dilation': True,
            'transposition': True
        }

        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()
        self.split = None
        self.list_ids = {'train': [], 'validation': [], 'test': []}
        self.iterator_gen = iterator_gen
        self.num_frames = num_frames
        self.samples_per_frame = 2048
        self.transformations = transformations

        #  Building/loading the dataset
        if os.path.isfile(self.dataset_file):
            self.load()
        else:
            print(f'Building dataset {str(self)}')
            self.make_tensor_dataset()

    def __str__(self):
        prefix = str(self.iterator_gen)
        name = f'Wavetable-' \
               f'{prefix}-' \
               f'{self.num_frames}'
        return name

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids[self.split])

    @property
    def data_folder_name(self):
        # Same as __str__ but without the sequence_len
        name = f'Wavetable-{self.iterator_gen}'
        return name

    @property
    def cache_dir(self):
        cache_dir = f'{os.path.expanduser("~")}/Data/dataset_cache/Wavetable'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    @property
    def dataset_file(self):
        dataset_dir = f'{self.cache_dir}/{str(self)}'
        return dataset_dir

    def save(self):
        # Only save list_ids
        with open(self.dataset_file, 'wb') as ff:
            pickle.dump(self.list_ids, ff, 2)

    def load(self):
        """
        Load a dataset while avoiding local parameters specific to the machine used
        :return:
        """
        with open(self.dataset_file, 'rb') as ff:
            list_ids = pickle.load(ff)
        self.list_ids = list_ids

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        id = self.list_ids[self.split][index]

        # Load data and extract subsequence
        sequence = {}
        with open(
                f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/length.txt'
        ) as ff:
            sequence_length = int(ff.read())

        # start_time can be negative, used for padding
        start_time = id['start_time']
        sequence_start_time = max(start_time, 0)

        end_time = min(id['start_time'] + self.sequence_size, sequence_length)
        fpr_pitch = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/pitch',
            dtype=int,
            mode='r',
            shape=(sequence_length))
        sequence['pitch'] = fpr_pitch[sequence_start_time:end_time]
        del fpr_pitch
        fpr_velocity = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/velocity',
            dtype=int,
            mode='r',
            shape=(sequence_length))
        sequence['velocity'] = fpr_velocity[sequence_start_time:end_time]
        del fpr_velocity
        fpr_duration = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/duration',
            dtype='float32',
            mode='r',
            shape=(sequence_length))
        sequence['duration'] = fpr_duration[sequence_start_time:end_time]
        del fpr_duration
        fpr_time_shift = np.memmap(
            f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/time_shift',
            dtype='float32',
            mode='r',
            shape=(sequence_length))
        sequence['time_shift'] = fpr_time_shift[sequence_start_time:end_time]
        del fpr_time_shift
        """ttt = time.time() - ttt
        print(f'Loading text files: {ttt}')
        ttt = time.time()"""

        # Perform data augmentations (only for train split)
        if (self.transformations['velocity_shift']) and (self.split
                                                         == 'train'):
            velocity_shift = int(self.velocity_shift *
                                 (2 * random.random() - 1))
            sequence['velocity'] = np.maximum(
                0, np.minimum(127, sequence['velocity'] + velocity_shift))
        else:
            velocity_shift = 0
        if (self.transformations['time_dilation']) and (self.split == 'train'):
            time_dilation_factor = 1 - self.time_dilation_factor + 2 * self.time_dilation_factor * random.random(
            )
            sequence['duration'] = sequence['duration'] * time_dilation_factor
            sequence[
                'time_shift'] = sequence['time_shift'] * time_dilation_factor
        else:
            time_dilation_factor = 1
        if (self.transformations['transposition']) and (self.split == 'train'):
            transposition = int(
                random.uniform(-self.max_transposition,
                               self.max_transposition))
            sequence['pitch'] = sequence['pitch'] + transposition
            sequence['pitch'] = np.where(
                sequence['pitch'] > self.pitch_range.stop - 1,
                sequence['pitch'] - 12, sequence['pitch']
            )  # lower one octave for sequence['pitch'] too high
            sequence['pitch'] = np.where(
                sequence['pitch'] < self.pitch_range.start,
                sequence['pitch'] + 12,
                sequence['pitch'])  # raise one octave for pitch too low
        else:
            transposition = 0
        """ttt = time.time() - ttt
        print(f'Data augmentation: {ttt}')
        ttt = time.time()"""

        # Add pad, start and end symbols
        sequence = self.add_start_end_symbols(sequence,
                                              start_time=start_time,
                                              sequence_size=self.sequence_size)
        """ttt = time.time() - ttt
        print(f'Adding meta symbols: {ttt}')
        ttt = time.time()"""

        # Tokenize
        sequence = self.tokenize(sequence)
        """ttt = time.time() - ttt
        print(f'Tokenizing: {ttt}')

        print(f'###################################')"""

        return {
            'pitch': torch.tensor(sequence['pitch']).long(),
            'velocity': torch.tensor(sequence['velocity']).long(),
            'duration': torch.tensor(sequence['duration']).long(),
            'time_shift': torch.tensor(sequence['time_shift']).long(),
            'index': index,
            'data_augmentations': {
                'time_dilation': time_dilation_factor,
                'velocity_shift': velocity_shift,
                'transposition': transposition
            }
        }

    def tokenize(self, sequence):
        sequence['pitch'] = [
            self.value2index['pitch'][e] for e in sequence['pitch']
        ]
        sequence['velocity'] = [
            self.value2index['velocity'][e] for e in sequence['velocity']
        ]
        # legacy...
        # TODO use only one table?!
        # This if state is always True
        if hasattr(self, 'time_table_duration'):
            sequence['duration'] = [
                self.value2index['duration'][find_nearest_value(
                    self.time_table_duration, e)] if e not in [
                        PAD_SYMBOL, END_SYMBOL, START_SYMBOL
                    ] else self.value2index['duration'][e]
                for e in sequence['duration']
            ]
            sequence['time_shift'] = [
                self.value2index['time_shift'][find_nearest_value(
                    self.time_table_time_shift, e)] if e not in [
                        PAD_SYMBOL, END_SYMBOL, START_SYMBOL
                    ] else self.value2index['time_shift'][e]
                for e in sequence['time_shift']
            ]
        else:
            sequence['duration'] = [
                self.value2index['duration'][find_nearest_value(
                    self.time_table, e)] if e not in [
                        PAD_SYMBOL, END_SYMBOL, START_SYMBOL
                    ] else self.value2index['duration'][e]
                for e in sequence['duration']
            ]
            sequence['time_shift'] = [
                self.value2index['time_shift'][find_nearest_value(
                    self.time_table, e)] if e not in [
                        PAD_SYMBOL, END_SYMBOL, START_SYMBOL
                    ] else self.value2index['time_shift'][e]
                for e in sequence['time_shift']
            ]
        return sequence

    def iterator_gen(self):
        return (elem for elem in self.iterator_gen())

    def split_datasets(self, split=None, indexed_datasets=None):
        train_dataset = copy.copy(self)
        train_dataset.split = 'train'
        val_dataset = copy.copy(self)
        val_dataset.split = 'validation'
        test_dataset = copy.copy(self)
        test_dataset.split = 'test'
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def data_loaders(self,
                     batch_size,
                     num_workers,
                     shuffle_train=True,
                     shuffle_val=False):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param num_workers:
        :param shuffle_val:
        :param shuffle_train:
        :param batch_size:
        :param split:
        :return:
        """

        datasets = self.split_datasets()

        train_dl = data.DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_dl = data.DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return {'train': train_dl, 'val': val_dl, 'test': test_dl}

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        chunk_counter = {
            'train': 0,
            'validation': 0,
            'test': 0,
        }

        # Build x folder if not existing
        if not os.path.isfile(
                f'{self.cache_dir}/{self.data_folder_name}/xbuilt'):
            if os.path.isdir(f'{self.cache_dir}/{self.data_folder_name}'):
                shutil.rmtree(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/train')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/validation')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/test')
            # Iterate over files
            for wt_file, split in tqdm(self.iterator_gen()):
                # wav to wavetable
                wt = self.process_wave(wt_file)
                continue
                wt_name = f"{re.split('/', wt_file)[-2]}_{os.path.splitext(re.split('/', wt_file)[-1])[0]}"
                folder_name = f'{self.cache_dir}/{self.data_folder_name}/{split}/{wt_name}'
                if os.path.exists(folder_name):
                    print(f'Skipped {folder_name}')
                    continue
                os.mkdir(folder_name)

                # test mmap
                sequence_length = len(sequences['pitch'])
                with open(f'{folder_name}/length.txt', 'w') as ff:
                    ff.write(f'{sequence_length:d}')
                fp_pitch = np.memmap(f'{folder_name}/pitch',
                                     dtype=int,
                                     mode='w+',
                                     shape=(sequence_length))
                fp_pitch[:] = np.asarray(sequences['pitch']).astype(int)
                del fp_pitch
                fp_velocity = np.memmap(f'{folder_name}/velocity',
                                        dtype=int,
                                        mode='w+',
                                        shape=(sequence_length))
                fp_velocity[:] = np.asarray(sequences['velocity']).astype(int)
                del fp_velocity
                fp_duration = np.memmap(f'{folder_name}/duration',
                                        dtype='float32',
                                        mode='w+',
                                        shape=(sequence_length))
                fp_duration[:] = np.asarray(
                    sequences['duration']).astype('float32')
                del fp_duration
                fp_time_shift = np.memmap(f'{folder_name}/time_shift',
                                          dtype='float32',
                                          mode='w+',
                                          shape=(sequence_length))
                fp_time_shift[:] = np.asarray(
                    sequences['time_shift']).astype('float32')
                del fp_time_shift
            open(f'{self.cache_dir}/{self.data_folder_name}/xbuilt',
                 'w').close()

        # Build index of files
        for split in ['train', 'validation', 'test']:
            paths = glob.glob(
                f'{self.cache_dir}/{self.data_folder_name}/{split}/*')
            for path in paths:
                # read file
                with open(f'{path}/length.txt', 'r') as ff:
                    sequence_length = int(ff.read())
                score_name = path.split('/')[-1]

                # split in chunks
                # WARNING difference between self.sequence_size (size of the returned sequences) and sequence_length (actual size of the file)
                if self.pad_before:
                    start_at = -self.sequence_size + 1
                else:
                    start_at = -1
                for start_time in range(start_at, sequence_length,
                                        self.hop_size):
                    chunk_counter[split] += 1
                    self.list_ids[split].append({
                        'score_name': score_name,
                        'start_time': start_time,
                    })

        print(f'Chunks: {chunk_counter}\n')

        # Save class (actually only serve for self.list_ids, helps with reproducibility)
        self.save()
        return

    def process_wave(self, wt_file):
        wt, sample_rate = torchaudio.load(wt_file)
        self.visualize_wt(wt)
        assert sample_rate == 44100
        return wt

    def tensor_to_wavetable(self, sequences, fill_features):
        """
        Input: torch tensor
        Output: wavetable as a wave file
        """
        raise NotImplementedError

    def visualize_wt(self, wt):
        # Truncate
        wt_split = wt.view(self.samples_per_frame, -1)



    def visualise_batch(self, piano_sequences, writing_dir, filepath):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if len(piano_sequences.size()) == 1:
            piano_sequences = torch.unsqueeze(piano_sequences, dim=0)

        num_batches = len(piano_sequences)

        for batch_ind in range(num_batches):
            midipath = f"{writing_dir}/{filepath}_{batch_ind}.mid"
            score = self.tensor_to_score(sequence=piano_sequences[batch_ind],
                                         selected_features=None)
            score.write(midipath)


if __name__ == '__main__':
    iterator_gen = WavetableIteratorGenerator(num_elements=None)
    num_frames = 32
    transformations = {}
    dataset = WavetableDataset(iterator_gen=iterator_gen,
                               num_frames=num_frames,
                               transformations=transformations)

    dataloaders = dataset.data_loaders(batch_size=32,
                                       num_workers=0,
                                       shuffle_train=True,
                                       shuffle_val=True)

    for x in dataloaders['train']:
        # Write back to midi
        print('yoyo')
