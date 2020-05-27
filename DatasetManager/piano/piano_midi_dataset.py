import copy
import glob
import itertools
import math
import os
import pickle
import random
import re
import shutil
import time

import numpy as np
import pretty_midi
import torch
from torch.utils import data
from tqdm import tqdm

from DatasetManager.piano.piano_helper import extract_cc, find_nearest_value, MaestroIteratorGenerator

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""

START_SYMBOL = 'START'
END_SYMBOL = 'END'
PAD_SYMBOL = 'XX'


class PianoMidiDataset(data.Dataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 sequence_size,
                 smallest_time_shift,
                 max_transposition,
                 time_dilation_factor,
                 velocity_shift,
                 transformations
                 ):
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
        self.list_ids = {
            'train': [],
            'validation': [],
            'test': []
        }

        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        self.hop_size = min(sequence_size // 4, 10)

        #  features
        self.smallest_time_shift = smallest_time_shift
        self.time_table = self.get_time_table()
        self.pitch_range = range(21, 109)
        self.velocity_range = range(128)
        self.programs = range(128)

        # Index 2 value
        self.index2value = {}
        self.value2index = {}
        self.default_value = {
            'pitch': 60,
            'duration': 0.2,
            'time_shift': 0.1,
            'velocity': 80
        }
        self.silence_value = {
            'pitch': 60,
            'duration': 0.5,
            'time_shift': 0.5,
            'velocity': 0
        }

        #  Building/loading the dataset
        if os.path.isfile(self.dataset_file):
            self.load()
        else:
            print(f'Building dataset {str(self)}')
            self.make_tensor_dataset()

        #  data augmentations have to be initialised after loading
        self.max_transposition = max_transposition
        self.time_dilation_factor = time_dilation_factor
        self.velocity_shift = velocity_shift
        self.transformations = transformations
        return

    def __str__(self):
        prefix = str(self.corpus_it_gen)
        name = f'PianoMidi-' \
               f'{prefix}-' \
               f'{self.sequence_size}_' \
               f'{self.smallest_time_shift}'
        return name

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids[self.split])

    def get_time_table(self):
        short_time_shifts = np.arange(0, 1.0, self.smallest_time_shift)
        medium_time_shifts = np.arange(1.0, 5.0, 5.0 * self.smallest_time_shift)
        long_time_shifts = np.arange(5.0, 20., 50 * self.smallest_time_shift)
        time_shift_bins = np.concatenate((short_time_shifts,
                                          medium_time_shifts,
                                          long_time_shifts))
        return time_shift_bins

    @property
    def data_folder_name(self):
        # Same as __str__ but without the sequence_len
        name = f'PianoMidi-{self.corpus_it_gen}'
        return name

    @property
    def cache_dir(self):
        cache_dir = f'{os.path.expanduser("~")}/Data/dataset_cache/PianoMidi'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    @property
    def dataset_file(self):
        dataset_dir = f'{self.cache_dir}/{str(self)}'
        return dataset_dir

    def save(self):
        f = open(self.dataset_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self):
        """
        Load a dataset while avoiding local parameters specific to the machine used
        :return:
        """
        f = open(self.dataset_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        for k, v in tmp_dict.items():
            if k != 'local_parameters':
                self.__dict__[k] = v

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        """ttt = time.time()"""
        id = self.list_ids[self.split][index]
        """ttt = time.time() - ttt
        print(f'Get indices: {ttt}')
        ttt = time.time()"""

        ################################################################################################
        ################################################################################################
        ################################################################################################
        # Load data and extract subsequence
        with open(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/length.txt') as ff:
            sequence_length = int(ff.read())
        start_time = id['start_time']
        end_time = min(id['start_time'] + self.sequence_size, sequence_length)
        fpr_pitch = np.memmap(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/pitch',
                              dtype=int, mode='r', shape=(sequence_length))
        pitch = fpr_pitch[start_time:end_time]
        del fpr_pitch
        fpr_velocity = np.memmap(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/velocity',
                              dtype=int, mode='r', shape=(sequence_length))
        velocity = fpr_velocity[start_time:end_time]
        del fpr_velocity
        fpr_duration = np.memmap(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/duration',
                              dtype='float32', mode='r', shape=(sequence_length))
        duration = fpr_duration[start_time:end_time]
        del fpr_duration
        fpr_time_shift = np.memmap(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/time_shift',
                              dtype='float32', mode='r', shape=(sequence_length))
        time_shift = fpr_time_shift[start_time:end_time]
        del fpr_time_shift
        """ttt = time.time() - ttt
        print(f'Loading text files: {ttt}')
        ttt = time.time()"""
        ################################################################################################
        ################################################################################################
        ################################################################################################

        # pitch = np.loadtxt(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/pitch.txt',
        #                    dtype=int)
        # velocity = np.loadtxt(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/velocity.txt',
        #                       dtype=int)
        # duration = np.loadtxt(f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/duration.txt',
        #                       dtype=np.float32)
        # time_shift = np.loadtxt(
        #     f'{self.cache_dir}/{self.data_folder_name}/{self.split}/{id["score_name"]}/time_shift.txt',
        #     dtype=np.float32)
        # ttt = time.time() - ttt
        # print(f'Loading text files: {ttt}')
        # ttt = time.time()
        #
        # start_time = id['start_time']
        # end_time = min(id['start_time'] + self.sequence_size, len(pitch))
        # pitch = pitch[start_time:end_time]
        # velocity = velocity[start_time:end_time]
        # duration = duration[start_time:end_time]
        # time_shift = time_shift[start_time:end_time]
        #
        # ttt = time.time() - ttt
        # print(f'Chunking: {ttt}')
        # ttt = time.time()

        # Perform data augmentations (only for train split)
        if (self.transformations['velocity_shift']) and (self.split == 'train'):
            velocity_shift = int(self.velocity_shift * (2 * random.random() - 1))
            velocity = np.maximum(0, np.minimum(127, velocity + velocity_shift))
        else:
            velocity_shift = 0
        if (self.transformations['time_dilation']) and (self.split == 'train'):
            time_dilation_factor = 1 - self.time_dilation_factor + 2 * self.time_dilation_factor * random.random()
            duration = duration * time_dilation_factor
            time_shift = time_shift * time_dilation_factor
        else:
            time_dilation_factor = 1
        if (self.transformations['transposition']) and (self.split == 'train'):
            transposition = int(random.uniform(-self.max_transposition, self.max_transposition))
            pitch = pitch + transposition
            pitch = np.where(pitch > self.pitch_range.stop - 1, pitch - 12,
                             pitch)  # lower one octave for pitch too high
            pitch = np.where(pitch < self.pitch_range.start, pitch + 12, pitch)  # raise one octave for pitch too low
        else:
            transposition = 0
        """ttt = time.time() - ttt
        print(f'Data augmentation: {ttt}')
        ttt = time.time()"""

        # Add pad, start and end symbols
        pitch = list(pitch)
        velocity = list(velocity)
        duration = list(duration)
        time_shift = list(time_shift)
        if start_time == 0:
            pitch = [START_SYMBOL] + pitch[:-1]
            velocity = [START_SYMBOL] + velocity[:-1]
            duration = [START_SYMBOL] + duration[:-1]
            time_shift = [START_SYMBOL] + time_shift[:-1]
        end_padding_length = self.sequence_size - len(pitch)
        if end_padding_length > 0:
            pitch += [END_SYMBOL] + [PAD_SYMBOL] * (end_padding_length - 1)
            velocity += [END_SYMBOL] + [PAD_SYMBOL] * (end_padding_length - 1)
            duration += [END_SYMBOL] + [PAD_SYMBOL] * (end_padding_length - 1)
            time_shift += [END_SYMBOL] + [PAD_SYMBOL] * (end_padding_length - 1)

        """ttt = time.time() - ttt
        print(f'Adding meta symbols: {ttt}')
        ttt = time.time()"""

        # Tokenize
        pitch = [self.value2index['pitch'][e] for e in pitch]
        velocity = [self.value2index['velocity'][e] for e in velocity]
        duration = [self.value2index['duration'][find_nearest_value(self.time_table, e)]
                    if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL] else self.value2index['duration'][e]
                    for e in duration]
        time_shift = [self.value2index['time_shift'][find_nearest_value(self.time_table, e)]
                      if e not in [PAD_SYMBOL, END_SYMBOL, START_SYMBOL] else self.value2index['time_shift'][e]
                      for e in time_shift]

        """ttt = time.time() - ttt
        print(f'Tokenizing: {ttt}')

        print(f'###################################')"""

        return {'pitch': torch.tensor(pitch).long(),
                'velocity': torch.tensor(velocity).long(),
                'duration': torch.tensor(duration).long(),
                'time_shift': torch.tensor(time_shift).long(),
                'index': index,
                'data_augmentations': {
                    'time_dilation': time_dilation_factor,
                    'velocity_shift': velocity_shift,
                    'transposition': transposition
                }
                }

    def iterator_gen(self):
        return (elem for elem in self.corpus_it_gen())

    def split_datasets(self, split=None, indexed_datasets=None):
        train_dataset = copy.copy(self)
        train_dataset.split = 'train'
        val_dataset = copy.copy(self)
        val_dataset.split = 'validation'
        test_dataset = copy.copy(self)
        test_dataset.split = 'test'
        return {'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
                }

    def data_loaders(self, batch_size, num_workers,
                     shuffle_train=True, shuffle_val=False):
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
        return {'train': train_dl,
                'val': val_dl,
                'test': test_dl}

    def compute_index_dicts(self):
        ######################################################################
        #  Index 2 value
        for feat_name in ['pitch', 'duration', 'velocity', 'time_shift']:
            index2value = {}
            value2index = {}
            index = 0

            if feat_name == 'time_shift':
                values = self.time_table
            elif feat_name == 'duration':
                values = self.time_table[1:]
            elif feat_name == 'pitch':
                values = self.pitch_range
            elif feat_name == 'velocity':
                values = self.velocity_range
            else:
                raise Exception

            for value in values:
                index2value[index] = value
                value2index[value] = index
                index2value[index] = value
                value2index[value] = index
                index += 1
            # Pad
            index2value[index] = PAD_SYMBOL
            value2index[PAD_SYMBOL] = index
            index += 1
            # Start
            index2value[index] = START_SYMBOL
            value2index[START_SYMBOL] = index
            index += 1
            # End
            index2value[index] = END_SYMBOL
            value2index[END_SYMBOL] = index
            index += 1

            self.index2value[feat_name] = index2value
            self.value2index[feat_name] = value2index
        return

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Loading index dictionnary')

        self.compute_index_dicts()

        print('Making tensor dataset')

        chunk_counter = {
            'train': 0,
            'validation': 0,
            'test': 0,
        }

        # Build x folder if not existing
        if not os.path.isfile(f'{self.cache_dir}/{self.data_folder_name}/xbuilt'):
            if os.path.isdir(f'{self.cache_dir}/{self.data_folder_name}'):
                shutil.rmtree(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/train')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/validation')
            os.mkdir(f'{self.cache_dir}/{self.data_folder_name}/test')
            # Iterate over files
            for midi_file, split in tqdm(self.iterator_gen()):
                # midi to sequence
                sequences = self.process_score(midi_file)
                midi_name = os.path.splitext(re.split('/', midi_file)[-1])[0]
                folder_name = f'{self.cache_dir}/{self.data_folder_name}/{split}/{midi_name}'
                os.mkdir(folder_name)
                # np.savetxt(f'{folder_name}/pitch.txt', np.asarray(sequences['pitch']).astype(int), fmt='%d')
                # np.savetxt(f'{folder_name}/velocity.txt', np.asarray(sequences['velocity']).astype(int), fmt='%d')
                # np.savetxt(f'{folder_name}/duration.txt', np.asarray(sequences['duration']).astype(np.float32),
                #            fmt='%.3f')
                # np.savetxt(f'{folder_name}/time_shift.txt', np.asarray(sequences['time_shift']).astype(np.float32),
                #            fmt='%.3f')

                # test mmap
                sequence_length = len(sequences['pitch'])
                with open(f'{folder_name}/length.txt', 'w') as ff:
                    ff.write(f'{sequence_length:d}')
                fp_pitch = np.memmap(f'{folder_name}/pitch', dtype=int, mode='w+', shape=(sequence_length))
                fp_pitch[:] = np.asarray(sequences['pitch']).astype(int)
                del fp_pitch
                fp_velocity = np.memmap(f'{folder_name}/velocity', dtype=int, mode='w+', shape=(sequence_length))
                fp_velocity[:] = np.asarray(sequences['velocity']).astype(int)
                del fp_velocity
                fp_duration = np.memmap(f'{folder_name}/duration', dtype='float32', mode='w+', shape=(sequence_length))
                fp_duration[:] = np.asarray(sequences['duration']).astype('float32')
                del fp_duration
                fp_time_shift = np.memmap(f'{folder_name}/time_shift', dtype='float32', mode='w+',
                                          shape=(sequence_length))
                fp_time_shift[:] = np.asarray(sequences['time_shift']).astype('float32')
                del fp_time_shift
            open(f'{self.cache_dir}/{self.data_folder_name}/xbuilt', 'w').close()

        # Build index of files
        for split in ['train', 'validation', 'test']:
            paths = glob.glob(f'{self.cache_dir}/{self.data_folder_name}/{split}/*')
            for path in paths:
                # read file
                with open(f'{path}/length.txt', 'r') as ff:
                    sequence_length = int(ff.read())
                score_name = path.split('/')[-1]
                # split in chunks
                for start_time in range(0, sequence_length, self.hop_size):
                    chunk_counter[split] += 1
                    self.list_ids[split].append({
                        'score_name': score_name,
                        'start_time': start_time,
                    })

        print(f'Chunks: {chunk_counter}\n')

        # Save class (actually only serve for self.list_ids, helps with reproducibility)
        self.save()
        return

    def process_score(self, midi_file):
        #  Preprocess midi
        midi = pretty_midi.PrettyMIDI(midi_file)
        raw_sequence = list(itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in self.programs and not inst.is_drum]))
        control_changes = list(itertools.chain(*[
            inst.control_changes for inst in midi.instruments
            if inst.program in self.programs and not inst.is_drum]))
        # sort by starting time
        raw_sequence.sort(key=lambda x: x.start)
        control_changes.sort(key=lambda x: x.time)

        #  pedal, cc = 64
        sustain_pedal_time, sustain_pedal_value = extract_cc(control_changes=control_changes,
                                                             channel=64,
                                                             binarize=True)

        # sostenuto pedal, cc = 66
        sostenuto_pedal_time, sostenuto_pedal_value = extract_cc(control_changes=control_changes,
                                                                 channel=66,
                                                                 binarize=True)

        # soft pedal, cc = 67
        soft_pedal_time, soft_pedal_value = extract_cc(control_changes=control_changes,
                                                       channel=67,
                                                       binarize=True)

        seq_len = len(raw_sequence)

        pitch_sequence = []
        velocity_sequence = []
        duration_sequence = []
        time_shift_sequence = []
        for event_ind in range(seq_len):
            # Get values
            event = raw_sequence[event_ind]
            event_values = {}

            # Compute duration taking sustain
            sustained_index_start = np.searchsorted(sustain_pedal_time, event.start, side='left') - 1
            if sustain_pedal_value[sustained_index_start] == 1:
                if (sustained_index_start + 1) >= len(sustain_pedal_time):
                    event_end_sustained = 0
                else:
                    event_end_sustained = sustain_pedal_time[sustained_index_start + 1]
                event_end = max(event.end, event_end_sustained)
            else:
                event_end = event.end

            #  also check if pedal is pushed before the end of the note !!
            sustained_index_end = np.searchsorted(sustain_pedal_time, event.end, side='left') - 1
            if sustain_pedal_value[sustained_index_end] == 1:
                if (sustained_index_end + 1) >= len(sustain_pedal_time):
                    # notes: that's a problem, means a sustain pedal is not switched off....
                    event_end_sustained = 0
                else:
                    event_end_sustained = sustain_pedal_time[sustained_index_end + 1]
                event_end = max(event.end, event_end_sustained)

            duration_value = find_nearest_value(self.time_table[1:], event_end - event.start)

            event_values['duration'] = duration_value
            if event.pitch in self.pitch_range:
                event_values['pitch'] = event.pitch
            else:
                continue
            event_values['velocity'] = event.velocity
            if event_ind != seq_len - 1:
                next_event = raw_sequence[event_ind + 1]
                event_values['time_shift'] = find_nearest_value(self.time_table, next_event.start - event.start)
            else:
                event_values['time_shift'] = 0

            #  Convert to str
            pitch_sequence.append(event_values['pitch'])
            velocity_sequence.append(event_values['velocity'])
            duration_sequence.append(event_values['duration'])
            time_shift_sequence.append(event_values['time_shift'])
        return {
            'pitch': pitch_sequence,
            'velocity': velocity_sequence,
            'duration': duration_sequence,
            'time_shift': time_shift_sequence
        }

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        raise NotImplementedError

    def interleave_silences_batch(self, sequences, index_order):
        ret = []
        silence_frame = torch.tensor(
            [self.value2index[feat_name][self.silence_value[feat_name]] for feat_name in index_order])
        for e in sequences:
            ret.extend(e)
            ret.append(silence_frame)
            ret.append(silence_frame)
            ret.append(silence_frame)
        ret_stack = torch.stack(ret, dim=0)
        return ret_stack

    def fill_missing_features(self, sequence, selected_features_indices):
        # Fill in missing features with default values
        default_frame = [self.value2index[feat_name][self.default_value[feat_name]] for feat_name in self.index_order]
        sequence_filled = torch.tensor([default_frame] * len(sequence))
        sequence_filled[:, selected_features_indices] = sequence
        return sequence_filled

    def tensor_to_score(self, sequences, fill_features):
        # Create score
        score = pretty_midi.PrettyMIDI()
        # 'Acoustic Grand Piano', 'Bright Acoustic Piano',
        #                   'Electric Grand Piano', 'Honky-tonk Piano',
        #                   'Electric Piano 1', 'Electric Piano 2', 'Harpsichord',
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # Fill in missing features with default values
        a_key = list(sequences.keys())[0]
        sequence_length = len(sequences[a_key])
        if fill_features is not None:
            for feature in fill_features:
                sequences[feature] = [self.default_value[feature]] * sequence_length

        start_time = 0.0
        for t in range(sequence_length):
            pitch_ind = int(sequences['pitch'][t])
            duration_ind = int(sequences['duration'][t])
            velocity_ind = int(sequences['velocity'][t])
            time_shift_ind = int(sequences['time_shift'][t])

            pitch_value = self.index2value['pitch'][pitch_ind]
            duration_value = self.index2value['duration'][duration_ind]
            velocity_value = self.index2value['velocity'][velocity_ind]
            time_shift_value = self.index2value['time_shift'][time_shift_ind]

            if pitch_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL] or \
                    duration_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL] or \
                    velocity_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL] or \
                    time_shift_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                continue

            note = pretty_midi.Note(
                velocity=velocity_value, pitch=pitch_value, start=start_time, end=start_time + duration_value)

            piano.notes.append(note)

            start_time += time_shift_value

        score.instruments.append(piano)
        return score

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
    corpus_it_gen = MaestroIteratorGenerator(
        composers_filter=[],
        num_elements=None
    )
    sequence_size = 120
    smallest_time_shift = 0.02
    max_transposition = 6
    time_dilation_factor = 0.1
    velocity_shift = 10
    transformations = {
        'time_shift': True,
        'time_dilation': True,
        'velocity_shift': True,
        'transposition': True,
    }
    dataset = PianoMidiDataset(corpus_it_gen,
                               sequence_size,
                               smallest_time_shift,
                               max_transposition,
                               time_dilation_factor,
                               velocity_shift,
                               transformations)

    dataloaders = dataset.data_loaders(
        batch_size=32,
        num_workers=0,
        shuffle_train=True,
        shuffle_val=True
    )

    for x in dataloaders['train']:
        # Write back to midi
        print('yoyo')
