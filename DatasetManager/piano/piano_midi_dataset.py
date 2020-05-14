import itertools
import math
import os
import pickle
import random
import shutil
import copy
import numpy as np
import pretty_midi
import torch
from torch.utils import data
from tqdm import tqdm

from DatasetManager.piano.piano_helper import extract_cc, find_nearest_value

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
        cache_dir = f'{os.path.expanduser("~")}/Data/dataset_cache'
        # create cache dir if it doesn't exist
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        self.split = None
        self.list_ids = {
            'train': [],
            'validation': [],
            'test': []
        }

        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        assert self.sequence_size // 4, 'sequence_size needs to be a multiple of 4'

        #  features
        self.smallest_time_shift = smallest_time_shift
        self.time_table = self.get_time_table()
        self.pitch_range = range(21, 109)
        self.velocity_range = range(128)
        self.programs = range(128)

        # Index 2 value
        self.index2value = {}
        self.value2index = {}
        self.index_order = ['pitch', 'duration', 'velocity', 'time_shift']
        self.index_order_dict = {v: k for k, v in enumerate(self.index_order)}
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

        # Chunking scores
        self.hop_size = None

        # Precomputed values (for convenience)
        self.padding_chunk = None
        self.start_chunk = None
        self.end_chunk = None
        self.zero_time_shift = None

        dataset_dir = f'{cache_dir}/{str(self)}'
        filename = f'{dataset_dir}/dataset.pkl'
        self.local_parameters = {
            'dataset_dir': dataset_dir,
            'filename': filename
        }

        #  Building/loading the dataset
        if os.path.isfile(filename):
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

    def save(self):
        f = open(self.local_parameters['filename'], 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self):
        """
        Load a dataset while avoiding local parameters specific to the machine used
        :return:
        """
        f = open(self.local_parameters['filename'], 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        for k, v in tmp_dict.items():
            if k != 'local_parameters':
                self.__dict__[k] = v

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        dataset_dir = self.local_parameters["dataset_dir"]
        # Select sample
        id = self.list_ids[self.split][index]

        # Load data and get label
        x = np.load(f'{dataset_dir}/x/{id}_{self.split}.npy')

        # Apply transformations, only on train dataset
        if self.split == 'train':
            x, time_dilation, velocity_dilation = self.transform(x)
        else:
            x = x[:self.sequence_size]
            time_dilation = 1
            velocity_dilation = 1
        x = torch.tensor(x)
        return x, index, time_dilation, velocity_dilation

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
        for feat_name in self.index_order:
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

        ######################################################################
        # Precomputed vectors
        self.padding_chunk = [self.value2index[feat_name][PAD_SYMBOL] for feat_name in self.index_order]
        self.start_chunk = [self.value2index[feat_name][START_SYMBOL] for feat_name in self.index_order]
        self.end_chunk = [self.value2index[feat_name][END_SYMBOL] for feat_name in self.index_order]
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
        dataset_dir = self.local_parameters["dataset_dir"]
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        os.mkdir(f'{dataset_dir}/x')

        # Iterate over files
        for score_id, (midi_file, split) in tqdm(enumerate(self.iterator_gen())):

            # midi to sequence
            sequence, _, _ = self.process_score(midi_file)

            # split in chunks
            for start_time in range(0, len(sequence) - self.sequence_size + 1, self.hop_size):
                end_time = start_time + self.sequence_size
                # Take extra size to dynamically shift the chunks to the right when loading them (see def transform)
                # but not for edges
                edge_chunk = (start_time == 0) or (end_time == len(sequence))
                if not edge_chunk:
                    end_time += self.hop_size
                chunk = sequence[start_time:end_time]
                np.save(f'{dataset_dir}/x/{chunk_counter[split]}_{split}', chunk)

                self.list_ids[split].append(chunk_counter[split])
                chunk_counter[split] += 1

        print(f'Chunks: {chunk_counter}\n')

        # Save class
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

        structured_sequence = []
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

            #  Convert to indices
            this_note = []
            for feat_name in self.index_order:
                this_note.append(self.value2index[feat_name][event_values[feat_name]])
            structured_sequence.append(this_note)

        #  Build sequence
        self.hop_size = self.sequence_size // 4
        prepend_length = self.hop_size
        # Ensure seq_length is a multiple of hop_size
        raw_seq_len = len(structured_sequence)
        seq_length = math.ceil((prepend_length + raw_seq_len) / self.hop_size) * self.hop_size
        #  Append length (does not count final ts0 - end - ts0, so -3)
        append_length = seq_length - (prepend_length + raw_seq_len)
        if append_length == 0:
            append_length = self.hop_size
            seq_length += self.hop_size

        sequence = []
        # prepend
        sequence += [self.padding_chunk] * (prepend_length - 1)
        sequence.append(self.start_chunk)
        # content
        sequence += structured_sequence
        # append
        sequence.append(self.end_chunk)
        sequence += [self.padding_chunk] * (append_length - 1)
        sequence = np.array(sequence)
        return sequence, prepend_length, append_length

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        raise NotImplementedError

    def interleave_silences_batch(self, sequences):
        ret = []
        silence_frame = torch.tensor(
            [self.value2index[feat_name][self.silence_value[feat_name]] for feat_name in self.index_order])
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

    def tensor_to_score(self, sequence, selected_features, fill_features_bool):
        # Create score
        score = pretty_midi.PrettyMIDI()
        # 'Acoustic Grand Piano', 'Bright Acoustic Piano',
        #                   'Electric Grand Piano', 'Honky-tonk Piano',
        #                   'Electric Piano 1', 'Electric Piano 2', 'Harpsichord',
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        # values
        sequence = sequence.numpy()

        if selected_features is None:
            selected_features = self.index_order
        selected_features_indices = [self.index_order_dict[feat_name] for feat_name in selected_features]

        # Fill in missing features with default values
        if fill_features_bool:
            default_frame = [self.value2index[feat_name][self.default_value[feat_name]] for feat_name in
                             self.index_order]
            sequence_filled = np.array([default_frame] * len(sequence))
            sequence_filled[:, selected_features_indices] = sequence
            sequence = sequence_filled

        start_time = 0.0
        for t in range(len(sequence)):
            pitch_ind = sequence[t, self.index_order_dict['pitch']]
            duration_ind = sequence[t, self.index_order_dict['duration']]
            velocity_ind = sequence[t, self.index_order_dict['velocity']]
            time_shift_ind = sequence[t, self.index_order_dict['time_shift']]

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

    def transform(self, x):

        ts_pos = self.index_order_dict['time_shift']
        vel_pos = self.index_order_dict['velocity']
        duration_pos = self.index_order_dict['duration']
        pitch_pos = self.index_order_dict['pitch']

        ############################
        #  Time shift
        if self.transformations['time_shift']:
            if len(x) > self.sequence_size:
                time_shift = math.floor(random.uniform(0, self.hop_size))
                x = x[time_shift:time_shift + self.sequence_size]
        else:
            x = x[:self.sequence_size]

        ############################
        #  Time dilation
        time_dilation_factor = 1
        if self.transformations['time_dilation']:
            time_dilation_factor = 1 - self.time_dilation_factor + 2 * self.time_dilation_factor * random.random()
            #  todo: version matricielle ??? Ca parait chaud quand meme...
            for t in range(len(x)):
                # time_shift
                old_ts = x[t, ts_pos]
                ts_value = self.index2value['time_shift'][old_ts]
                if ts_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                    continue
                ts_dilated = find_nearest_value(self.time_table, ts_value * time_dilation_factor)
                new_ts = self.value2index['time_shift'][ts_dilated]
                x[t, ts_pos] = new_ts

                #  duration
                old_duration = x[t, duration_pos]
                duration_value = self.index2value['duration'][old_duration]
                if duration_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                    continue
                duration_dilated = find_nearest_value(self.time_table, duration_value * time_dilation_factor)
                if duration_dilated == 0.0:
                    # smallest duration
                    new_duration = 0
                else:
                    new_duration = self.value2index['duration'][duration_dilated]
                x[t, duration_pos] = new_duration

        ############################
        #  Velocity dilation
        if self.transformations['velocity_shift']:
            velocity_shift = int(self.velocity_shift * (2 * random.random() - 1))
            for t in range(len(x)):
                # velocity
                old_vel = x[t, vel_pos]
                vel_value = self.index2value['velocity'][old_vel]
                if vel_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                    continue
                # Quantized here
                vel_dilated = round(vel_value + velocity_shift)
                vel_dilated = max(self.velocity_range[0], min(self.velocity_range[-1]-1, vel_dilated))
                new_vel = self.value2index['velocity'][vel_dilated]
                x[t, vel_pos] = new_vel

        ############################
        # Transposition
        if self.transformations['transposition']:
            # Draw a random transposition
            transposition = int(random.uniform(-self.max_transposition, self.max_transposition))

            if transposition != 0:
                x_trans = x
                for t in range(len(x)):
                    # pitch
                    old_pitch = x[t, pitch_pos]
                    pitch_value = self.index2value['pitch'][old_pitch]
                    if pitch_value in [PAD_SYMBOL, START_SYMBOL, END_SYMBOL]:
                        continue
                    pitch_transposed = pitch_value + transposition
                    if pitch_transposed not in self.value2index['pitch'].keys():
                        # Transposition not allowed for that chunk... don't transpose then
                        x_trans = x
                        break
                    new_pitch = self.value2index['pitch'][pitch_transposed]
                    x_trans[t, pitch_pos] = new_pitch
                x = x_trans

        return x, time_dilation_factor, velocity_shift
