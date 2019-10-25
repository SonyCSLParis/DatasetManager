import math
import os
import pickle
import random
import shutil

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from DatasetManager.helpers import END_SYMBOL, START_SYMBOL, \
    PAD_SYMBOL
from DatasetManager.piano.piano_helper import preprocess_midi, EventSeq, PianoIteratorGenerator, get_midi_type, \
    find_nearest_value

TS_0 = 'TSZ'

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class HarpsichordMidiDataset(data.Dataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    excluded_features = ['note_off', 'velocity']

    def __init__(self,
                 corpus_it_gen,
                 sequence_size,
                 max_transposition,
                 time_dilation_factor,
                 ):
        """
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
        self.list_ids = []
        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        assert self.sequence_size // 8, 'sequence_size needs to be a multiple of 8'
        self.last_index = None

        #  Data augmentations
        self.max_transposition = max_transposition
        self.time_dilation_factor = time_dilation_factor
        self.transformations = {
            'time_shift': True,
            'time_dilation': True,
            'transposition': True
        }

        self.time_table = EventSeq.time_shift_bins(insert_zero_time_token=True)

        # Midi 2 value
        self.value2midi = {}
        self.midi2value = {}
        self.midi_positions = ['note_on', 'note_off', 'time_shift', 'velocity']

        # Index 2 value
        self.index2value = {}
        self.value2index = {}
        self.index_positions = ['pitch', 'duration', 'velocity', 'time_shift']

        # Chunks
        self.hop_size = None

        # Precomputed values (for convenience)
        self.padding_chunk = None
        self.start_chunk = None
        self.end_chunk = None
        self.zero_time_shift = None

        dataset_dir = f'{cache_dir}/{self}'
        filename = f'{dataset_dir}/dataset.pkl'
        self.local_parameters = {
            'dataset_dir': dataset_dir,
            'filename': filename
        }

        #  Building/loading the dataset
        if os.path.isfile(filename):
            self.load()
        else:
            print(f'Building dataset {self}')
            self.make_tensor_dataset()
        return

    def __repr__(self):
        prefix = '-'.join(self.corpus_it_gen.subsets)
        name = f'HarpsichordMidi-' \
               f'{prefix}-' \
               f'{self.sequence_size}-' \
               f'{self.max_transposition}'
        return name

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

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

    def extract_subset(self, list_ids):
        instance = HarpsichordMidiDataset(corpus_it_gen=self.corpus_it_gen,
                                          sequence_size=self.sequence_size,
                                          max_transposition=self.max_transposition,
                                          time_dilation_factor=self.time_dilation_factor)
        instance.list_ids = list_ids
        return instance

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        dataset_dir = self.local_parameters["dataset_dir"]
        # Select sample
        id = self.list_ids[index]
        # Load data and get label
        x = torch.load(f'{dataset_dir}/x/{id}.pt')
        y = None  #  Reconstruction with notes_off and velocity in a near future
        # Apply transformations
        x, y = self.transform(x, y)
        return x, y

    def iterator_gen(self):
        return (arrangement_pair for arrangement_pair in self.corpus_it_gen())

    def data_loaders(self, batch_size, split=(0.85, 0.10), DEBUG_BOOL_SHUFFLE=True):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1

        num_examples = len(self)
        a, b = split
        train_ids = self.list_ids[: int(a * num_examples)]
        val_ids = self.list_ids[int(a * num_examples): int((a + b) * num_examples)]
        eval_ids = self.list_ids[int((a + b) * num_examples):]

        train_dataset = self.extract_subset(train_ids)
        val_dataset = self.extract_subset(val_ids)
        eval_dataset = self.extract_subset(eval_ids)

        train_dl = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=DEBUG_BOOL_SHUFFLE,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl

    def compute_index_dicts(self):
        #  From performance rnn
        # 100 duration from 10ms to 1s
        # 32 vellocities
        # dimension = 388
        # 30 seconds ~ 1200 frames
        # lowest_note, highest_note = self.reference_tessitura
        # lowest_pitch = note_to_midiPitch(lowest_note)
        # highest_pitch = note_to_midiPitch(highest_note)
        # list_midiPitch = sorted(list(range(lowest_pitch, highest_pitch + 1)))
        # list_velocity = list(range(self.velocity_quantization))
        # list_duration = list(self.duration_quantization)

        ######################################################################
        #  Midi 2 value
        self.midi_ranges = EventSeq.feat_ranges(excluded_features=[],
                                                insert_zero_time_token=False)
        for feat_name, feat_range in self.midi_ranges.items():
            midi2value = {}
            value2midi = {}
            for midi in feat_range:
                midi_shift = midi - feat_range[0]
                if feat_name == 'time_shift':
                    # +1 is because we don't use t=0 token
                    value = self.time_table[midi_shift+1]
                elif feat_name in ['note_on', 'note_off']:
                    value = EventSeq.pitch_range[midi_shift]
                elif feat_name == 'velocity':
                    value = EventSeq.velocity_range[midi_shift]
                else:
                    raise Exception

                midi2value[midi] = value
                value2midi[value] = midi
            self.midi2value[feat_name] = midi2value
            self.value2midi[feat_name] = value2midi

        ######################################################################
        #  Index 2 value
        for feat_name in self.index_positions:
            index2value = {}
            value2index = {}
            index = 0

            if feat_name == 'time_shift':
                values = self.time_table
            elif feat_name == 'duration':
                values = self.time_table[1:]
            elif feat_name == 'pitch':
                values = EventSeq.pitch_range
            elif feat_name == 'velocity':
                values = EventSeq.velocity_range
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
        self.padding_chunk = [self.value2index[feat_name][PAD_SYMBOL] for feat_name in self.index_positions]
        self.start_chunk = [self.value2index[feat_name][START_SYMBOL] for feat_name in self.index_positions]
        self.end_chunk = [self.value2index[feat_name][END_SYMBOL] for feat_name in self.index_positions]
        return

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Loading index dictionnary')

        self.compute_index_dicts()

        print('Making tensor dataset')

        chunk_counter = 0
        dataset_dir = self.local_parameters["dataset_dir"]
        if os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
        os.mkdir(f'{dataset_dir}/x')
        os.mkdir(f'{dataset_dir}/message_type')

        # Iterate over files
        for score_id, midi_file in tqdm(enumerate(self.iterator_gen())):

            #  Preprocess midi
            sequence = preprocess_midi(midi_file,
                                       excluded_features=[],
                                       insert_zero_time_token=False)

            structured_sequence = []
            for midi_ind in range(len(sequence)):
                midi = sequence[midi_ind]
                midi_type = get_midi_type(midi, self.midi_ranges)

                if midi_type == 'velocity':
                    velocity_value = self.midi2value['velocity'][midi]
                    velocity = self.value2index['velocity'][velocity_value]
                elif midi_type == 'note_on':
                    pitch_value = self.midi2value['note_on'][midi]
                    pitch = self.value2index['pitch'][pitch_value]
                    # Check the next midi symbol to see if there is a time_shift
                    next_midi = sequence[midi_ind + 1]
                    time_shift_value = 0.0 \
                        if (get_midi_type(next_midi, self.midi_ranges) != 'time_shift') \
                        else self.midi2value['time_shift'][next_midi]
                    time_shift = self.value2index['time_shift'][time_shift_value]
                    # Find next note_off (not efficient but fuck it)
                    duration_value = 0
                    for future_midi_ind in range(midi_ind + 1, len(sequence)):
                        future_midi = sequence[future_midi_ind]
                        future_midi_type = get_midi_type(future_midi, self.midi_ranges)
                        if future_midi_type == 'note_off':
                            if self.midi2value['note_off'][future_midi] == pitch_value:
                                break
                        elif future_midi_type == 'time_shift':
                            duration_value += self.midi2value['time_shift'][future_midi]
                    duration_value = find_nearest_value(self.time_table, duration_value)
                    duration = self.value2index['duration'][duration_value]

                    #  Add note
                    this_note = {
                        'duration': duration,
                        'time_shift': time_shift,
                        'pitch': pitch,
                        'velocity': velocity
                    }
                    this_note = [this_note[feat_name] for feat_name in self.index_positions]
                    structured_sequence.append(this_note)

            #  Build sequence
            self.hop_size = self.sequence_size // 4
            prepend_length = self.hop_size
            # Ensure seq_length is a multiple of hop_size
            raw_seq_len = len(structured_sequence)
            seq_length = math.ceil((prepend_length + raw_seq_len) / self.hop_size) * self.hop_size
            #  Append length (does not count final ts0 - end - ts0, so -3)
            append_length = seq_length - (prepend_length + raw_seq_len)

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

            # now split in chunks
            for start_time in range(0, len(sequence) - self.sequence_size + 1, self.hop_size):
                end_time = start_time + self.sequence_size
                # Take extra size to dynamically shift the chunks to the right when loading them (see def transform)
                # but not for edges
                edge_chunk = (start_time == 0) or (end_time == len(sequence))
                if not edge_chunk:
                    end_time += self.hop_size
                chunk = sequence[start_time:end_time]
                x_tensor = torch.tensor(chunk).long()
                torch.save(x_tensor, f'{dataset_dir}/x/{chunk_counter}.pt')
                self.list_ids.append(chunk_counter)
                chunk_counter += 1
        print(f'Chunks: {chunk_counter}\n')
        # Save class
        self.save()
        return

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        raise NotImplementedError

    def tensor_to_score(self, sequence, midipath):
        zero_time_event = self.feat_ranges['time_shift'][0]
        #  Filter out meta events
        removed_tokens = [zero_time_event] + self.meta_range
        sequence_clean = [int(e) for e in sequence if e not in removed_tokens]
        # Create EventSeq
        note_seq = EventSeq.from_array(sequence_clean, self.excluded_features, self.insert_zero_time_token).to_note_seq(
            self.insert_zero_time_token)
        note_seq.to_midi_file(midipath)

    def visualise_batch(self, piano_sequences, writing_dir, filepath):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if len(piano_sequences.size()) == 1:
            piano_sequences = torch.unsqueeze(piano_sequences, dim=0)

        num_batches = len(piano_sequences)

        for batch_ind in range(num_batches):
            self.tensor_to_score(sequence=piano_sequences[batch_ind],
                                 midipath=f"{writing_dir}/{filepath}_{batch_ind}.mid")

    def transform(self, x, messages):
        ############################
        #  Time shift
        if self.transformations['time_shift']:
            if len(x) > self.sequence_size:
                time_shift = math.floor(random.uniform(0, self.hop_size))
                x = x[time_shift:time_shift + self.sequence_size]
                messages = messages[time_shift:time_shift + self.sequence_size]

        ############################
        #  Time dilation
        if self.transformations['time_dilation']:
            avoid_dilation = False
            dilation_factor = 1 - self.time_dilation_factor + 2 * self.time_dilation_factor * random.random()
            # print(dilation_factor)
            ts_position = self.feature2position['time_shift']
            time_shift = list(x[:, ts_position].numpy())
            time_shift_absolute = [self.index2value[ts_position][e] for e in time_shift]
            time_shift_absolute_dilated = dilation_factor * time_shift_absolute
            new_time_shift = self.value2index[time_shift_absolute_dilated]
            for ind in range(len(x)):
                event_index = x[ind]
                feat_range = self.feat_ranges['time_shift']
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    abs_time = EventSeq.time_shift_bins(self.insert_zero_time_token)[event_value]
                    # scale time
                    scaled_abs_time = abs_time * dilation_factor
                    # if scaled_abs_time > EventSeq.time_shift_bins[-1]:
                    #     avoid_dilation = True
                    #     break
                    new_event_value = np.searchsorted(EventSeq.time_shift_bins(self.insert_zero_time_token),
                                                      scaled_abs_time, side='right') - 1
                    new_event_index = new_event_value + feat_range.start
                else:
                    new_event_index = event_index
                new_x.append(new_event_index)
            if not avoid_dilation:
                x = torch.tensor(new_x).long()

        ############################
        # Transposition
        # Draw a random transposition
        if self.transformations['transposition']:
            transposition = int(random.uniform(-self.max_transposition, self.max_transposition))
            if transposition == 0:
                return x, messages

            x_trans = x

            # First check transposition is possible
            transposable_types = [e for e in ['note_on', 'note_off'] if e not in self.excluded_features]
            for message_type in transposable_types:
                ranges = self.feat_ranges[message_type]
                min_value, max_value = min(ranges), max(ranges)
                authorized_transposition = torch.all((self.message_type_to_index[message_type] != messages) +
                                                     ((x + transposition <= max_value) * (
                                                             x + transposition >= min_value)))
                if not authorized_transposition:
                    return x, messages

            # Then transpose
            for message_type in transposable_types:
                # Mask
                x_trans = torch.where(self.message_type_to_index[message_type] == messages,
                                      x_trans + transposition,
                                      x_trans)
            x = x_trans

        return x, messages


if __name__ == '__main__':

    subsets = [
        # 'ecomp_piano_dataset',
        # 'classic_piano_dataset',
        'debug'
    ]
    corpus_it_gen = PianoIteratorGenerator(
        subsets=subsets,
        num_elements=None
    )

    dataset = HarpsichordMidiDataset(corpus_it_gen=corpus_it_gen,
                                     sequence_size=100,
                                     max_transposition=6,
                                     time_dilation_factor=0.1
                                     )

    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(batch_size=16, DEBUG_BOOL_SHUFFLE=False)

    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    number_dump = 100
    writing_dir = f'{os.path.expanduser("~")}/Data/dump/piano_midi/writing'
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, message_type_batch = sample_batched
        if i_batch > number_dump:
            break
        dataset.visualise_batch(piano_batch, writing_dir, filepath=f"{i_batch}")
