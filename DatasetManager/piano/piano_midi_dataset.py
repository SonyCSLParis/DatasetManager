import math
import os
import pickle
import random
import shutil

import torch
from torch.utils import data
from tqdm import tqdm

from DatasetManager.helpers import REST_SYMBOL, END_SYMBOL, START_SYMBOL, \
    PAD_SYMBOL
from DatasetManager.piano.piano_helper import preprocess_midi, EventSeq, PianoIteratorGenerator

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class PianoMidiDataset(data.Dataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 sequence_size,
                 max_transposition,
                 excluded_features):
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

        self.excluded_features = excluded_features
        self.list_ids = []
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        self.max_transposition = max_transposition
        self.last_index = None

        # One hot encoding
        self.feat_ranges = None
        self.meta_symbols_to_index = {}
        self.index_to_meta_symbols = {}
        self.meta_range = None
        self.message_type_to_index = {}
        self.index_to_message_type = {}
        self.one_hot_dimension = None

        # Chunks
        self.hop_size = None

        self.precomputed_vectors_piano = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            REST_SYMBOL: None,
        }

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
        name = f'PianoMidi-' \
            f'{self.name}-' \
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
        instance = PianoMidiDataset(corpus_it_gen=None,
                                    name=self.name,
                                    sequence_size=self.sequence_size,
                                    max_transposition=self.max_transposition,
                                    excluded_features=self.excluded_features)
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
        messages = torch.load(f'{dataset_dir}/message_type/{id}.pt')

        x_trans, mess_trans = self.transform(x, messages)

        return x_trans, mess_trans

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

        self.feat_ranges = EventSeq.feat_ranges(excluded_features=self.excluded_features)
        last_index = 0
        for k, v in self.feat_ranges.items():
            if v.stop > last_index:
                last_index = v.stop
        index = last_index
        self.last_index = last_index
        # Pad
        self.meta_symbols_to_index[PAD_SYMBOL] = index
        self.index_to_meta_symbols[index] = PAD_SYMBOL
        index += 1
        # Start
        self.meta_symbols_to_index[START_SYMBOL] = index
        self.index_to_meta_symbols[index] = START_SYMBOL
        index += 1
        # End
        self.meta_symbols_to_index[END_SYMBOL] = index
        self.index_to_meta_symbols[index] = END_SYMBOL
        index += 1
        self.meta_range = list(range(last_index, index))

        self.one_hot_dimension = index

        # Message types to index
        index = 0
        for message_type in self.feat_ranges.keys():
            self.message_type_to_index[message_type] = index
            self.index_to_message_type[index] = message_type
            index += 1
        self.message_type_to_index['meta'] = index
        self.index_to_message_type[index] = 'meta'
        return

    def extract_subsequence(self, seq, start, end):
        """
        Extract a clean midi subsequence from a sequence, i.e. a sequence starting with a note on
        :param seq:
        :param start:
        :param end:
        :return:
        """
        # subsequence = list(seq[start:end])
        # ret = []
        # for i in range(len(subsequence)):
        #     elem = subsequence.pop(0)
        #     if elem in self.feat_ranges['note_on']:
        #         ret.append(elem)
        #         break
        #     elif elem in self.feat_ranges['velocity']:
        #         ret.append(elem)
        # ret.extend(subsequence)
        return seq[start:end]

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
            sequence = preprocess_midi(midi_file, excluded_features=self.excluded_features)

            #  Assert only note, velocity and duration are here for now
            assert sequence.max() < self.last_index

            # Splits:
            # - constant size chunks (no really padding),
            # - with shift of a constant Half window size
            # - add a vector which indicate the locations of
            #       - time_shift
            #       - notes_on
            #       - notes_off
            #       - velocity
            #       - meta symbols
            seq_len = len(sequence)
            self.sequence_size = self.sequence_size
            self.hop_size = self.sequence_size // 4

            for t in range(-self.hop_size, seq_len, self.hop_size):
                prepend_seq = []
                append_seq = []

                start = max(0, t)
                virtual_last_index = t + self.sequence_size + self.hop_size
                end = min(seq_len, virtual_last_index)
                subsequence = self.extract_subsequence(sequence, start, end)

                #  Add Start and End plus padding for sides
                edge_chunk = False  #  This is used to disallow time shift in data augmentations
                if t < 0:
                    prepend_seq = [self.meta_symbols_to_index[PAD_SYMBOL]] * (-t - 1) + \
                                  [self.meta_symbols_to_index[START_SYMBOL]]
                    edge_chunk = True
                elif virtual_last_index > seq_len:
                    append_length = virtual_last_index - seq_len
                    #  Replace append_seq previously calculated with a new one containing the END symbol
                    append_seq = [self.meta_symbols_to_index[END_SYMBOL]] + \
                                 [self.meta_symbols_to_index[PAD_SYMBOL]] * (append_length - 1)
                    if t + self.sequence_size > seq_len:
                        edge_chunk = True

                chunk = prepend_seq + list(subsequence) + append_seq
                if edge_chunk:
                    chunk = chunk[:self.sequence_size]
                x_tensor = torch.tensor(chunk).long()
                torch.save(x_tensor, f'{dataset_dir}/x/{chunk_counter}.pt')

                # Build symbol_type sequence
                message_type_chunk = []
                for message in chunk:
                    meta_message = True
                    for message_type, ranges in self.feat_ranges.items():
                        if message in ranges:
                            message_type_chunk.append(self.message_type_to_index[message_type])
                            meta_message = False
                            break
                    if meta_message:
                        message_type_chunk.append(self.message_type_to_index['meta'])
                mess_tensor = torch.tensor(message_type_chunk).long()
                torch.save(mess_tensor, f'{dataset_dir}/message_type/{chunk_counter}.pt')

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
        #  Filter out meta events
        sequence_clean = [int(e) for e in sequence if e not in self.meta_range]
        # Create EventSeq
        EventSeq.from_array(sequence_clean).to_note_seq().to_midi_file(midipath)

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
        if len(x) > self.sequence_size:
            time_shift = math.floor(random.uniform(0, self.hop_size))
            x = x[time_shift:time_shift + self.sequence_size]
            messages = messages[time_shift:time_shift + self.sequence_size]

        ############################
        # Transposition
        # Draw a random transposition
        transposition = int(random.uniform(-self.max_transposition, self.max_transposition))
        if transposition == 0:
            return x, messages

        mess_trans = messages
        x_trans = x

        # First check transposition is possible
        transposable_types = [e for e in ['note_on', 'note_off'] if e not in self.excluded_features]
        for message_type in transposable_types:
            ranges = self.feat_ranges[message_type]
            min_value, max_value = min(ranges), max(ranges)
            authorized_transposition = torch.all((self.message_type_to_index[message_type] != messages) +
                                                 ((x + transposition <= max_value) * (x + transposition >= min_value)))
            if not authorized_transposition:
                return x, messages

        # Then transpose
        for message_type in transposable_types:
            # Mask
            x_trans = torch.where(self.message_type_to_index[message_type] == messages,
                                  x_trans + transposition,
                                  x_trans)

        return x_trans, mess_trans


if __name__ == '__main__':

    excluded_features = ['note_off', 'velocity']
    # excluded_features = []

    subsets = [
        # 'ecomp_piano_dataset',
        # 'classic_piano_dataset',
        'debug'
    ]
    corpus_it_gen = PianoIteratorGenerator(
        subsets=subsets,
        num_elements=None
    )

    name = '-'.join(subsets)
    name += '_' + '-'.join(excluded_features)
    dataset = PianoMidiDataset(corpus_it_gen=corpus_it_gen,
                               name=name,
                               sequence_size=600,
                               max_transposition=6,
                               excluded_features=excluded_features)

    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(batch_size=16, DEBUG_BOOL_SHUFFLE=False)

    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    number_dump = 100
    writing_dir = f"../dump/piano_midi/writing"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, message_type_batch = sample_batched
        if i_batch > number_dump:
            break
        dataset.visualise_batch(piano_batch, writing_dir, filepath=f"{i_batch}")
