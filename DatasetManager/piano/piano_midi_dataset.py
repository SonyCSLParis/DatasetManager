import json
import music21
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from DatasetManager.config import get_config
from DatasetManager.helpers import REST_SYMBOL, END_SYMBOL, START_SYMBOL, \
    PAD_SYMBOL, MASK_SYMBOL
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.piano.piano_helper import preprocess_midi, EventSeq

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class PianoMidiDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 sequence_size,
                 max_transposition,
                 integrate_discretization):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()

        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.sequence_size = sequence_size
        self.max_transposition = max_transposition
        self.integrate_discretization = integrate_discretization
        self.velocity_quantization = 32
        self.duration_quantization = np.arange(10, 1000, 10)  # in milliseconds

        config = get_config()

        arrangement_path = config["arrangement_path"]
        reference_tessitura_path = f'{arrangement_path}/reference_tessitura.json'
        self.dump_folder = config["dump_folder"]

        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = (music21.note.Note(tessitura['Piano'][0]), music21.note.Note(tessitura['Piano'][1]))

        self.feat_ranges = None
        self.meta_symbols_to_index = {}
        self.index_to_meta_symbols = {}
        self.meta_range = None
        self.message_type_to_index = {}
        self.index_to_message_type = {}

        self.precomputed_vectors_piano = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            MASK_SYMBOL: None,
            REST_SYMBOL: None,
        }

        return

    def __repr__(self):
        name = f'PianosMidipianoDataset-' \
            f'{self.name}-' \
            f'{self.sequence_size}-' \
            f'{self.max_transposition}'
        return name

    def iterator_gen(self):
        return (score for score in self.corpus_it_gen())

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

        # #  note on
        # index = 0
        # for midi_pitch in list_midiPitch:
        #     symbol = f'{midi_pitch}_on'
        #     self.symbol2index[symbol] = index
        #     self.index2symbol[index] = symbol
        #     index += 1
        #
        # #  note off
        # for midi_pitch in list_midiPitch:
        #     symbol = f'{midi_pitch}_off'
        #     self.symbol2index[symbol] = index
        #     self.index2symbol[index] = symbol
        #     index += 1
        #
        # # durations
        # for duration in list_duration:
        #     symbol = f'dur_{duration}'
        #     self.symbol2index[symbol] = index
        #     self.index2symbol[index] = symbol
        #     index += 1
        #
        # # velocities
        # for vel in list_velocity:
        #     symbol = f'vel_{vel}'
        #     self.symbol2index[symbol] = index
        #     self.index2symbol[index] = symbol
        #     index += 1
        #
        # # Mask (for nade like inference schemes)
        # self.symbol2index[MASK_SYMBOL] = index
        # self.index2symbol[index] = MASK_SYMBOL
        # index += 1
        # # Pad
        # self.symbol2index[PAD_SYMBOL] = index
        # self.index2symbol[index] = PAD_SYMBOL
        # index += 1
        # # Start
        # self.symbol2index[START_SYMBOL] = index
        # self.index2symbol[index] = START_SYMBOL
        # index += 1
        # # End
        # self.symbol2index[END_SYMBOL] = index
        # self.index2symbol[index] = END_SYMBOL
        # index += 1

        self.feat_ranges = EventSeq.feat_ranges()
        last_index = 0
        for k, v in self.feat_ranges.items():
            if v.stop > last_index:
                last_index = v.stop
        # Mask (for nade like inference schemes)
        index = last_index
        self.last_index = last_index
        self.meta_symbols_to_index[MASK_SYMBOL] = index
        self.index_to_meta_symbols[index] = MASK_SYMBOL
        index += 1
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

        # Message types to index
        index = 0
        for message_type in self.feat_ranges.keys():
            self.message_type_to_index[message_type] = index
            self.index_to_message_type[index] = message_type
            index += 1
        self.message_type_to_index['meta'] = index
        self.index_to_message_type[index] = 'meta'
        return

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    def get_score_tensor(self, scores, offsets):
        return None

    def get_metadata_tensor(self, score):
        return None

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return None

    def make_tensor_dataset(self, frame_orchestra=None):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Loading index dictionnary')

        self.compute_index_dicts()

        print('Making tensor dataset')

        total_chunk_counter = 0
        chunk_counter = 0
        impossible_transposition = 0

        #  Dataset
        piano_tensor_dataset = []
        # Store message types, to easily filter different messages depending on the application
        message_type_dataset = []

        # Iterate over files
        for score_id, midi_file in tqdm(enumerate(self.iterator_gen())):

            sequence = preprocess_midi(midi_file)
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
            chunk_size = self.sequence_size
            hop_size = chunk_size // 2
            for t in range(-hop_size, seq_len, hop_size):
                chunk_counter += 1
                total_chunk_counter += 1
                prepend_seq = []
                append_seq = []
                if t < 0:
                    prepend_seq = [self.meta_symbols_to_index[PAD_SYMBOL]] * (-t - 1) + \
                                  [self.meta_symbols_to_index[START_SYMBOL]]
                elif t + chunk_size > seq_len:
                    append_length = t + chunk_size - seq_len
                    append_seq = [self.meta_symbols_to_index[END_SYMBOL]] + \
                                 [self.meta_symbols_to_index[PAD_SYMBOL]] * (append_length - 1)
                start = max(0, t)
                end = min(seq_len, t + chunk_size)
                chunk = prepend_seq + list(sequence[start:end]) + append_seq
                piano_tensor_dataset.append(torch.tensor(chunk).long())
                # Build symbol_type sequence
                message_type_chunk = []
                for message in chunk:
                    meta_message = True
                    for message_type, ranges in self.feat_ranges.items():
                        if message in ranges:
                            message_type_chunk.append(self.message_type_to_index[message_type])
                            meta_message = False
                    if meta_message:
                        message_type_chunk.append(self.message_type_to_index['meta'])
                message_type_dataset.append(torch.tensor(message_type_chunk).long())

                #  Transpose
                #   shift up or down everything between 0 and 88 and 88 and 176
                #    after checking that it does not go out of range
                for transposition in range(-self.max_transposition, self.max_transposition):
                    if transposition == 0:
                        continue
                    chunk_transposed = []
                    skip_this_transposition = False
                    for message in chunk:
                        new_message = message
                        #  Transpose only if its a note message
                        for message_type in ['note_on', 'note_off']:
                            if message in self.feat_ranges[message_type]:
                                new_message = message + transposition
                                if new_message not in self.feat_ranges[message_type]:
                                    skip_this_transposition = True
                                    break
                        if skip_this_transposition:
                            break
                        chunk_transposed.append(new_message)
                    if skip_this_transposition:
                        impossible_transposition += 1
                        continue
                    else:
                        total_chunk_counter += 1
                        piano_tensor_dataset.append(torch.tensor(chunk_transposed).long())
                        #  Types are unchanged
                        message_type_dataset.append(torch.tensor(message_type_chunk).long())

        piano_tensor_dataset = torch.stack(piano_tensor_dataset, 0)
        message_type_dataset = torch.stack(message_type_dataset, 0)

        #######################
        #  Create Tensor Dataset
        dataset = TensorDataset(piano_tensor_dataset,
                                message_type_dataset)
        #######################

        print(
            f'### Sizes: \n'
            f'Piano: {piano_tensor_dataset.size()}\n'
            f'Chunks: {chunk_counter}\n'
            f'with transpos: {total_chunk_counter}\n'
            f'Impossible transpo: {impossible_transposition}')

        return dataset

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):
        raise NotImplementedError

    def tensor_to_score(self, sequence, midipath):
        print('coucou')
        #  Filter out meta events
        sequence_clean = [int(e) for e in sequence if e not in self.meta_range]
        # Create EventSeq
        EventSeq.from_array(sequence_clean).to_note_seq().to_midi_file(midipath)

    def visualise_batch(self, piano_sequences, writing_dir, filepath):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/piano_midi"

        if len(piano_sequences.size()) == 1:
            piano_sequences = torch.unsqueeze(piano_sequences, dim=0)

        num_batches = len(piano_sequences)

        for batch_ind in range(num_batches):
            self.tensor_to_score(sequence=piano_sequences[batch_ind],
                                 midipath=f"{writing_dir}/{filepath}_{batch_ind}.mid")
