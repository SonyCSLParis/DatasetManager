import json
import os
import re
import shutil

import torch
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from tqdm import tqdm
import music21
import numpy as np

from DatasetManager.arrangement.instrumentation import get_instrumentation, get_instrumentation_grouped
from DatasetManager.arrangement.instrument_grouping import get_instrument_grouping, get_instrument_grouping_section

from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, UNKNOWN_SYMBOL, TIME_SHIFT, PAD_SYMBOL, STOP_SYMBOL
from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align

from DatasetManager.config import get_config

from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, shift_pr_along_pitch_axis, \
    note_to_midiPitch, new_events, flatten_dict_pr

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END X X X

"""


class ArrangementMidipianoDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 corpus_it_gen_instru_range,
                 name,
                 mean_number_messages_per_time_frame,
                 subdivision=2,
                 sequence_size=3,
                 max_transposition=3,
                 transpose_to_sounding_pitch=True,
                 cache_dir=None,
                 compute_statistics_flag=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(ArrangementMidipianoDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.corpus_it_gen_instru_range = corpus_it_gen_instru_range
        self.subdivision = subdivision  # We use only on beats notes so far
        assert sequence_size % 2 == 1
        self.sequence_size = sequence_size
        self.max_transposition = max_transposition
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch

        #  Tessitura computed on data or use the reference tessitura ?
        self.compute_tessitura = True

        config = get_config()
        arrangement_path = config["arrangement_path"]
        reference_tessitura_path = f'{arrangement_path}/reference_tessitura.json'
        simplify_instrumentation_path = f'{arrangement_path}/simplify_instrumentation.json'

        self.dump_folder = config["dump_folder"]
        self.statistic_folder = self.dump_folder + '/arrangement/statistics'
        if os.path.isdir(self.statistic_folder):
            shutil.rmtree(self.statistic_folder)
        os.makedirs(self.statistic_folder)

        # Reference tessitura for each instrument
        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = {k: (music21.note.Note(v[0]), music21.note.Note(v[1])) for k, v in tessitura.items()}
        self.observed_tessitura = {}

        # Maps parts name found in mxml files to standard names
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)

        #  Instrumentation used for learning
        self.instrumentation = get_instrumentation()
        self.instrument_grouping = get_instrument_grouping()

        # Mapping between instruments and indices
        self.index2instrument = {}
        self.instrument2index = {}
        self.index2midi_pitch = {}
        self.midi_pitch2index = {}
        # Instruments presence
        self.instruments_presence2index = {}
        self.index2instruments_presence = {}
        #  Piano
        self.max_number_messages_piano = self.sequence_size * mean_number_messages_per_time_frame
        self.smallest_padding_length = self.max_number_messages_piano
        self.symbol2index_piano = {}
        self.index2symbol_piano = {}
        # Dimensions
        self.number_instruments = None

        # Often used vectors, computed in compute_index_dicts
        self.precomputed_vectors_orchestra = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            REST_SYMBOL: None,
        }

        self.precomputed_vectors_orchestra_instruments_presence = {
            UNKNOWN_SYMBOL: None
        }

        # Compute statistics slows down the construction of the dataset
        self.compute_statistics_flag = compute_statistics_flag
        return

    def __repr__(self):
        name = f'ArrangementMidipianoDataset-' \
            f'{self.name}-' \
            f'{self.subdivision}-' \
            f'{self.sequence_size}-' \
            f'{self.max_transposition}'
        return name

    def iterator_gen(self):
        return (arrangement_pair for arrangement_pair in self.corpus_it_gen())

    def iterator_gen_complementary(self):
        return (score for score in self.corpus_it_gen_instru_range())

    @staticmethod
    def pair2index(one_hot_0, one_hot_1):
        return one_hot_0 * 12 + one_hot_1

    def compute_index_dicts(self):
        if self.compute_tessitura:
            ############################################################
            ############################################################
            #  Mapping midi_pitch to token for each instrument

            # Hardcoded real life piano tessitura, so that we are not fucked at generation time if we see a score with out of range notes
            set_midiPitch_per_instrument = {'Piano': set(range(21, 108))}

            ############################################################
            # First pass over the database to create the mapping pitch <-> index for each instrument
            for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

                if arr_pair is None:
                    continue

                # Compute pianoroll representations of score (more efficient than manipulating the music21 streams)
                # pianoroll_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'], self.subdivision,
                #                                                       None,
                #                                                       self.instrument_grouping,
                #                                                       self.transpose_to_sounding_pitch)
                # pitch_set_this_track = set(np.where(np.sum(pianoroll_piano['Piano'], axis=0) > 0)[0])
                # set_midiPitch_per_instrument['Piano'] = set_midiPitch_per_instrument['Piano'].union(
                #     pitch_set_this_track)

                pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                                              self.simplify_instrumentation,
                                                                              self.instrument_grouping,
                                                                              self.transpose_to_sounding_pitch)
                for instrument_name in pianoroll_orchestra:
                    if instrument_name not in set_midiPitch_per_instrument.keys():
                        set_midiPitch_per_instrument[instrument_name] = set()
                    pitch_set_this_track = set(np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                    set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[instrument_name].union(
                        pitch_set_this_track)

            ############################################################
            # Potentially, we may want to also include ranges from an other database
            if self.corpus_it_gen_instru_range is not None:
                for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen_complementary())):

                    if arr_pair is None:
                        continue

                    pianoroll_orchestra, _, _ = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                                   self.simplify_instrumentation,
                                                                   self.instrument_grouping,
                                                                   self.transpose_to_sounding_pitch)
                    for instrument_name in pianoroll_orchestra:
                        if instrument_name not in set_midiPitch_per_instrument.keys():
                            set_midiPitch_per_instrument[instrument_name] = set()
                        pitch_set_this_track = set(
                            np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                        set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[
                            instrument_name].union(
                            pitch_set_this_track)
        else:
            set_midiPitch_per_instrument = {}
            instrument_name_list = list(self.instrumentation.keys())
            instrument_name_list.append("Piano")
            for instrument_name in instrument_name_list:
                lowest_note, highest_note = self.reference_tessitura[instrument_name]
                lowest_pitch = note_to_midiPitch(lowest_note)
                highest_pitch = note_to_midiPitch(highest_note)
                set_pitches = set(range(lowest_pitch, highest_pitch + 1))
                set_midiPitch_per_instrument[instrument_name] = set_pitches

        ############################################################
        # Save statistics in a file
        if self.compute_statistics_flag:
            with open(f"{self.statistic_folder}/note_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        ############################################################
        # Orchestra dicts
        midi_pitch2index_per_instrument = {}
        index2midi_pitch_per_instrument = {}
        for instrument_name, set_midiPitch in set_midiPitch_per_instrument.items():
            min_pitch = min(set_midiPitch)
            max_pitch = max(set_midiPitch)
            self.observed_tessitura[instrument_name] = {
                "min": min_pitch,
                "max": max_pitch
            }
            if instrument_name == "Piano":
                continue
            # Use range to avoid gaps in instruments tessitura (needed since we use
            # pitch transpositions as data augmentations
            list_midiPitch = sorted(list(range(min_pitch, max_pitch + 1)))
            midi_pitch2index_per_instrument[instrument_name] = {}
            index2midi_pitch_per_instrument[instrument_name] = {}
            for index, midi_pitch in enumerate(list_midiPitch):
                midi_pitch2index_per_instrument[instrument_name][midi_pitch] = index
                index2midi_pitch_per_instrument[instrument_name][index] = midi_pitch
            # Silence
            index += 1
            midi_pitch2index_per_instrument[instrument_name][REST_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = REST_SYMBOL
            #  Slur
            index += 1
            midi_pitch2index_per_instrument[instrument_name][SLUR_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = SLUR_SYMBOL
            # Pad
            index += 1
            midi_pitch2index_per_instrument[instrument_name][PAD_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = PAD_SYMBOL
            # Start
            index += 1
            midi_pitch2index_per_instrument[instrument_name][START_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = START_SYMBOL
            # End
            index += 1
            midi_pitch2index_per_instrument[instrument_name][END_SYMBOL] = index
            index2midi_pitch_per_instrument[instrument_name][index] = END_SYMBOL

        # Print instruments avoided
        print("Instruments not used")
        for instrument_name in midi_pitch2index_per_instrument.keys():
            if self.instrumentation[instrument_name] == 0:
                print(f'# {instrument_name}')

        # Mapping instruments <-> indices
        index_counter = 0
        for instrument_name, number_instruments in self.instrumentation.items():
            if instrument_name == "Piano":
                continue
            #  Check if instrument appears in the dataset
            if instrument_name not in midi_pitch2index_per_instrument.keys():
                continue

            #  Don't use instruments which are assigned 0 voices
            if number_instruments == 0:
                continue

            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments

        # Mapping pitch <-> index per voice (that's the one we'll use, easier to manipulate when training)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                self.midi_pitch2index[instrument_index] = midi_pitch2index_per_instrument[instrument_name]
                self.index2midi_pitch[instrument_index] = index2midi_pitch_per_instrument[instrument_name]
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # Piano dicts
        min_pitch_piano = self.observed_tessitura['Piano']['min']
        max_pitch_piano = self.observed_tessitura['Piano']['max'] + 1
        index = 0
        for midi_pitch in range(min_pitch_piano, max_pitch_piano):
            symbol = str(midi_pitch) + '_on'
            self.symbol2index_piano[symbol] = index
            self.index2symbol_piano[index] = symbol
            index += 1
            symbol = str(midi_pitch) + '_off'
            self.symbol2index_piano[symbol] = index
            self.index2symbol_piano[index] = symbol
            index += 1
        # Time shift
        self.symbol2index_piano[TIME_SHIFT] = index
        self.index2symbol_piano[index] = TIME_SHIFT
        index += 1
        # Time shift
        self.symbol2index_piano[STOP_SYMBOL] = index
        self.index2symbol_piano[index] = STOP_SYMBOL
        index += 1
        # Padding
        self.symbol2index_piano[PAD_SYMBOL] = index
        self.index2symbol_piano[index] = PAD_SYMBOL
        index += 1
        # Start
        self.symbol2index_piano[START_SYMBOL] = index
        self.index2symbol_piano[index] = START_SYMBOL
        index += 1
        # End
        self.symbol2index_piano[END_SYMBOL] = index
        self.index2symbol_piano[index] = END_SYMBOL
        ############################################################
        ############################################################

        ############################################################
        ############################################################
        # Encoding for orchestra presence
        # Same mapping for all instruments
        #  Unknown symbol is used for dropout during training, and also when generating if you don't want to
        # hard constrain the presence/absence of a note
        self.instruments_presence2index = {
            YES_SYMBOL: 0,
            NO_SYMBOL: 1,
            UNKNOWN_SYMBOL: 2
        }
        self.index2instruments_presence = {
            0: YES_SYMBOL,
            1: NO_SYMBOL,
            2: UNKNOWN_SYMBOL
        }
        ############################################################
        ############################################################

        self.number_instruments = len(self.midi_pitch2index)

        ############################################################
        ############################################################
        # These are the one-hot representation of several useful (especially during generation) vectors
        orchestra_start_vector = []
        orchestra_end_vector = []
        orchestra_pad_vector = []
        orchestra_rest_vector = []
        for instru_ind, mapping in self.midi_pitch2index.items():
            orchestra_start_vector.append(mapping[START_SYMBOL])
            orchestra_pad_vector.append(mapping[PAD_SYMBOL])
            orchestra_end_vector.append(mapping[END_SYMBOL])
            orchestra_rest_vector.append(mapping[REST_SYMBOL])
        self.precomputed_vectors_orchestra[START_SYMBOL] = torch.from_numpy(np.asarray(orchestra_start_vector)).long()
        self.precomputed_vectors_orchestra[PAD_SYMBOL] = torch.from_numpy(np.asarray(orchestra_pad_vector)).long()
        self.precomputed_vectors_orchestra[END_SYMBOL] = torch.from_numpy(np.asarray(orchestra_end_vector)).long()
        self.precomputed_vectors_orchestra[REST_SYMBOL] = torch.from_numpy(np.asarray(orchestra_rest_vector)).long()
        #
        unknown_vector = np.ones((self.number_instruments)) * self.instruments_presence2index[UNKNOWN_SYMBOL]
        self.precomputed_vectors_orchestra_instruments_presence[UNKNOWN_SYMBOL] = torch.from_numpy(
            unknown_vector).long()
        ############################################################
        ############################################################
        return

    def make_tensor_dataset(self, frame_orchestra=None):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        self.compute_index_dicts()

        total_chunk_counter = 0
        too_many_instruments_frame = 0
        impossible_transposition = 0

        # Variables for statistics
        if self.compute_statistics_flag:
            scores = []
            num_frames_with_different_pitch_class = 0
            total_frames_counter = 0
            open(f"{self.statistic_folder}/different_set_pc.txt", 'w').close()

        # List storing piano and orchestra datasets
        piano_tensor_dataset = []
        orchestra_tensor_dataset = []
        orchestra_instruments_presence_tensor_dataset = []

        # Iterate over files
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            ############################################################
            #  Align (we can use non transposed scores, changes nothing to the alignement
            if arr_pair is None:
                continue

            corresponding_frames, this_scores = self.align_score(arr_pair['Piano'], arr_pair['Orchestra'])
            if self.compute_statistics_flag:
                scores.extend(this_scores)
            # Get corresponding pitch_classes (for statistics)
            pc_piano_list = [e[0][1] for e in corresponding_frames]
            pc_orchestra_list = [e[1][1] for e in corresponding_frames]
            ############################################################

            # Prepare chunks of indices
            chunks_piano_indices, chunks_orchestra_indices = self.prepare_chunk_from_corresponding_frames(
                corresponding_frames)

            # Compute original pianorolls
            pr_piano, onsets_piano, _ = score_to_pianoroll(
                arr_pair['Piano'],
                self.subdivision,
                None,
                self.instrument_grouping,
                self.transpose_to_sounding_pitch)
            pr_piano = pr_piano["Piano"]
            onsets_piano = onsets_piano["Piano"]
            #  Binarize piano score
            pr_piano = np.where(pr_piano > 0, 1, 0)

            pr_orchestra, onsets_orchestra, _ = score_to_pianoroll(
                arr_pair['Orchestra'],
                self.subdivision,
                self.simplify_instrumentation,
                self.instrument_grouping,
                self.transpose_to_sounding_pitch)

            pr_pair = {"Piano": pr_piano, "Orchestra": pr_orchestra}
            onsets_pair = {"Piano": onsets_piano, "Orchestra": onsets_orchestra}

            # First get non transposed score
            transposition_semi_tone = 0
            minimum_transpositions_allowed = None
            maximum_transpositions_allowed = None
            minimum_transpositions_allowed, maximum_transpositions_allowed, \
            piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
            total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                              chunks_piano_indices, chunks_orchestra_indices,
                                              minimum_transpositions_allowed, maximum_transpositions_allowed,
                                              piano_tensor_dataset, orchestra_tensor_dataset,
                                              orchestra_instruments_presence_tensor_dataset,
                                              total_chunk_counter, too_many_instruments_frame, impossible_transposition)

            for transposition_semi_tone in range(-self.max_transposition, self.max_transposition + 1):
                if transposition_semi_tone == 0:
                    continue
                _, _, piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, total_chunk_counter, too_many_instruments_frame, impossible_transposition = \
                    self.transpose_loop_iteration(pr_pair, onsets_pair, transposition_semi_tone,
                                                  chunks_piano_indices, chunks_orchestra_indices,
                                                  minimum_transpositions_allowed, maximum_transpositions_allowed,
                                                  piano_tensor_dataset, orchestra_tensor_dataset,
                                                  orchestra_instruments_presence_tensor_dataset,
                                                  total_chunk_counter, too_many_instruments_frame,
                                                  impossible_transposition)

            if self.compute_statistics_flag:
                for pc_piano, pc_orchestra in zip(pc_piano_list, pc_orchestra_list):
                    total_frames_counter += 1
                    # Statistics: compare pitch class in orchestra and in piano
                    if pc_piano != pc_orchestra:
                        num_frames_with_different_pitch_class += 1
                        with open(f"{self.statistic_folder}/different_set_pc.txt", "a") as ff:
                            for this_pc in pc_piano:
                                ff.write(f"{this_pc} ")
                            ff.write("// ")
                            for this_pc in pc_orchestra:
                                ff.write(f"{this_pc} ")
                            ff.write("\n")

        piano_tensor_dataset = torch.cat(piano_tensor_dataset, 0)
        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)
        orchestra_instruments_presence_tensor_dataset = torch.cat(orchestra_instruments_presence_tensor_dataset, 0)

        print(f'Smalllest padding length: {self.smallest_padding_length}')

        #######################
        if self.compute_statistics_flag:
            # NW statistics
            mean_score = np.mean(scores)
            variance_score = np.var(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            nw_statistics_folder = f"{self.statistic_folder}/nw"
            if os.path.isdir(nw_statistics_folder):
                shutil.rmtree(nw_statistics_folder)
            os.makedirs(nw_statistics_folder)
            with open(f"{nw_statistics_folder}/scores.txt", "w") as ff:
                ff.write(f"Mean score: {mean_score}\n")
                ff.write(f"Variance score: {variance_score}\n")
                ff.write(f"Max score: {max_score}\n")
                ff.write(f"Min score: {min_score}\n")
                for elem in scores:
                    ff.write(f"{elem}\n")
            # Histogram
            n, bins, patches = plt.hist(scores, 50)
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Histogram NW scores')
            plt.savefig(f'{nw_statistics_folder}/histogram_score.pdf')

            # Pitch class statistics
            pitch_class_statistics_folder = f"{self.statistic_folder}/pitch_class"
            if os.path.isdir(pitch_class_statistics_folder):
                shutil.rmtree(pitch_class_statistics_folder)
            os.makedirs(pitch_class_statistics_folder)
            # Move different set pc
            shutil.move(f"{self.statistic_folder}/different_set_pc.txt",
                        f"{pitch_class_statistics_folder}/different_set_pc.txt")
            # Write the ratio of matching frames
            with open(f"{pitch_class_statistics_folder}/ratio_matching_pc_set.txt", "w") as ff:
                ff.write(f"Different PC sets: {num_frames_with_different_pitch_class}\n")
                ff.write(f"Total number frames: {total_frames_counter}\n")
                ff.write(f"Ratio: {num_frames_with_different_pitch_class / total_frames_counter}\n")
        #######################

        #######################
        #  Create Tensor Dataset
        dataset = TensorDataset(piano_tensor_dataset,
                                orchestra_tensor_dataset,
                                orchestra_instruments_presence_tensor_dataset)
        #######################

        print(
            f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n Orchestra: {orchestra_tensor_dataset.size()}\n')
        print(
            f'Chunks: {total_chunk_counter}\nToo many instru chunks: {too_many_instruments_frame}\nImpossible transpo: {impossible_transposition}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, score, datatype):
        #  Get pianorolls
        if datatype == 'piano':
            simplify_instrumentation = None
        elif datatype == 'orchestra':
            simplify_instrumentation = self.simplify_instrumentation
        pianoroll, onsets, _ = score_to_pianoroll(score, self.subdivision,
                                                  simplify_instrumentation,
                                                  self.instrument_grouping,
                                                  self.transpose_to_sounding_pitch)
        flat_pr = flatten_dict_pr(pianoroll)

        #  Get new events indices (diff matrices)
        events = new_events(pianoroll, onsets)
        #  Pad at the end of the pitch axis to get a multiple of 12 (number of pitch classes)
        pr_event = np.pad(flat_pr[events], pad_width=[(0, 0), (0, 4)], mode='constant', constant_values=0)

        # Reduce on 12 pitch-classes
        length = len(events)
        # test = np.repeat(np.expand_dims(np.arange(0, 132), 0), length, axis=0)
        pcs = np.sum(np.reshape(pr_event, (length, 11, 12)), axis=1)

        list_pc = []
        for frame_index in range(len(events)):
            list_pc.append((events[frame_index], set(np.where(pcs[frame_index] > 0)[0])))

        return list_pc

    def align_score(self, piano_score, orchestra_score):
        list_pc_piano = self.score_to_list_pc(piano_score, "piano")
        list_pc_orchestra = self.score_to_list_pc(orchestra_score, "orchestra")

        only_pc_piano = [e[1] for e in list_pc_piano]
        only_pc_orchestra = [e[1] for e in list_pc_orchestra]

        corresponding_indices, score_matrix = nw_align.nwalign(only_pc_piano, only_pc_orchestra, gapOpen=-3,
                                                               gapExtend=-1)

        corresponding_frames = [(list_pc_piano[ind_piano], list_pc_orchestra[ind_orchestra])
                                for ind_piano, ind_orchestra in corresponding_indices]

        return corresponding_frames, score_matrix

    def get_allowed_transpositions_from_pr(self, pr, frames, instrument_name):
        #  Get min and max pitches
        pr_frames = np.asarray(
            [pr[frame] for frame in frames if frame not in [PAD_SYMBOL, REST_SYMBOL, START_SYMBOL, END_SYMBOL]])
        # [pr[frame] for frame in frames if frame not in [REST_SYMBOL, MASK_SYMBOL, START_SYMBOL, END_SYMBOL]])
        flat_pr = pr_frames.sum(axis=0)
        non_zeros_pitches = list(np.where(flat_pr > 0)[0])
        if len(non_zeros_pitches) > 0:
            min_pitch = min(non_zeros_pitches)
            max_pitch = max(non_zeros_pitches)

            # Compare with reference tessitura, and ensure min <= 0 and max >= 0
            allowed_transposition_down = min(0, self.observed_tessitura[instrument_name]["min"] - min_pitch)
            allowed_transposition_up = max(0, self.observed_tessitura[instrument_name]["max"] - max_pitch)
        else:
            allowed_transposition_down = None
            allowed_transposition_up = None

        return allowed_transposition_down, allowed_transposition_up

    def prepare_chunk_from_corresponding_frames(self, corresponding_frames):
        chunks_piano_indices = []
        chunks_orchestra_indices = []
        number_corresponding_frames = len(corresponding_frames)
        for index_frame in range(0, number_corresponding_frames):
            # if we consider the time in the middle is the one of interest, we must pad half of seq size at the beginning and half at the end
            start_index = index_frame - (self.sequence_size - 1) // 2
            start_index_truncated = max(0, start_index)
            #  Always add at least one None frame at the beginning (instead of a the real previous frame)
            #  Hence, we avoid the model observe slurs from unseen previous frame
            padding_beginning = start_index_truncated - start_index
            end_index = index_frame + (self.sequence_size - 1) // 2
            end_index_truncated = min(number_corresponding_frames, end_index)
            padding_end = max(0, end_index - number_corresponding_frames + 1)

            #  Always include a None frame as first frame instead of the real previous frame.
            # This is because we don't want to have slurs from frames that the model cannot observe
            this_piano_chunk = [e[0][0] for e in corresponding_frames[start_index_truncated:end_index_truncated + 1]]
            this_orchestra_chunk = [e[1][0] for e in
                                    corresponding_frames[start_index_truncated:end_index_truncated + 1]]

            # Padding
            if padding_beginning == 0:
                prepend_vector = []
            else:
                prepend_vector = (padding_beginning - 1) * [PAD_SYMBOL] + [START_SYMBOL]

            if padding_end == 0:
                append_vector = []
            else:
                append_vector = [END_SYMBOL] + (padding_end - 1) * [PAD_SYMBOL]

            this_piano_chunk = prepend_vector + this_piano_chunk + append_vector
            this_orchestra_chunk = prepend_vector + this_orchestra_chunk + append_vector
            chunks_piano_indices.append(this_piano_chunk)
            chunks_orchestra_indices.append(this_orchestra_chunk)
        return chunks_piano_indices, chunks_orchestra_indices

    def transpose_loop_iteration(self, pianorolls_pair, onsets_pair, transposition_semi_tone,
                                 chunks_piano_indices, chunks_orchestra_indices,
                                 minimum_transposition_allowed, maximum_transposition_allowed,
                                 piano_tensor_dataset, orchestra_tensor_dataset,
                                 orchestra_instruments_presence_tensor_dataset,
                                 total_chunk_counter, too_many_instruments_frame, impossible_transposition):

        ############################################################
        # Transpose pianorolls
        this_pr_piano = shift_pr_along_pitch_axis(pianorolls_pair["Piano"], transposition_semi_tone)
        this_onsets_piano = shift_pr_along_pitch_axis(onsets_pair["Piano"], transposition_semi_tone)

        this_pr_orchestra = {}
        this_onsets_orchestra = {}
        for instrument_name in pianorolls_pair["Orchestra"].keys():
            # Pr
            pr = pianorolls_pair["Orchestra"][instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            this_pr_orchestra[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_pair["Orchestra"][instrument_name]
            shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
            this_onsets_orchestra[instrument_name] = shifted_onsets
        ############################################################

        if minimum_transposition_allowed is None:
            if transposition_semi_tone != 0:
                raise Exception("Possible transpositions should be computed on non transposed pianorolls")
            # We have to construct the possible transpose
            build_allowed_transposition_flag = True
            minimum_transposition_allowed = []
            maximum_transposition_allowed = []
        else:
            build_allowed_transposition_flag = False

        for chunk_index in range(len(chunks_piano_indices)):
            this_chunk_piano_indices = chunks_piano_indices[chunk_index]
            this_chunk_orchestra_indices = chunks_orchestra_indices[chunk_index]
            avoid_this_chunk = False
            total_chunk_counter += 1

            ############################################################
            #  Get list of allowed transpositions
            if build_allowed_transposition_flag:
                min_transposition = -self.max_transposition
                max_transposition = self.max_transposition

                # Observe tessitura for each instrument for this chunk. Use non transposed pr of course
                this_min_transposition, this_max_transposition = \
                    self.get_allowed_transpositions_from_pr(this_pr_piano,
                                                            this_chunk_piano_indices,
                                                            "Piano")

                if (this_min_transposition is None) or (this_max_transposition is None):
                    min_transposition = None
                    max_transposition = None
                else:
                    min_transposition = max(this_min_transposition, min_transposition)
                    max_transposition = min(this_max_transposition, max_transposition)

                # Use reference tessitura or compute tessitura directly on the files ?
                if min_transposition is not None:
                    for instrument_name, pr in this_pr_orchestra.items():
                        this_min_transposition, this_max_transposition = \
                            self.get_allowed_transpositions_from_pr(pr,
                                                                    this_chunk_orchestra_indices,
                                                                    instrument_name)
                        if this_min_transposition is not None:  # If instrument not in this chunk, None was returned
                            min_transposition = max(this_min_transposition, min_transposition)
                            max_transposition = min(this_max_transposition, max_transposition)

                    this_minimum_transposition_allowed = min(0, min_transposition)
                    this_maximum_transposition_allowed = max(0, max_transposition)
                else:
                    this_minimum_transposition_allowed = None
                    this_maximum_transposition_allowed = None
                minimum_transposition_allowed.append(this_minimum_transposition_allowed)
                maximum_transposition_allowed.append(this_maximum_transposition_allowed)
            else:
                this_minimum_transposition_allowed = minimum_transposition_allowed[chunk_index]
                this_maximum_transposition_allowed = maximum_transposition_allowed[chunk_index]

            ############################################################
            #  Test if the transposition is possible
            if (this_maximum_transposition_allowed is None) or (this_minimum_transposition_allowed is None):
                impossible_transposition += 1
                continue
            if (this_minimum_transposition_allowed > transposition_semi_tone) \
                    or (this_maximum_transposition_allowed < transposition_semi_tone):
                impossible_transposition += 1
                continue

            ############################################################
            #  Extract representations
            local_piano_tensor = []
            local_orchestra_tensor = []
            local_orchestra_instruments_presence_tensor = []
            previous_frame_orchestra = None
            for index_frame in range(len(this_chunk_piano_indices)):
                if index_frame != 0:
                    previous_frame_piano = this_chunk_piano_indices[index_frame - 1]
                    if previous_frame_piano in [START_SYMBOL, END_SYMBOL]:
                        previous_frame_piano = None
                else:
                    previous_frame_piano = None
                frame_piano = this_chunk_piano_indices[index_frame]
                frame_orchestra = this_chunk_orchestra_indices[index_frame]

                if frame_orchestra in [START_SYMBOL, PAD_SYMBOL, END_SYMBOL]:
                    if frame_piano == END_SYMBOL:
                        #  Remove TS (after asserting there is one...)
                        assert local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT], "Weirdness"
                        local_piano_tensor.pop()
                        # Add end symbol, then the stop symbol
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                        local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])
                    else:
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = \
                        self.precomputed_vectors_orchestra_instruments_presence[UNKNOWN_SYMBOL].clone().detach()
                else:
                    local_piano_tensor = self.pianoroll_to_piano_tensor(
                        pr=this_pr_piano,
                        onsets=this_onsets_piano,
                        previous_frame_index=previous_frame_piano,
                        frame_index=frame_piano,
                        piano_vector=local_piano_tensor)

                    #  Time shift piano
                    local_piano_tensor.append(self.symbol2index_piano[TIME_SHIFT])

                    orchestra_t_encoded, orchestra_instruments_presence_t_encoded = self.pianoroll_to_orchestral_tensor(
                        this_pr_orchestra,
                        this_onsets_orchestra,
                        previous_frame=previous_frame_orchestra,
                        frame_index=frame_orchestra)

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)
                previous_frame_orchestra = orchestra_t_encoded

            # If last symbol is not a PAD  Replace the last TIME_SHIFT added by a STOP
            if local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT]:
                local_piano_tensor.pop()
                local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])

            #  Pad piano vector to max length
            padding_length = self.max_number_messages_piano - len(local_piano_tensor)
            if padding_length < 0:
                raise Exception(
                    f"Padding length is equal to {padding_length}, consider increasing the max length of the encoding vector")
            self.smallest_padding_length = min(self.smallest_padding_length, padding_length)
            for _ in range(padding_length):
                local_piano_tensor.append(self.symbol2index_piano[PAD_SYMBOL])
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.max_number_messages_piano
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.LongTensor(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
               total_chunk_counter, too_many_instruments_frame, impossible_transposition

    def pianoroll_to_piano_tensor(self, pr, onsets, previous_frame_index, frame_index, piano_vector):

        notes = [(e, True) for e in list(np.where(onsets[frame_index])[0])]
        if previous_frame_index is not None:
            delta_pr = pr[previous_frame_index] - pr[frame_index]
            notes += [(e, False) for e in list(np.where(delta_pr)[0])]

        # Sort notes from lowest to highest
        #  Todo check register individually for each voice for smaller categorical representations
        notes = sorted(notes, key=lambda e: e[0])

        #  Build symbols and append corresponding index in midi_messages
        previous_note = None
        for (note, on) in notes:
            #  Filter out note_off for repeated notes (uselessely crowds representation)
            if (previous_note == note) and (not on):
                continue
            if on:
                symbol = str(note) + '_on'
            else:
                symbol = str(note) + '_off'
            index = self.symbol2index_piano[symbol]
            piano_vector.append(index)
            previous_note = note
        return piano_vector

    def pianoroll_to_orchestral_tensor(self, pianoroll, onsets, previous_frame, frame_index):
        orchestra_encoded = np.zeros((self.number_instruments))
        orchestra_instruments_presence = np.zeros((self.number_instruments))
        for instrument_name, indices_instruments in self.instrument2index.items():
            number_of_parts = len(indices_instruments)
            if instrument_name not in pianoroll:
                notes_played = []
            else:
                notes_played = list(np.where(pianoroll[instrument_name][frame_index])[0])
            if len(notes_played) > number_of_parts:
                return None, None

            notes_played = sorted(notes_played)

            # Pad with silences
            notes_played.extend([REST_SYMBOL] * (number_of_parts - len(notes_played)))
            for this_note, this_instrument_index in zip(notes_played, indices_instruments):
                slur_bool = False
                if this_note != REST_SYMBOL:
                    slur_bool = (onsets[instrument_name][frame_index, this_note] == 0)

                if slur_bool and (previous_frame is not None):
                    #  After alignement,  it is possible that some frames have been removed, leading to inconsistent slurs symbols
                    # (like slur after a rest)
                    if previous_frame[this_instrument_index] == self.midi_pitch2index[this_instrument_index][this_note]:
                        orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][
                            SLUR_SYMBOL]
                    else:
                        orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][
                            this_note]
                else:
                    if this_note not in self.midi_pitch2index[this_instrument_index].keys():
                        this_note = REST_SYMBOL
                    orchestra_encoded[this_instrument_index] = self.midi_pitch2index[this_instrument_index][this_note]

                # Keep trace of instruments presence
                if this_note in [REST_SYMBOL, START_SYMBOL, END_SYMBOL]:
                    orchestra_instruments_presence[this_instrument_index] = self.instruments_presence2index[NO_SYMBOL]
                else:
                    orchestra_instruments_presence[this_instrument_index] = self.instruments_presence2index[
                        YES_SYMBOL]

        orchestra_tensor = torch.from_numpy(orchestra_encoded).long()
        orchestra_instruments_presence_tensor = torch.from_numpy(orchestra_instruments_presence).long()

        return orchestra_tensor, orchestra_instruments_presence_tensor

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[],
                                 subdivision=None):

        ts_ind = self.symbol2index_piano[TIME_SHIFT]
        start_ind = self.symbol2index_piano[START_SYMBOL]
        pad_ind = self.symbol2index_piano[PAD_SYMBOL]
        end_ind = self.symbol2index_piano[END_SYMBOL]
        meta_inds = [ts_ind, start_ind, end_ind, pad_ind]
        stop_ind = self.symbol2index_piano[STOP_SYMBOL]

        # Get pianorolls
        score_piano = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.subdivision
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(score_piano,
                                                              subdivision,
                                                              simplify_instrumentation=None,
                                                              instrument_grouping=self.instrument_grouping,
                                                              transpose_to_sounding_pitch=self.transpose_to_sounding_pitch)

        rhythm_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        piano_init_list = []
        previous_frame_index = None
        for frame_index in rhythm_piano:
            piano_init_list = self.pianoroll_to_piano_tensor(
                pr=pianoroll_piano["Piano"],
                onsets=onsets_piano,
                previous_frame_index=previous_frame_index,
                frame_index=frame_index,
                piano_vector=piano_init_list)
            previous_frame_index = frame_index

            #  Time shift piano
            piano_init_list.append(ts_ind)

        #  Remove the last time shift added
        piano_init_list.pop()

        # Prepend rests frames at the beginning and end of the piano score
        piano_init_list = [pad_ind] * (context_length - 1) + \
                          [start_ind] + \
                          piano_init_list + \
                          [end_ind] + \
                          [pad_ind] * (context_length - 1)

        # Use a non-sliced version for writing back the piano file
        piano_write = torch.tensor(piano_init_list).unsqueeze(0).repeat(batch_size, 1)

        #  Prepare chunks with padding for piano input
        def write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag):
            piano_init[chunk_index, :len(this_chunk)] = torch.tensor(this_chunk)
            if writing_stop_flag:
                piano_init[chunk_index, len(this_chunk)] = stop_ind
                piano_init[chunk_index, len(this_chunk) + 1:] = pad_ind
            else:
                piano_init[chunk_index, len(this_chunk):] = pad_ind
            return piano_init

        piano_init = torch.zeros(len(rhythm_piano), self.max_number_messages_piano)
        this_chunk = []
        chunk_index = 0
        frame_counter = 0
        writing_stop_flag = True

        for piano_symbol in piano_init_list:
            if piano_symbol in meta_inds:
                if frame_counter >= self.sequence_size - 1:
                    #  Write in piano_init
                    piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)
                    #  Clean this_chunk
                    while this_chunk[0] not in meta_inds:
                        this_chunk.pop(0)
                    this_chunk.pop(0)
                    chunk_index += 1
                frame_counter += 1

            if piano_symbol == end_ind:
                this_chunk.append(end_ind)
                this_chunk.append(stop_ind)
                writing_stop_flag = False
            else:
                this_chunk.append(piano_symbol)

        #  Don't forget last chunk
        piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)

        # Orchestra
        padding_length = context_length * 2
        num_frames = len(rhythm_piano) + padding_length
        orchestra_silences, orchestra_unknown, orchestra_init = self.init_orchestra(num_frames, context_length,
                                                                                    banned_instruments,
                                                                                    unknown_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)

        return piano_init.long().cuda(), piano_write.long(), rhythm_piano, \
               orchestra_init.long().cuda(), orchestra_silences, orchestra_unknown

    def init_orchestra(self, num_frames, context_length, banned_instruments, unknown_instruments):
        # Set orchestra constraints in the form of banned instruments
        orchestra_silences = []
        orchestra_unknown = []
        orchestra_init = torch.zeros(num_frames, self.number_instruments)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                if instrument_name in banned_instruments:
                    # -1 is a silence
                    orchestra_silences.append(1)
                    orchestra_init[:, instrument_index] = self.midi_pitch2index[instrument_index][REST_SYMBOL]
                elif instrument_name in unknown_instruments:
                    # Note that an instrument can't be both banned and unknown
                    orchestra_unknown.append(1)
                else:
                    orchestra_silences.append(0)
                    orchestra_unknown.append(0)
                    #  Initialise with last
                    masking_value = len(self.midi_pitch2index[instrument_index])
                    orchestra_init[:, instrument_index] = masking_value

        # Start and end symbol at the beginning and end
        orchestra_init[:context_length - 1] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        orchestra_init[context_length - 1] = self.precomputed_vectors_orchestra[START_SYMBOL]
        orchestra_init[-context_length] = self.precomputed_vectors_orchestra[END_SYMBOL]
        orchestra_init[-context_length:] = self.precomputed_vectors_orchestra[PAD_SYMBOL]
        return orchestra_silences, orchestra_unknown, orchestra_init

    def piano_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
        # (batch, num_parts, notes_encoding)
        elems = list(tensor_score.numpy())

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            # Note: using len(elems) yields a duration vector which is too long, but fuck it
            durations = np.ones((len(elems))) * subdivision

        #  Init score
        stream = music21.stream.Stream()
        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(0, music21_instrument)

        # Keep track of the duration of notes on
        notes_on_duration = {}
        frame_index = 0
        global_offset = 0
        for elem in elems:
            #  DEBUG
            if elem == -1:
                # This is for debugging, to separate between batches
                f = music21.note.Rest()
                f.quarterLength = 1
                this_part.insert(global_offset / subdivision, f)
                global_offset += subdivision
                continue

            #  Get symbol corresponding to one-hot encoding
            symbol = self.index2symbol_piano[elem]

            if symbol in [START_SYMBOL, END_SYMBOL, PAD_SYMBOL]:
                global_offset += durations[frame_index]
                # frame_index += 1
            elif symbol in [TIME_SHIFT, STOP_SYMBOL]:
                # print(f'TS')
                # Increment duration in notes_on_duration
                global_offset += durations[frame_index]
                for pitch, _ in notes_on_duration.items():
                    notes_on_duration[pitch]['duration'] += durations[frame_index]
                #  increment frame counter
                frame_index += 1

                if symbol == STOP_SYMBOL:
                    # Write the last notes_on
                    for pitch, values in notes_on_duration.items():
                        duration = values['duration']
                        offset = values['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    break
            else:
                pitch, on = re.split('_', symbol)

                # print(f'{pitch} {on}')

                #  Repeat case
                if on == 'on':
                    if pitch in notes_on_duration.keys():
                        #  Write this pitch with its associated duration
                        duration = notes_on_duration[pitch]['duration']
                        offset = notes_on_duration[pitch]['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    #  Either it's a repeat or not, add in notes_on_duration
                    #  Write 0, as durations are all updated when a time shift event is met
                    else:
                        #  If it's not a note repeat need to instantiate the note in the dict
                        notes_on_duration[pitch] = {}
                    notes_on_duration[pitch]['duration'] = 0
                    notes_on_duration[pitch]['offset'] = global_offset
                # Note off
                elif on == 'off':
                    # Write this pitch with its associated duration
                    if pitch not in notes_on_duration.keys():
                        continue
                    duration = notes_on_duration[pitch]['duration']
                    offset = notes_on_duration[pitch]['offset']
                    f = music21.note.Note(int(pitch))
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)
                    #  Remove from notes_on_duration
                    notes_on_duration.pop(pitch, None)
                else:
                    raise Exception("should not happen")

        this_part.atSoundingPitch = self.transpose_to_sounding_pitch
        stream.append(this_part)

        return stream

    def orchestra_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
        # (batch, num_parts, notes_encoding)
        orchestra_matrix = tensor_score.numpy()
        length = len(orchestra_matrix)

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            durations = np.ones((length)) * subdivision
        else:
            assert length == len(durations), "Rhythm vector must be the same length as tensor[0]"

        total_duration_ql = sum(durations) / subdivision

        # First store every in a dict {instrus : [time [notes]]}
        score_dict = {}
        for instrument_index in range(self.number_instruments):
            # Get instrument name
            instrument_name = self.index2instrument[instrument_index]
            if instrument_name not in score_dict:
                score_dict[instrument_name] = []

            duration = 0
            offset = 0
            this_offset = 0
            pitch = None
            for frame_index, this_duration in enumerate(durations):
                try:
                    this_pitch = self.index2midi_pitch[instrument_index][orchestra_matrix[frame_index, instrument_index]]
                except:
                    print('yo')

                if this_pitch == SLUR_SYMBOL:
                    duration += this_duration
                else:
                    if pitch is not None:
                        #  Write previous pitch

                        #  For debugging
                        # if pitch == MASK_SYMBOL:
                        #     # Special treatment for PADDING frames
                        #     score_dict[instrument_name].append((0, offset, subdivision))  # In fact, pitch 0 = C4
                        # elif pitch == START_SYMBOL:
                        #     # Special treatment for PADDING frames
                        #     score_dict[instrument_name].append((0, offset, subdivision))
                        #     score_dict[instrument_name].append((6, offset, subdivision))
                        # elif pitch == END_SYMBOL:
                        #     # Special treatment for PADDING frames
                        #     score_dict[instrument_name].append((6, offset, subdivision))
                        #     score_dict[instrument_name].append((7, offset, subdivision))
                        # elif pitch != REST_SYMBOL:
                        #     #  Write previous event
                        #     score_dict[instrument_name].append((pitch, offset, duration))

                        if pitch not in [START_SYMBOL, END_SYMBOL, REST_SYMBOL, PAD_SYMBOL]:
                            # if pitch not in [MASK_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL]:
                            #  Write previous event
                            score_dict[instrument_name].append((pitch, offset, duration))

                    # Reset values
                    duration = this_duration
                    pitch = this_pitch
                    offset = this_offset

                this_offset += this_duration

            # Last note
            if pitch != REST_SYMBOL:
                #  Todo DEBUG
                # if pitch == MASK_SYMBOL:
                #     score_dict[instrument_name].append((0, offset, duration))
                # elif pitch == START_SYMBOL:
                #     # Special treatment for PADDING frames
                #     score_dict[instrument_name].append((0, offset, subdivision))
                #     score_dict[instrument_name].append((6, offset, subdivision))
                # elif pitch == END_SYMBOL:
                #     # Special treatment for PADDING frames
                #     score_dict[instrument_name].append((6, offset, subdivision))
                #     score_dict[instrument_name].append((7, offset, subdivision))
                # elif pitch != REST_SYMBOL:

                # if pitch not in [REST_SYMBOL, END_SYMBOL, START_SYMBOL, MASK_SYMBOL]:
                if pitch not in [REST_SYMBOL, END_SYMBOL, START_SYMBOL, PAD_SYMBOL]:
                    score_dict[instrument_name].append((pitch, offset, duration))

        #  Batch is used as time in the score
        stream = music21.stream.Stream()

        for instrument_name, elems in score_dict.items():
            this_part = music21.stream.Part(id=instrument_name)
            #  re is for removing underscores in instrument names which raise errors in music21
            if instrument_name == "Cymbal":
                music21_instrument = music21.instrument.Cymbals()
            elif instrument_name == "Woodwind":
                music21_instrument = music21.instrument.fromString("Clarinet")
            elif instrument_name == "String":
                music21_instrument = music21.instrument.fromString("Violoncello")
            elif instrument_name == "Brass":
                music21_instrument = music21.instrument.fromString("Horn")
            else:
                music21_instrument = music21.instrument.fromString(re.sub('_', ' ', instrument_name))
            this_part.insert(0, music21_instrument)

            # Tempo
            # t = music21.tempo.MetronomeMark(writing_tempo)
            # this_part.insert(0, t)

            if elems == []:
                f = music21.note.Rest()
                f.quarterLength = total_duration_ql
                this_part.insert(0, f)
            else:
                for elem in elems:
                    pitch, offset, duration = elem
                    f = music21.note.Note(pitch)
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)

            # this_part_chordified = this_part.chordify()
            # this_part_chordified.atSoundingPitch = self.transpose_to_sounding_pitch
            # stream.append(this_part_chordified)

            this_part.atSoundingPitch = self.transpose_to_sounding_pitch
            stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano=None, writing_dir=None,
                        filepath=None, writing_tempo='adagio', subdivision=None):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/arrangement"

        if len(piano_pianoroll.size()) == 1:
            piano_pianoroll = torch.unsqueeze(piano_pianoroll, dim=0)
            orchestra_pianoroll = torch.unsqueeze(orchestra_pianoroll, dim=0)

        num_batches = len(piano_pianoroll)

        for batch_ind in range(num_batches):
            piano_part = self.piano_tensor_to_score(piano_pianoroll[batch_ind], durations_piano,
                                                    writing_tempo=writing_tempo,
                                                    subdivision=subdivision)
            orchestra_stream = self.orchestra_tensor_to_score(orchestra_pianoroll[batch_ind], durations_piano,
                                                              writing_tempo=writing_tempo,
                                                              subdivision=subdivision)

            piano_part.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_piano.mid", fmt='midi')
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_orchestra.mid", fmt='midi')
            # Both in the same score
            orchestra_stream.append(piano_part)
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_both.mid", fmt='midi')


