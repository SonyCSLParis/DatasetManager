import csv
import json
import math
import os
import re
import shutil

import torch
from abc import ABC

from arrangement.instrumentation import get_instrumentation
from torch.utils.data import TensorDataset
from tqdm import tqdm
import music21
import numpy as np

from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align

from arrangement.arrangement_helper import note_to_midiPitch, score_to_pianoroll, pianoroll_to_orchestral_tensor, \
    midiPitch_to_octave_pc, midiPitch_to_note, octave_pc_to_midiPitch


class ArrangementFrameDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 metadatas=[],
                 subdivision=4,
                 transpose_to_sounding_pitch=True,
                 reference_tessitura_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/reference_tessitura.json',
                 observed_tessitura_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/statistics/statistics.csv',
                 simplify_instrumentation_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/simplify_instrumentation.json',
                 cache_dir=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(ArrangementFrameDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.metadatas = metadatas
        self.subdivision = subdivision  # We use only on beats notes so far

        # Reference tessitura for each instrument
        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = {k: (music21.note.Note(v[0]), music21.note.Note(v[1])) for k, v in tessitura.items()}

        # Observed tessitura
        with open(observed_tessitura_path, 'r') as ff:
            dict_reader = csv.DictReader(ff, delimiter=';')
            for row in dict_reader:
                if row['instrument_name'] == 'Piano':
                    self.observed_piano_tessitura = {'lowest': int(row['lowest_pitch']),
                                                     'highest': int(row['highest_pitch'])}
                    break

        # Maps parts name found in mxml files to standard names
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)

        #  Instrumentation used for learning
        self.instrumentation = get_instrumentation()

        #  Do we use written or sounding pitch
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch

        # Mapping between instruments and indices
        self.piano_tessitura = {}
        self.index2instrument = {}
        self.instrument2index = {}
        self.indices2midi_pitch = {}
        self.midi_pitch2indices = {}
        self.one_hot_structure = {}

        # Compute some dimensions for the returned tensors
        self.number_parts = 0
        for k, v in self.instrumentation.items():
            self.number_parts += v
        self.number_pitch_class = 12
        self.number_octave = None
        return

    def __repr__(self):
        return f'ArrangementFrameDataset(' \
            f'{self.name},' \
            f'{[metadata.name for metadata in self.metadatas]},' \
            f'{self.subdivision})'

    def iterator_gen(self):
        return (self.sort_arrangement_pairs(arrangement_pair)
                for arrangement_pair in self.corpus_it_gen())

    @staticmethod
    def pair2index(one_hot_0, one_hot_1):
        return one_hot_0 * 12 + one_hot_1

    def compute_index_dicts(self):
        # Mapping piano notes <-> indices
        self.piano_tessitura['lowest_pitch'] = self.observed_piano_tessitura['lowest']
        self.piano_tessitura['highest_pitch'] = self.observed_piano_tessitura['highest']

        # Mapping instruments <-> indices
        index_counter = 0
        for instrument_name, number_instruments in self.instrumentation.items():
            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments

        #  Mapping midi_pitch to one-hot octave,pc,silence indices
        # Get lowest and highest octaves
        # USE REFERENCE OR OBSERVED ?
        lowest_pitch = math.inf
        highest_pitch = 0
        for instrument_name, this_tessitura in self.reference_tessitura.items():
            lowest_pitch = min(lowest_pitch, note_to_midiPitch(this_tessitura[0]))
            highest_pitch = max(highest_pitch, note_to_midiPitch(this_tessitura[1]))
        # Take the closest multiple of 12
        lowest_pitch = lowest_pitch - (lowest_pitch % 12)
        highest_pitch = highest_pitch + (12 - highest_pitch) % 12
        lowest_octave, _ = midiPitch_to_octave_pc(lowest_pitch)
        highest_octave, _ = midiPitch_to_octave_pc(highest_pitch)
        self.number_octave = highest_octave - lowest_octave + 1
        for midi_pitch in range(lowest_pitch, highest_pitch):
            octave, pc = midiPitch_to_octave_pc(midi_pitch)
            relative_octave = octave - lowest_octave
            self.midi_pitch2indices[midi_pitch] = (relative_octave, self.number_octave + pc)
            if relative_octave not in self.indices2midi_pitch.keys():
                self.indices2midi_pitch[relative_octave] = {}
            self.indices2midi_pitch[relative_octave][pc] = midi_pitch

        # Save structure, for easier reconstruction and manipulation
        start_ind = 0
        end_ind = self.number_octave
        self.one_hot_structure = {'octave': (start_ind, end_ind)}
        start_ind = end_ind
        end_ind = start_ind + self.number_pitch_class
        self.one_hot_structure['pitch_class'] = (start_ind, end_ind)
        start_ind = end_ind
        end_ind = start_ind
        self.one_hot_structure['silence'] = (start_ind, end_ind)
        self.one_hot_structure['encoding_size'] = end_ind + 1
        return

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        self.compute_index_dicts()

        # one_tick = 1 / self.subdivision
        piano_tensor_dataset = []
        orchestra_tensor_dataset = []
        # metadata_tensor_dataset = []

        total_frames_counter = 0
        missed_frames_counter = 0

        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            # Alignement
            corresponding_offsets = self.align_score(arr_pair['Piano'], arr_pair['Orchestra'])

            # Compute pianoroll representations of score (more efficient than manipulating the music21 streams)
            pianoroll_piano = score_to_pianoroll(arr_pair['Piano'], self.subdivision, self.simplify_instrumentation,
                                                 self.transpose_to_sounding_pitch)
            pianoroll_orchestra = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                     self.simplify_instrumentation, self.transpose_to_sounding_pitch)

            # main loop
            for (offset_piano, offset_orchestra) in corresponding_offsets:
                #################################################
                # NO TRANSPOSITIONS ALLOWED FOR NOW
                # Only consider orchestra to compute the possible transpositions
                # BUT IF WE ALLOW THEM, WE SHOULD
                # 1/ COMPUTE ALL TRANSPOSISITONS IN -3 +3 RANGE
                #  2/ SELECT FRAMES WHERE IT'S POSSIBLE
                # transp_minus, transp_plus = self.possible_transposition(arr_pair['Orchestra'], offset=offset_orchestra)
                # for semi_tone in range(transp_minus, transp_plus + 1):
                #################################################

                total_frames_counter += 1

                #  Piano
                assert (pianoroll_piano.keys() != 'Piano'), 'More than one instrument in the piano score'
                piano_np = pianoroll_piano['Piano'][int(offset_piano * self.subdivision)]
                # Squeeze to observed tessitura
                piano_np_reduced = piano_np[
                                   self.piano_tessitura['lowest_pitch']: self.piano_tessitura['highest_pitch'] + 1]
                local_piano_tensor = torch.from_numpy(piano_np_reduced)

                # Orchestra
                orchestral_vector_shape = (self.number_parts, self.one_hot_structure["encoding_size"])
                local_orchestra_tensor = pianoroll_to_orchestral_tensor(pianoroll_orchestra,
                                                                        int(offset_orchestra * self.subdivision),
                                                                        self.instrument2index,
                                                                        self.midi_pitch2indices,
                                                                        self.one_hot_structure,
                                                                        orchestral_vector_shape)

                if local_orchestra_tensor is None:
                    missed_frames_counter += 1
                    continue

                # append and add batch dimension
                # cast to int
                piano_tensor_dataset.append(
                    local_piano_tensor[None, :].int())
                orchestra_tensor_dataset.append(
                    local_orchestra_tensor[None, :, :].int())
                # metadata_tensor_dataset.append(
                #     local_metadata_tensor[None, :, :, :].int())

        piano_tensor_dataset = torch.cat(piano_tensor_dataset, 0)
        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)
        # metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)

        #######################
        # Check all vectors respect data format
        num_batch, instru_dim, notes_dim = orchestra_tensor_dataset.shape
        integrity_mask = torch.from_numpy(
            np.asarray([1, ] * self.number_octave + [2, ] * self.number_pitch_class + [4, ])).int()
        orchestra_tensor_dataset_reshaped = orchestra_tensor_dataset.view(num_batch * instru_dim, notes_dim)
        checked_sum = (torch.mv(orchestra_tensor_dataset_reshaped, integrity_mask)).numpy()
        #  Correct values are either
        # 3: note played
        # 7: silence
        assert np.all((checked_sum == 3) + (checked_sum == 7)).astype(bool), print()
        #######################

        dataset = TensorDataset(piano_tensor_dataset,
                                orchestra_tensor_dataset)
        #                       metadata_tensor_dataset)

        print(
            # f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n {orchestra_tensor_dataset.size()}\n {metadata_tensor_dataset.size()}\n')
            f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n {orchestra_tensor_dataset.size()}\n')
        print(f'Missed frames ratio: {missed_frames_counter / total_frames_counter}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, score):
        # Need only the flatten orchestra for aligning
        score_flat = score.flat
        # Useful for avoiding zeros at the start/end of file?
        start_offset = score_flat.lowestOffset
        end_offset = score_flat.highestOffset
        # Take only notes on beat
        frames = [(off, [pc.pitch.pitchClass if pc.isNote else pc.pitchClasses
                         for pc in score_flat.getElementsByOffset(off, mustBeginInSpan=False,
                                                                  classList=[music21.note.Note,
                                                                             music21.chord.Chord]).notes])
                  for off in np.arange(start_offset, end_offset + 1, 1 / self.subdivision)
                  ]
        # Flatten and remove silence frames
        list_pc = []
        for (off, elem) in frames:
            elem_flat = []
            for e in elem:
                if type(e) == list:
                    elem_flat.extend(e)
                else:
                    elem_flat.append(e)
            if len(elem_flat) > 0:
                list_pc.append((off, set(elem_flat)))

        return list_pc

    def align_score(self, piano_score, orchestra_score):
        list_pc_piano = self.score_to_list_pc(piano_score)
        list_pc_orchestra = self.score_to_list_pc(orchestra_score)

        only_pc_piano = [e[1] for e in list_pc_piano]
        only_pc_orchestra = [e[1] for e in list_pc_orchestra]

        print("aligning...")
        corresponding_indices = nw_align.nwalign(only_pc_piano, only_pc_orchestra, gapOpen=-3, gapExtend=-1)
        print("aligned")

        corresponding_offsets = [(list_pc_piano[ind_piano][0], list_pc_orchestra[ind_orchestra][0])
                                 for ind_piano, ind_orchestra in corresponding_indices]

        return corresponding_offsets

    def possible_transposition(self, arr_orch, offset):
        """
        returns None if no note present in one of the voices -> no transposition
        :param arr_orch:
        :param offset:
        :return:
        """
        transp_minus = -self.transposition_max_allowed
        transp_plus = self.transposition_max_allowed
        print(f"## {offset}")
        for part in arr_orch.parts:
            this_part_instrument_name = part.getInstrument().instrumentName

            voice_pitches = self.voice_range_in_part(part, offset=offset)

            print(f"{part}: {voice_pitches}")

            if voice_pitches is None:
                continue
            voice_lowest, voice_highest = voice_pitches
            # Get lowest/highest allowed transpositions
            down_interval = music21.interval.notesToChromatic(voice_lowest,
                                                              self.reference_tessitura[this_part_instrument_name][0])
            up_interval = music21.interval.notesToChromatic(voice_highest,
                                                            self.reference_tessitura[this_part_instrument_name][1])
            transp_minus = max(transp_minus, down_interval.semitones)
            transp_plus = min(transp_plus, up_interval.semitones)

        transp_minus = music21.interval.ChromaticInterval(min(transp_minus, 0))
        transp_plus = music21.interval.ChromaticInterval(max(transp_plus, 0))

        return transp_minus, transp_plus

    def voice_range_in_part(self, part, offset):
        """
        return the min and max pitches of an frame
        :param part: music21 part
        :param offset: offset at which ambitus is measured
        :return: pair of music21 notes

        """
        notes_and_chords = part.flat.getElementsByOffset(offset, mustBeginInSpan=False,
                                                         classList=[music21.note.Note,
                                                                    music21.chord.Chord])
        # Get list of notes
        notes_list = [
            n.pitch if n.isNote else n.pitches
            for n in notes_and_chords
        ]
        # Return lowest and highest
        if len(notes_list) > 0:
            return min(notes_list), max(notes_list)
        else:
            return None

    def sort_arrangement_pairs(self, arrangement_pair):
        # Find which score is piano and which is orchestral
        if len(self.list_instru_score(arrangement_pair[0])) > len(self.list_instru_score(arrangement_pair[1])):
            return {'Orchestra': arrangement_pair[0], 'Piano': arrangement_pair[1]}
        elif len(self.list_instru_score(arrangement_pair[0])) < len(self.list_instru_score(arrangement_pair[1])):
            return {'Piano': arrangement_pair[0], 'Orchestra': arrangement_pair[1]}
        else:
            raise Exception('The two scores have the same number of instruments')

    def list_instru_score(self, score):
        list_instru = []
        for part in score.parts:
            list_instru.append(part.partName)
        return list_instru

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    def piano_tensor_to_score(self, tensor_score):
        piano_vector = tensor_score.numpy()
        piano_notes = np.where(piano_vector > 0)
        note_list = []
        for piano_note in piano_notes[0]:
            pitch = self.piano_tessitura['lowest_pitch'] + piano_note
            velocity = piano_vector[piano_note]
            duration = 1.0
            f = music21.note.Note(pitch)
            f.volume.velocity = velocity
            note_list.append(f)
        chord = music21.chord.Chord(note_list)
        return chord

    def orchestra_tensor_to_score(self, tensor_score):

        # First store everything in a dict, then each instrument will be a part in the final stream
        score_dict = {}

        orchestra_matrix = tensor_score.numpy()

        for instrument_index in range(self.number_parts):

            # Get instrument name
            instrument_name = self.index2instrument[instrument_index]

            # One hot vectors stacked
            octave_indices = self.one_hot_structure["octave"]
            pitch_class_indices = self.one_hot_structure["pitch_class"]
            silence_indices = self.one_hot_structure["silence"]

            #  Convert to
            relative_octave = orchestra_matrix[instrument_index, octave_indices[0]:octave_indices[1]]
            pitch_class = orchestra_matrix[instrument_index, pitch_class_indices[0]:pitch_class_indices[1]]
            is_silence = orchestra_matrix[instrument_index, silence_indices[0]]

            if is_silence == 0:
                # Sanity check one-hot encoding before argmax
                assert (relative_octave.max() == 1) and (relative_octave.sum() == 1)
                assert (pitch_class.max() == 1) and (pitch_class.sum() == 1)
                relative_octave = np.argmax(relative_octave)
                pitch_class = np.argmax(pitch_class)
                # A bit heavy but super safe
                midi_pitch = self.indices2midi_pitch[relative_octave][pitch_class]
                if instrument_name not in score_dict.keys():
                    score_dict[instrument_name] = []
                score_dict[instrument_name].append(midi_pitch)

        stream = music21.stream.Stream()
        for (instrument_name, notes) in score_dict.items():
            this_part = music21.stream.Part(id=instrument_name)
            # re is for removing underscores in instrument names which raise errors in music21
            music21_instrument = music21.instrument.fromString(re.sub('_', ' ', instrument_name))
            this_part.insert(music21_instrument)

            list_notes = []
            for midi_pitch in notes:
                # Single note
                # this_note = midiPitch_to_note(midi_pitch)
                this_note = music21.note.Note(midi_pitch)
                list_notes.append(this_note)

            this_chord = music21.chord.Chord(list_notes)
            this_part.append(this_chord)
            this_part.atSoundingPitch = self.transpose_to_sounding_pitch
            # if self.transpose_to_sounding_pitch:
            #     transposition = music21_instrument.transposition
            #     this_part.
            stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path='/home/leo/Recherche/databases/Orchestration/arrangement_mxml',
        subsets=[
            # 'bouliane',
            # 'imslp',
            # 'liszt_classical_archives',
            # 'hand_picked_Spotify',
            'debug'
        ],
        num_elements=10
    )

    kwargs = {}
    kwargs.update(
        {'name': "arrangement_frame_test",
         'corpus_it_gen': corpus_it_gen,
         'cache_dir': '/home/leo/Recherche/DatasetManager/DatasetManager/dataset_cache'
         })

    dataset = ArrangementFrameDataset(**kwargs)
    # if os.path.exists(dataset.filepath):
    #     print(f'Loading {dataset.__repr__()} from {dataset.filepath}')
    #     dataset = torch.load(dataset.filepath)
    #     print(f'(the corresponding TensorDataset is not loaded)')
    # else:
    print(f'Creating {dataset.__repr__()}, '
          f'both tensor dataset and parameters')
    # initialize and force the computation of the tensor_dataset
    # first remove the cached data if it exists
    if os.path.exists(dataset.tensor_dataset_filepath):
        os.remove(dataset.tensor_dataset_filepath)
    # recompute dataset parameters and tensor_dataset
    # this saves the tensor_dataset in dataset.tensor_dataset_filepath
    tensor_dataset = dataset.tensor_dataset

    # Data loaders
    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(
        batch_size=16,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=False
    )

    # Visualise a few examples
    writing_dir = f"{os.getcwd()}/dump"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    i_example = 0
    for sample_batched in train_dataloader:
        piano_batch, orchestra_batch = sample_batched
        if i_example > 40:
            break
        for piano_vector, orchestra_vector in zip(piano_batch, orchestra_batch):
            if i_example > 40:
                break
            piano_stream = dataset.piano_tensor_to_score(piano_vector)
            piano_stream.write(fp=f"{writing_dir}/{i_example}_piano.xml", fmt='musicxml')
            orchestra_stream = dataset.orchestra_tensor_to_score(orchestra_vector)
            orchestra_stream.write(fp=f"{writing_dir}/{i_example}_orchestra.xml", fmt='musicxml')
            i_example += 1
