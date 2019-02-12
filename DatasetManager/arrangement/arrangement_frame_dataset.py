import json
from abc import ABC

from tqdm import tqdm
import music21
import numpy as np

from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align


class ArrangementFrameDataset(MusicDataset, ABC):
    """
    Class for all arrangement dataset
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 metadatas=None,
                 subdivision=4,
                 tessitura_path='reference_tessitura.json',
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

        with open(tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.tessitura = {k: (music21.note.Note(v[0]), music21.note.Note(v[1])) for k, v in tessitura.items()}

        return

    def __repr__(self):
        return f'ArrangementFrameDataset(' \
            f'{self.name},' \
            f'{[metadata.name for metadata in self.metadatas]},' \
            f'{self.subdivision})'

    def iterator_gen(self):
        return (self.sort_arrangement_pairs(arrangement_pair)
                for arrangement_pair in self.corpus_it_gen())

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        # one_tick = 1 / self.subdivision
        # chorale_tensor_dataset = []
        # metadata_tensor_dataset = []

        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            # Alignement
            corresponding_offsets = self.align_score(arr_pair['Piano'], arr_pair['Orchestra'])

            # main loop
            for (offset_piano, offset_orchestra) in corresponding_offsets:
                # Only consider orchestra to compute the possible transpositions
                transp_minus, transp_plus = self.possible_transposition(arr_pair['Orchestra'], offset=offset_orchestra)

        #         for semi_tone in range(transp_minus, transp_plus + 1):
        #             try:
        #                 # get transposed tensor
        #                 (piano_tensor,
        #                  orchestra_tensor,
        #                  metadata_tensor) = (
        #                      self.transposed_score_and_metadata_tensors(
        #                          arr_pair,
        #                          semi_tone=semi_tone))
        #
        #                     piano_tensor = piano_transpositions[semi_tone]
        #                     orchestra_tensor = orchestra_transpositions[semi_tone]
        #                     metadata_tensor = metadatas_transpositions[semi_tone]
        #
        #                 local_piano_tensor = self.extract_piano_tensor(
        #                     piano_tensor,
        #                     frame_tick_piano)
        #                 local_orchestra_tensor = self.extract_orchestra_tensor(
        #                     orchestra_tensor,
        #                     frame_tick_orchestra)
        #
        #                 local_metadata_tensor = self.extract_metadata_with_padding(
        #                     metadata_tensor,
        #                     frame_tick)
        #
        #                 # append and add batch dimension
        #                 # cast to int
        #                 chorale_tensor_dataset.append(
        #                     local_chorale_tensor[None, :, :].int())
        #                 metadata_tensor_dataset.append(
        #                     local_metadata_tensor[None, :, :, :].int())
        #
        #             except KeyError:
        #                 # some problems may occur with the key analyzer
        #                 print(f'KeyError with chorale {chorale_id}')
        #
        # chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        # metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        #
        # dataset = TensorDataset(chorale_tensor_dataset,
        #                         metadata_tensor_dataset)
        #
        # print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        # return dataset

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return None

    def get_metadata_tensor(self, score):
        return None

    def get_score_tensor(self, score, offset, offsetEnd):
        return None

    # def part_to_tensor(self, part, part_id, offsetStart, offsetEnd):
    #     """
    #     :param part:
    #     :param part_id:
    #     :param offsetStart:
    #     :param offsetEnd:
    #     :return: torch IntTensor (1, length)
    #     """
    #     # TOUT REECRIRE EN UTILISANT PART_TO_LIST
    #
    #     list_notes_and_rests = list(part.flat.getElementsByOffset(
    #         offsetStart=offsetStart,
    #         offsetEnd=offsetEnd,
    #         classList=[music21.note.Note,
    #                    music21.note.Rest]))
    #     list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
    #                                      for n in list_notes_and_rests
    #                                      if n.isNote]
    #     length = int((offsetEnd - offsetStart) * self.subdivision)  # in ticks
    #
    #     # add entries to dictionaries if not present
    #     # should only be called by make_dataset when transposing
    #     note2index = self.note2index_dicts[part_id]
    #     index2note = self.index2note_dicts[part_id]
    #     voice_range = self.voice_ranges[part_id]
    #     min_pitch, max_pitch = voice_range
    #     for note_name, pitch in list_note_strings_and_pitches:
    #         # if out of range
    #         if pitch < min_pitch or pitch > max_pitch:
    #             note_name = OUT_OF_RANGE
    #
    #         if note_name not in note2index:
    #             new_index = len(note2index)
    #             index2note.update({new_index: note_name})
    #             note2index.update({note_name: new_index})
    #             print('Warning: Entry ' + str(
    #                 {new_index: note_name}) + ' added to dictionaries')
    #
    #     # construct sequence
    #     j = 0
    #     i = 0
    #     t = np.zeros((length, 2))
    #     is_articulated = True
    #     num_notes = len(list_notes_and_rests)
    #     while i < length:
    #         if j < num_notes - 1:
    #             if (list_notes_and_rests[j + 1].offset > i
    #                     / self.subdivision + offsetStart):
    #                 t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
    #                                                     voice_range=voice_range)],
    #                            is_articulated]
    #                 i += 1
    #                 is_articulated = False
    #             else:
    #                 j += 1
    #                 is_articulated = True
    #         else:
    #             t[i, :] = [note2index[standard_name(list_notes_and_rests[j],
    #                                                 voice_range=voice_range)],
    #                        is_articulated]
    #             i += 1
    #             is_articulated = False
    #     seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
    #     # todo padding
    #     tensor = torch.from_numpy(seq).long()[None, :]
    #     return tensor
    #
    # def get_score_dictList(score):
    #     dictList = {}
    #     for part_id, part in enumerate(score.parts):
    #         instru_name = part.getInstrument().instrumentName
    #         if instru_name in dictList.keys():
    #             raise Exception('Instrument present two times in the mxml file')
    #
    #         orchestra_notes = part.getElementsByOffset(
    #             offsetStart=part.lowestOffset,
    #             offsetEnd=part.highestOffset,
    #             classList=[music21.note.Note,
    #                        music21.chord.Chord])
    #
    #         score_tensor[instru_name] = part_tensor

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
                                                              self.tessitura[this_part_instrument_name][0])
            up_interval = music21.interval.notesToChromatic(voice_highest,
                                                              self.tessitura[this_part_instrument_name][1])
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

    def tensor_to_score(self, tensor_score):
        return None
