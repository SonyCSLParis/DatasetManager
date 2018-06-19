from fractions import Fraction

import music21
import re
import os

import torch
from bson import ObjectId

import numpy as np
from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, standard_name, \
    standard_note, PAD_SYMBOL
from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import altered_pitches_music21_to_dict, REST, \
    getUnalteredPitch, getAccidental, getOctave, note_duration, \
    is_tied_left, general_note, FakeNote, assert_no_time_signature_changes, NC, set_metadata, \
    notes_and_chords, \
    leadsheet_on_ticks, standard_chord
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.lsdb.lsdb_exceptions import *
from torch.utils.data import TensorDataset
from tqdm import tqdm


class LsdbDataset(MusicDataset):
    def __init__(self, corpus_it_gen,
                 name,
                 sequences_size,
                 cache_dir):
        """

        :param corpus_it_gn:
        :param sequences_size: in beats
        """
        super(LsdbDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.tick_values = [0,
                            Fraction(1, 4),
                            Fraction(1, 3),
                            Fraction(1, 2),
                            Fraction(2, 3),
                            Fraction(3, 4)]
        self.tick_durations = self.compute_tick_durations()
        self.number_of_beats = 4
        self.num_voices = 2
        self.NOTES = 0
        self.CHORDS = 1
        self.leadsheet_iterator_gen = corpus_it_gen
        self.sequences_size = sequences_size
        self.subdivision = len(self.tick_values)
        self.pitch_range = [55, 84]
        self.init_index_dicts()

    def __repr__(self):
        # TODO
        return f'LsdbDataset(' \
               f'{self.name},' \
               f'{self.sequences_size})'

    def compute_tick_durations(self):
        diff = [n - p
                for n, p in zip(self.tick_values[1:], self.tick_values[:-1])]
        diff = diff + [1 - self.tick_values[-1]]
        return diff

    def transpose_leadsheet(self,
                            leadsheet: music21.stream.Score,
                            interval: music21.interval.Interval
                            ):
        try:
            leadsheet_transposed = leadsheet.transpose(interval)
        except ValueError as e:
            raise LeadsheetParsingException(f'Leadsheet {leadsheet.metadata.title} '
                                            f'not properly formatted')
        return leadsheet_transposed

    def lead_and_chord_tensors(self, leadsheet, update_dicts=False):
        """

        :param leadsheet:
        :return: lead_tensor and chord_tensor
        """
        eps = 1e-4
        notes, chords = notes_and_chords(leadsheet)
        if not leadsheet_on_ticks(leadsheet, self.tick_values):
            raise LeadsheetParsingException(
                f'Leadsheet {leadsheet.metadata.title} has notes not on ticks')

        # LEAD
        j = 0
        i = 0
        length = int(leadsheet.highestTime * self.subdivision)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0

        note2index = self.symbol2index_dicts[self.NOTES]
        index2note = self.index2symbol_dicts[self.NOTES]
        while i < length:
            # update dicts when creating the dataset
            note_name = standard_name(notes[j])
            if update_dicts and note_name not in note2index:
                new_index = len(note2index)
                note2index[note_name] = new_index
                index2note[new_index] = note_name

            note_index = note2index[note_name]
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note_index,
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(self.tick_values)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note_index,
                           is_articulated]
                i += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        lead_tensor = torch.from_numpy(lead).long()[None, :]

        # CHORDS
        j = 0
        i = 0
        length = int(leadsheet.highestTime)
        t = np.zeros((length, 2))
        is_articulated = True
        num_chords = len(chords)
        chord2index = self.symbol2index_dicts[self.CHORDS]
        index2chord = self.index2symbol_dicts[self.CHORDS]
        while i < length:
            # update dicts when creating the dataset
            chord_name = standard_name(chords[j])
            if update_dicts and chord_name not in chord2index:
                new_index = len(chord2index)
                chord2index[chord_name] = new_index
                index2chord[new_index] = chord_name
            chord_index = chord2index[chord_name]
            if j < num_chords - 1:
                if chords[j + 1].offset > i:
                    t[i, :] = [chord_index,
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [chord_index,
                           is_articulated]
                i += 1
                is_articulated = False
        seq = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * chord2index[SLUR_SYMBOL]
        chord_tensor = torch.from_numpy(seq).long()[None, :]
        return lead_tensor, chord_tensor

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        # todo check on chorale with Chord
        print('Making tensor dataset')
        # todo not useful?
        # self.compute_index_dicts()
        lead_tensor_dataset = []
        chord_tensor_dataset = []
        for leadsheet_id, leadsheet in tqdm(enumerate(self.leadsheet_iterator_gen())):
            print(leadsheet.metadata.title)
            if not self.is_valid(leadsheet):
                continue
            try:
                possible_transpositions = self.all_transposition_intervals(leadsheet)
                for transposition_interval in possible_transpositions:
                    transposed_leadsheet = self.transpose_leadsheet(
                        leadsheet,
                        transposition_interval)

                    lead_tensor, chord_tensor = self.lead_and_chord_tensors(
                        transposed_leadsheet, update_dicts=True)
                    # lead
                    for offsetStart in range(-self.sequences_size + 1,
                                             int(transposed_leadsheet.highestTime)):
                        offsetEnd = offsetStart + self.sequences_size
                        local_lead_tensor = self.extract_with_padding(
                            tensor=lead_tensor,
                            start_tick=offsetStart * self.subdivision,
                            end_tick=offsetEnd * self.subdivision,
                            symbol2index=self.symbol2index_dicts[self.NOTES]
                        )
                        local_chord_tensor = self.extract_with_padding(
                            tensor=chord_tensor,
                            start_tick=offsetStart,
                            end_tick=offsetEnd,
                            symbol2index=self.symbol2index_dicts[self.CHORDS]
                        )

                        # append and add batch dimension
                        # cast to int
                        lead_tensor_dataset.append(
                            local_lead_tensor.int())
                        chord_tensor_dataset.append(
                            local_chord_tensor.int())
            except LeadsheetParsingException as e:
                print(e)

        lead_tensor_dataset = torch.cat(lead_tensor_dataset, 0)
        chord_tensor_dataset = torch.cat(chord_tensor_dataset, 0)

        dataset = TensorDataset(lead_tensor_dataset,
                                chord_tensor_dataset)

        print(f'Sizes: {lead_tensor_dataset.size()},'
              f' {chord_tensor_dataset.size()}')
        return dataset

    def contains_notes_and_chords(self, leadsheet):
        notes_and_rests, chords = notes_and_chords(leadsheet)
        notes = [n.pitch.midi for n in notes_and_rests if n.isNote]
        return len(notes) > 0 and len(chords) > 0

    def all_transposition_intervals(self, leadsheet):
        min_pitch, max_pitch = self.leadsheet_range(leadsheet)
        min_pitch_corpus, max_pitch_corpus = self.pitch_range

        min_transposition = min_pitch - min_pitch_corpus
        max_transposition = max_pitch_corpus - max_pitch

        transpositions = []
        for semi_tone in range(min_transposition, max_transposition + 1):
            interval_type, interval_nature = music21.interval.convertSemitoneToSpecifierGeneric(
                semi_tone)
            transposition_interval = music21.interval.Interval(
                str(interval_nature) + interval_type)
            transpositions.append(transposition_interval)

        return transpositions

    def extract_with_padding(self, tensor,
                             start_tick,
                             end_tick,
                             symbol2index):
        """

        :param tensor: (batch_size, length)
        :param start_tick:
        :param end_tick:
        :param symbol2index:
        :return: (batch_size, end_tick - start_tick)
        """
        assert start_tick < end_tick
        assert end_tick > 0
        batch_size, length = tensor.size()

        padded_tensor = []
        if start_tick < 0:
            start_symbols = np.array([symbol2index[PAD_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            start_symbols[:, -1] = symbol2index[START_SYMBOL]
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[PAD_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            end_symbols[:, 0] = symbol2index[END_SYMBOL]
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    def is_lead(self, voice_id):
        return voice_id == self.NOTES

    def is_chord(self, voice_id):
        return voice_id == self.CHORDS

    def init_index_dicts(self):
        print('Initialize index_dicts')
        self.index2symbol_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.symbol2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            note_set.add(PAD_SYMBOL)

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2symbol_dicts,
                                                    self.symbol2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

    # Unused
    # def compute_index_dicts(self):
    #     print('Computing index dicts')
    #     self.index2symbol_dicts = [
    #         {} for _ in range(self.num_voices)
    #     ]
    #     self.symbol2index_dicts = [
    #         {} for _ in range(self.num_voices)
    #     ]
    #
    #     # create and add additional symbols
    #     note_sets = [set() for _ in range(self.num_voices)]
    #     for note_set in note_sets:
    #         note_set.add(SLUR_SYMBOL)
    #         note_set.add(START_SYMBOL)
    #         note_set.add(END_SYMBOL)
    #         note_set.add(PAD_SYMBOL)
    #
    #     # get all notes
    #     for leadsheet in tqdm(self.leadsheet_iterator_gen()):
    #         if self.is_in_range(leadsheet):
    #             # part is either lead or chords as lists
    #             for part_id, part in enumerate(notes_and_chords(leadsheet)):
    #                 for n in part:
    #                     note_sets[part_id].add(standard_name(n))
    #
    #     # create tables
    #     for note_set, index2note, note2index in zip(note_sets,
    #                                                 self.index2symbol_dicts,
    #                                                 self.symbol2index_dicts):
    #         for note_index, note in enumerate(note_set):
    #             index2note.update({note_index: note})
    #             note2index.update({note: note_index})

    #
    #
    # def compute_lsdb_chord_dicts(self):
    #     # TODO must be created from xml folder
    #     # Search LSDB for chord names
    #     with LsdbMongo() as mongo_client:
    #         db = mongo_client.get_db()
    #         modes = db.modes
    #         cursor_modes = modes.find({})
    #         chord2notes = {}  # Chord to notes dictionary
    #         notes2chord = {}  # Notes to chord dictionary
    #         for chord in cursor_modes:
    #             notes = []
    #             # Remove white spaces from notes string
    #             for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
    #                 notes.append(note)
    #             notes = tuple(notes)
    #
    #             # Enter entries in dictionaries
    #             chord2notes[chord['mode']] = notes
    #             if notes in notes2chord:
    #                 notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
    #             else:
    #                 notes2chord[notes] = [chord["mode"]]
    #
    #         self.correct_chord_dicts(chord2notes, notes2chord)
    #
    #         return chord2notes, notes2chord
    #
    # def correct_chord_dicts(self, chord2notes, notes2chord):
    #     """
    #     Modifies chord2notes and notes2chord in place
    #     to correct errors in LSDB modes (dict of chord symbols with notes)
    #     :param chord2notes:
    #     :param notes2chord:
    #     :return:
    #     """
    #     # Add missing chords
    #     # b5
    #     notes2chord[('C4', 'E4', 'Gb4')] = notes2chord[('C4', 'E4', 'Gb4')] + ['b5']
    #     chord2notes['b5'] = ('C4', 'E4', 'Gb4')
    #     # b9#5
    #     notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'D#5')] = 'b9#b'
    #     chord2notes['b9#5'] = ('C4', 'E4', 'G#4', 'Bb4', 'D#5')
    #     # 7#5#11 is WRONG in the database
    #     # C4 F4 G#4 B-4 D5 instead  of C4 E4 G#4 B-4 D5
    #     notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'F#5')] = '7#5#11'
    #     chord2notes['7#5#11'] = ('C4', 'E4', 'G#4', 'Bb4', 'F#5')
    #
    # # F#7#9#11 is WRONG in the database

    def test(self):
        with LsdbMongo() as client:
            db = client.get_db()
            leadsheets = db.leadsheets.find(
                {'_id': ObjectId('5193841a58e3383974000079')})
            leadsheet = next(leadsheets)
            print(leadsheet['title'])
            score = self.leadsheet_to_music21(leadsheet)
            score.show()

    def is_in_range(self, leadsheet):
        min_pitch, max_pitch = self.leadsheet_range(leadsheet)
        return (min_pitch >= self.pitch_range[0]
                and max_pitch <= self.pitch_range[1])

    def is_valid(self, leadsheet):
        return (self.contains_notes_and_chords(leadsheet)
                and
                self.is_in_range(leadsheet)
                )

    def leadsheet_range(self, leadsheet):
        notes, chords = notes_and_chords(leadsheet)
        pitches = [n.pitch.midi for n in notes if n.isNote]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        return min_pitch, max_pitch

    def random_leadsheet_tensor(self, sequence_length):
        lead_tensor = np.random.randint(len(self.symbol2index_dicts[self.NOTES]),
                                        size=sequence_length * self.subdivision)
        chords_tensor = np.random.randint(len(self.symbol2index_dicts[self.CHORDS]),
                                          size=sequence_length)
        lead_tensor = torch.from_numpy(lead_tensor).long()
        chords_tensor = torch.from_numpy(chords_tensor).long()

        return lead_tensor, chords_tensor

    def tensor_leadsheet_to_score(self, tensor_lead, tensor_chords,
                                  realize_chords=False):
        """
        Converts leadsheet given as tensor_lead and tensor_chords
        to a true music21 score
        :param tensor_lead:
        :param tensor_chords:
        :return:
        """
        slur_index = self.symbol2index_dicts[self.NOTES][SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()

        # LEAD
        dur = 0
        f = music21.note.Rest()
        for tick_index, note_index in enumerate(tensor_lead):
            note_index = note_index.item()
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)
                # TODO two types of tick_durations
                dur = self.tick_durations[tick_index % self.subdivision]
                f = standard_note(self.index2symbol_dicts[self.NOTES][note_index])
            else:
                dur += self.tick_durations[tick_index % self.subdivision]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)

        # CHORD SYMBOLS
        slur_index = self.symbol2index_dicts[self.CHORDS][SLUR_SYMBOL]
        index2chord = self.index2symbol_dicts[self.CHORDS]
        chord2index = self.symbol2index_dicts[self.CHORDS]
        start_index = chord2index[START_SYMBOL]
        end_index = chord2index[END_SYMBOL]
        for beat_index, chord_index in enumerate(tensor_chords):
            chord_index = chord_index.item()
            # if it is a played chord
            if chord_index not in [slur_index, start_index, end_index]:
                # add chord
                part.insert(beat_index, standard_chord(index2chord[chord_index]))
        score.insert(part)

        if realize_chords:
            slur_index = self.symbol2index_dicts[self.CHORDS][SLUR_SYMBOL]
            index2chord = self.index2symbol_dicts[self.CHORDS]
            chord2index = self.symbol2index_dicts[self.CHORDS]
            chords_part = music21.stream.Part()
            dur = 0
            c = music21.note.Rest()
            for beat_index, chord_index in enumerate(tensor_chords):
                chord_index = chord_index.item()

                # if it is a played note
                if not chord_index == slur_index:
                    # add previous note
                    if dur > 0:
                        c.duration = music21.duration.Duration(dur)
                        chords_part.append(c)
                    dur = 1
                    try:
                        c = music21.chord.Chord([
                            p.transpose(12) for p in
                            standard_chord(index2chord[
                                               chord_index]).pitches])
                    except:
                        c = music21.note.Rest()
                else:
                    dur += 1
            # add last note
            c.duration = music21.duration.Duration(dur)
            chords_part.append(c)
            score.append(chords_part)

        return score

    def tensor_leadsheet_to_score_and_chord_list(self,
                                                 tensor_lead,
                                                 tensor_chords,
                                                 add_chord_symbols=True,
                                                 realize_chords=False):
        """
        Converts leadsheet given as tensor_lead to a true music21 score
        and the chords as a list
        :param tensor_lead:
        :param tensor_chords:
        :return:
        """
        slur_index = self.symbol2index_dicts[self.NOTES][SLUR_SYMBOL]

        score = music21.stream.Score()
        lead_part = music21.stream.Part()

        # LEAD PART
        dur = 0
        f = music21.note.Rest()
        for tick_index, note_index in enumerate(tensor_lead):
            note_index = note_index.item()
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    lead_part.append(f)
                # TODO two types of tick_durations
                dur = self.tick_durations[tick_index % self.subdivision]
                f = standard_note(self.index2symbol_dicts[self.NOTES][note_index])
            else:
                dur += self.tick_durations[tick_index % self.subdivision]
        # add last note
        f.duration = music21.duration.Duration(dur)
        lead_part.append(f)

        # CHORD SYMBOLS (in lead_part)
        if add_chord_symbols:
            slur_index = self.symbol2index_dicts[self.CHORDS][SLUR_SYMBOL]
            index2chord = self.index2symbol_dicts[self.CHORDS]
            chord2index = self.symbol2index_dicts[self.CHORDS]
            start_index = chord2index[START_SYMBOL]
            end_index = chord2index[END_SYMBOL]
            for beat_index, chord_index in enumerate(tensor_chords):
                chord_index = chord_index.item()
                # if it is a played chord
                if chord_index not in [slur_index, start_index, end_index]:
                    # add chord
                    lead_part.insert(beat_index,
                                     standard_chord(index2chord[chord_index]))

        score.append(lead_part)

        # REALIZED CHORD PART (in another part)
        if realize_chords:
            slur_index = self.symbol2index_dicts[self.CHORDS][SLUR_SYMBOL]
            index2chord = self.index2symbol_dicts[self.CHORDS]
            chord2index = self.symbol2index_dicts[self.CHORDS]
            chords_part = music21.stream.Part()
            dur = 0
            c = music21.note.Rest()
            for beat_index, chord_index in enumerate(tensor_chords):
                chord_index = chord_index.item()

                # if it is a played note
                if not chord_index == slur_index:
                    # add previous note
                    if dur > 0:
                        c.duration = music21.duration.Duration(dur)
                        chords_part.append(c)
                    dur = 1
                    try:
                        c = music21.chord.Chord([
                            p.transpose(12) for p in
                            standard_chord(
                                index2chord[chord_index]).pitches])
                    except:
                        c = music21.note.Rest()
                else:
                    dur += 1
            # add last note
            c.duration = music21.duration.Duration(dur)
            chords_part.append(c)
            score.append(chords_part)

        # for beat_index, chord_index in enumerate(tensor_chords):
        #     slur_index = self.symbol2index_dicts[self.CHORDS][SLUR_SYMBOL]
        #     if not chord_index == slur_index:
        #         chord_str = self.index2symbol_dicts[self.CHORDS][chord_index]
        #         # reduce the number of characters in the string until it
        #         # is parsed
        #         # if correct
        #         # TODO fix this
        #         num_chars = len(chord_str)
        #         while True:
        #             try:
        #                 chord = standard_chord_symbol(chord_str[:num_chars])
        #                 if chord is not None:
        #                     # None is returned if there is a start or end symbol
        #                     part.insert(beat_index, chord)
        #                 break
        #             except ValueError:
        #                 print(f'TO FIX: Chord {chord_str} is not parsable')
        #                 num_chars -= 1

        # part = part.makeMeasures(
        #     inPlace=False,
        #     refStreamOrTimeRange=[0.0, tensor_chords.size(0)]
        # )
        #
        # # add treble clef and key signature
        # part.measure(1).clef = music21.clef.TrebleClef()
        # # part.measure(1).keySignature = key_signature
        # score.insert(part)
        # score.show('txt')

        # CHORDS LIST
        chord_list = []
        for beat_index, chord_index in enumerate(tensor_chords):
            chord_index = chord_index.item()
            # todo standardize
            chord_list.append(index2chord[chord_index])
        return score, chord_list


if __name__ == '__main__':
    from DatasetManager.dataset_manager import DatasetManager

    dataset_manager = DatasetManager()
    leadsheet_dataset_kwargs = {
        'sequences_size': 32,
    }

    bach_chorales_dataset: LsdbDataset = dataset_manager.get_dataset(
        name='lsdb',
        **leadsheet_dataset_kwargs
    )

    dl, _, _ = bach_chorales_dataset.data_loaders(1)
    tensor_lead, tensor_chord = next(dl.__iter__())
    score, chord_list = bach_chorales_dataset.tensor_leadsheet_to_score_and_chord_list(
        tensor_lead[0],
        tensor_chord[0])
    score.show()
    print(chord_list)
