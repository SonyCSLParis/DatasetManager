import pickle
from fractions import Fraction

import music21
import re
import os

import torch
from bson import ObjectId

import numpy as np
from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, standard_name
from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import altered_pitches_music21_to_dict, REST, \
    getUnalteredPitch, getAccidental, getOctave, note_duration, \
    is_tied_left, general_note, FakeNote, assert_no_time_signature_changes, NC, \
    exclude_list_ids, set_metadata, notes_and_chords, LeadsheetIteratorGenerator, leadsheet_on_ticks
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.lsdb.lsdb_exceptions import *
from torch.utils.data import TensorDataset
from tqdm import tqdm


class LsdbDataset(MusicDataset):
    def __init__(self, corpus_it_gen,
                 name,
                 sequences_size):
        """

		:param corpus_it_gen:
		:param sequences_size: in beats
		"""
        super(LsdbDataset, self).__init__()
        self.name = name
        self.tick_values = [0,
                            Fraction(1, 4),
                            Fraction(1, 3),
                            Fraction(1, 2),
                            Fraction(2, 3),
                            Fraction(3, 4)]
        self.tick_durations = self.compute_tick_durations()
        self.number_of_beats = 4
        try:
            (self.lsdb_chord_to_notes,
             self.notes_to_chord_lsdb) = self.compute_chord_dicts()
        except (AttributeError, ValueError):
            print('Chords maps cannot be loaded')
            self.lsdb_chord_to_notes = None
            self.notes_to_chord_lsdb = None

        self.num_voices = 2
        self.NOTES = 0
        self.CHORDS = 1
        self.leadsheet_iterator_gen = corpus_it_gen
        self.sequences_size = sequences_size
        self.subdivision = len(self.tick_values)
        self.pitch_range = [55, 84]

    def __repr__(self):
        # TODO
        return f'LsdbDataset(' \
               f'{self.name})'

    def compute_tick_durations(self):
        diff = [n - p
                for n, p in zip(self.tick_values[1:], self.tick_values[:-1])]
        diff = diff + [1 - self.tick_values[-1]]
        return diff

    def lead_and_chord_tensors(self, leadsheet):
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
        while i < length:
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note2index[standard_name(notes[j])],
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(self.tick_values)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(notes[j])],
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
        while i < length:
            if j < num_chords - 1:
                if chords[j + 1].offset > i:
                    t[i, :] = [chord2index[standard_name(chords[j])],
                               is_articulated]
                    i += 1
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [chord2index[standard_name(chords[j])],
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
        self.compute_index_dicts()
        lead_tensor_dataset = []
        chord_tensor_dataset = []
        for leadsheet_id, leadsheet in tqdm(enumerate(self.leadsheet_iterator_gen())):
            # todo transpositions
            print(leadsheet.metadata.title)
            if not self.is_in_range(leadsheet):
                continue
            try:
                lead_tensor, chord_tensor = self.lead_and_chord_tensors(leadsheet)
                # lead
                for offsetStart in range(-self.sequences_size + 1,
                                         int(leadsheet.highestTime)):
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
                    lead_tensor_dataset.append(
                        local_lead_tensor)
                    chord_tensor_dataset.append(
                        local_chord_tensor)
            except LeadsheetParsingException as e:
                print(e)

        lead_tensor_dataset = torch.cat(lead_tensor_dataset, 0)
        chord_tensor_dataset = torch.cat(chord_tensor_dataset, 0)

        dataset = TensorDataset(lead_tensor_dataset,
                                chord_tensor_dataset)

        print(f'Sizes: {lead_tensor_dataset.size()},'
              f' {chord_tensor_dataset.size()}')
        return dataset

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
            start_symbols = np.array([symbol2index[START_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[END_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    def is_lead(self, voice_id):
        return voice_id == self.NOTES

    def is_chord(self, voice_id):
        return voice_id == self.CHORDS

    def compute_index_dicts(self):
        print('Computing index dicts')
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

        # get all notes
        # todo filter leadsheets not in voice range
        for leadsheet in tqdm(self.leadsheet_iterator_gen()):
            # part is either lead or chords as lists
            for part_id, part in enumerate(notes_and_chords(leadsheet)):
                for n in part:
                    note_sets[part_id].add(standard_name(n))

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2symbol_dicts,
                                                    self.symbol2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})

    def make_score_dataset(self):
        """
		Download all LSDB leadsheets, convert them into MusicXML and write them
		in xml folder

		:return:
		"""
        if not os.path.exists('xml'):
            os.mkdir('xml')

        # todo add query
        with LsdbMongo() as client:
            db = client.get_db()
            leadsheets = db.leadsheets.find({'_id': {
                '$nin': exclude_list_ids
            }})

            for leadsheet in leadsheets:
                # discard leadsheet with no title
                if 'title' not in leadsheet:
                    continue
                if os.path.exists(os.path.join('xml',
                                               f'{leadsheet["title"]}.xml'
                                               )):
                    print(leadsheet['title'])
                    print(leadsheet['_id'])
                    print('exists!')
                    continue
                print(leadsheet['title'])
                print(leadsheet['_id'])
                try:
                    score = self.leadsheet_to_music21(leadsheet)
                    export_file_name = os.path.join('xml',
                                                    f'{score.metadata.title}.xml'
                                                    )

                    score.write('xml', export_file_name)

                except (KeySignatureException,
                        TimeSignatureException,
                        LeadsheetParsingException) as e:
                    print(e)

    def leadsheet_to_music21(self, leadsheet):
        # must convert b to -
        if 'keySignature' not in leadsheet:
            raise KeySignatureException(f'Leadsheet {leadsheet["title"]} '
                                        f'has no keySignature')
        key_signature = leadsheet['keySignature'].replace('b', '-')
        key_signature = music21.key.Key(key_signature)

        altered_pitches_at_key = altered_pitches_music21_to_dict(
            key_signature.alteredPitches)

        if leadsheet["time"] != '4/4':
            raise TimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
                                         str(leadsheet['_id']) +
                                         ' is not in 4/4')
        if 'changes' not in leadsheet:
            raise LeadsheetParsingException('Leadsheet ' + leadsheet['title'] + ' ' +
                                            str(leadsheet['_id']) +
                                            ' do not contain "changes" attribute')
        assert_no_time_signature_changes(leadsheet)

        chords = []
        notes = []

        score = music21.stream.Score()
        part_notes = music21.stream.Part()
        part_chords = music21.stream.Part()
        for section_index, section in enumerate(leadsheet['changes']):
            for bar_index, bar in enumerate(section['bars']):
                # We consider only 4/4 pieces
                # Chords in bar
                chords_in_bar = self.chords_in_bar(bar)
                notes_in_bar = self.notes_in_bar(bar,
                                                 altered_pitches_at_key)
                chords.extend(chords_in_bar)
                notes.extend(notes_in_bar)

        # remove FakeNotes
        notes = self.remove_fake_notes(notes)
        chords = self.remove_rest_chords(chords)

        # voice_notes = music21.stream.Voice()
        # voice_chords = music21.stream.Voice()
        # todo there might be a cleaner way to do this
        part_notes.append(notes)
        part_chords.append(chords)
        for chord in part_chords.flat.getElementsByClass(
                [music21.harmony.ChordSymbol,
                 music21.expressions.TextExpression
                 ]):
            # put durations to 0.0 as required for a good rendering
            # handles both textExpression (for N.C.) and ChordSymbols
            if isinstance(chord, music21.harmony.ChordSymbol):
                new_chord = music21.harmony.ChordSymbol(chord.figure)
            elif isinstance(chord, music21.expressions.TextExpression):
                new_chord = music21.expressions.TextExpression(NC)
            else:
                raise ValueError
            part_notes.insert(chord.offset, new_chord)
        # new_chord = music21.harmony.ChordSymbol(chord.figure)
        # part_notes.insert(chord.offset, chord)
        # part_chords.append(chords)
        # voice_notes.append(notes)
        # voice_chords.append(chords)
        # part = music21.stream.Part()
        # part.insert(0, voice_notes)
        # part.insert(0, voice_chords)
        # score.append((part_notes, part_chords))
        # score.append(part)

        part_notes = part_notes.makeMeasures(
            inPlace=False,
            refStreamOrTimeRange=[0.0, part_chords.highestTime])

        # add treble clef and key signature
        part_notes.measure(1).clef = music21.clef.TrebleClef()
        part_notes.measure(1).keySignature = key_signature
        score.append(part_notes)
        set_metadata(score, leadsheet)
        # normally we should use this but it does not look good...
        # score = music21.harmony.realizeChordSymbolDurations(score)

        return score

    def remove_fake_notes(self, notes):
        """
		Transforms a list of notes possibly containing FakeNotes
		to a list of music21.note.Note with the correct durations
		:param notes:
		:return:
		"""
        previous_note = None

        true_notes = []
        for note in notes:
            if isinstance(note, FakeNote):
                assert note.symbol == SLUR_SYMBOL
                # will raise an error if the first note is a FakeNote
                cumulated_duration += note.duration.quarterLength
            else:
                if previous_note is not None:
                    previous_note.duration = music21.duration.Duration(
                        cumulated_duration)
                    true_notes.append(previous_note)
                previous_note = note
                cumulated_duration = previous_note.duration.quarterLength

        # add last note
        previous_note.duration = music21.duration.Duration(
            cumulated_duration)
        true_notes.append(previous_note)
        return true_notes

    # todo could be merged with remove_fake_notes
    def remove_rest_chords(self, chords):
        """
		Transforms a list of ChordSymbols possibly containing Rests
		to a list of ChordSymbols with the correct durations
		:param chords:
		:return:
		"""
        previous_chord = None

        true_chords = []
        for chord in chords:
            if isinstance(chord, music21.note.Rest):
                # if the first chord is a Rest,
                # replace it with a N.C.
                if previous_chord is None:
                    previous_chord = music21.expressions.TextExpression(NC)
                    cumulated_duration = 0
                cumulated_duration += chord.duration.quarterLength
            else:
                if previous_chord is not None:
                    previous_chord.duration = music21.duration.Duration(
                        cumulated_duration)
                    true_chords.append(previous_chord)
                previous_chord = chord
                cumulated_duration = previous_chord.duration.quarterLength

        # add last note
        previous_chord.duration = music21.duration.Duration(
            cumulated_duration)
        true_chords.append(previous_chord)
        return true_chords

    def notes_in_bar(self, bar,
                     altered_pitches_at_key):
        """

		:param bar:
		:param altered_pitches_at_key:
		:param current_altered_pitches:
		:return: list of music21.note.Note
		"""
        if 'melody' not in bar:
            raise LeadsheetParsingException('No melody')
        bar_melody = bar["melody"]
        current_altered_pitches = altered_pitches_at_key.copy()

        notes = []
        for json_note in bar_melody:
            # pitch is Natural pitch + accidental alteration
            # do not take into account key signatures and previous alterations
            pitch = self.pitch_from_json_note(
                json_note=json_note,
                current_altered_pitches=current_altered_pitches)

            duration = self.duration_from_json_note(json_note)

            note = general_note(pitch, duration)
            notes.append(note)
        return notes

    def duration_from_json_note(self, json_note):
        value = (json_note["duration"][:-1]
                 if json_note["duration"][-1] == 'r'
                 else json_note["duration"])
        dot = (int(json_note["dot"])
               if "dot" in json_note
               else 0)
        time_modification = 1.
        if "time_modification" in json_note:
            # a triolet is denoted as 3/2 in json format
            numerator, denominator = json_note["time_modification"].split('/')
            time_modification = int(denominator) / int(numerator)
        return note_duration(value, dot, time_modification)

    def pitch_from_json_note(self, json_note, current_altered_pitches) -> str:
        """
		Compute the real pitch of a json_note given the current_altered_pitches
		Modifies current_altered_pitches in place!
		:param json_note:
		:param current_altered_pitches:
		:return: string of the pitch or SLUR_SYMBOL if the note is tied
		"""
        # if it is a tied note
        if "tie" in json_note:
            if is_tied_left(json_note):
                return SLUR_SYMBOL

        displayed_pitch = (REST
                           if json_note["duration"][-1] == 'r'
                           else json_note["keys"][0])
        # if it is a rest
        if displayed_pitch == REST:
            return REST

        # Otherwise, if it is a true note
        # put real alterations
        unaltered_pitch = getUnalteredPitch(json_note)
        displayed_accidental = getAccidental(json_note)
        octave = getOctave(json_note)
        if displayed_accidental:
            # special case if natural
            if displayed_accidental == 'becarre':
                displayed_accidental = ''
            current_altered_pitches.update(
                {unaltered_pitch: displayed_accidental})
        # compute real pitch
        if unaltered_pitch in current_altered_pitches.keys():
            pitch = (unaltered_pitch +
                     current_altered_pitches[unaltered_pitch] +
                     octave)
        else:
            pitch = unaltered_pitch + octave
        return pitch

    def chords_in_bar(self, bar):
        """

		:param bar:
		:return: list of music21.chord.Chord with their durations
		if there are no chord on the first beat, a there is a rest
		of the correct duration instead
		"""
        chord_durations = self.chords_duration(bar=bar)
        rest_duration = chord_durations[0]

        # Rest chord during all the measure if no chords in bar
        if 'chords' not in bar:
            rest_chord = music21.note.Rest(duration=rest_duration)
            return [rest_chord]

        json_chords = bar['chords']
        chords = []

        # add Rest chord if there are no chord on the first beat
        if rest_duration.quarterLength > 0:
            rest_chord = music21.note.Rest(duration=rest_duration)
            chords.append(rest_chord)

        for json_chord, duration in zip(json_chords, chord_durations[1:]):
            chord = self.music21_chord_from_json_chord(json_chord)
            chord.duration = duration
            chords.append(chord)

        return chords

    def music21_chord_from_json_chord(self, json_chord):
        """
		Tries to find closest chordSymbol
		:param json_chord:
		:return:
		"""
        assert 'p' in json_chord
        # root
        json_chord_root = json_chord['p']
        # chord type
        if 'ch' in json_chord:
            json_chord_type = json_chord['ch']
        else:
            json_chord_type = ''

        # N.C. chords
        if json_chord_root == 'NC':
            return music21.expressions.TextExpression(NC)

        num_characters_chord_type = len(json_chord_type)
        while True:
            try:
                all_notes = self.lsdb_chord_to_notes[
                    json_chord_type[:num_characters_chord_type]]
                all_notes = [note.replace('b', '-')
                             for note in all_notes]

                interval = music21.interval.Interval(
                    noteStart=music21.note.Note('C4'),
                    noteEnd=music21.note.Note(json_chord_root))
                chord_symbol = self.chord_symbols_from_note_list(
                    all_notes=all_notes,
                    interval=interval
                )
                return chord_symbol
            except (AttributeError, KeyError):
                # if the preceding procedure did not work
                print(json_chord_type[:num_characters_chord_type])
                num_characters_chord_type -= 1

    def chord_symbols_from_note_list(self, all_notes, interval):
        """

		:param all_notes:
		:param interval:
		:return:
		"""
        skip_notes = 0
        while True:
            try:
                if skip_notes > 0:
                    notes = all_notes[:-skip_notes]
                else:
                    notes = all_notes
                chord_relative = music21.chord.Chord(notes)
                chord = chord_relative.transpose(interval)
                chord_root = chord_relative.bass().transpose(interval)
                chord.root(chord_root)
                chord_symbol = music21.harmony.chordSymbolFromChord(chord)
                # print(chord_symbol)
                return chord_symbol
            except (music21.pitch.AccidentalException,
                    ValueError) as e:
                # A7b13, m69, 13b9 not handled
                print(e)
                print(chord_relative, chord_relative.root())
                print(chord, chord.root())
                print('========')
                skip_notes += 1

    def chords_duration(self, bar):
        """

		:param bar:
		:return: list of Durations in beats of each chord in bar
		it is of length num_chords_in_bar + 1
		the first element indicates the duration of a possible
		__ chord (if there are no chords on the first beat)

		Example:(
		if bar has chords on beats 1 and 3
		[d.quarterLength
		for d in self.chords_duration(bar)] = [0, 2, 2]

		if bar has one chord on beat 3
		[d.quarterLength
		for d in self.chords_duration(bar)] = [2, 2]

		if there are no chord (in 4/4):
		[d.quarterLength
		for d in self.chords_duration(bar)] = [4]
		"""
        # if there are no chords
        if 'chords' not in bar:
            return [music21.duration.Duration(self.number_of_beats)]
        json_chords = bar['chords']
        chord_durations = [json_chord['beat']
                           for json_chord in json_chords]
        chord_durations += [self.number_of_beats + 1]
        chord_durations = np.array(chord_durations, dtype=np.float)

        # beat starts at 1...
        chord_durations -= 1
        chord_durations[1:] -= chord_durations[:-1]

        # convert to music21 objects
        chord_durations = [music21.duration.Duration(d)
                           for d in chord_durations]
        return chord_durations

    def compute_chord_dicts(self):
        # Search LSDB for chord names
        with LsdbMongo() as mongo_client:
            db = mongo_client.get_db()
            modes = db.modes
            cursor_modes = modes.find({})
            chord2notes = {}  # Chord to notes dictionary
            notes2chord = {}  # Notes to chord dictionary
            for chord in cursor_modes:
                notes = []
                # Remove white spaces from notes string
                for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
                    notes.append(note)
                notes = tuple(notes)

                # Enter entries in dictionaries
                chord2notes[chord['mode']] = notes
                if notes in notes2chord:
                    notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
                else:
                    notes2chord[notes] = [chord["mode"]]

            self.correct_chord_dicts(chord2notes, notes2chord)

            return chord2notes, notes2chord

    def correct_chord_dicts(self, chord2notes, notes2chord):
        """
		Modifies chord2notes and notes2chord in place
		to correct errors in LSDB modes (dict of chord symbols with notes)
		:param chord2notes:
		:param notes2chord:
		:return:
		"""
        # Add missing chords
        # b5
        notes2chord[('C4', 'E4', 'Gb4')] = notes2chord[('C4', 'E4', 'Gb4')] + ['b5']
        chord2notes['b5'] = ('C4', 'E4', 'Gb4')
        # b9#5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'D#5')] = 'b9#b'
        chord2notes['b9#5'] = ('C4', 'E4', 'G#4', 'Bb4', 'D#5')
        # 7#5#11 is WRONG in the database
        # C4 F4 G#4 B-4 D5 instead of C4 E4 G#4 B-4 D5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'F#5')] = '7#5#11'
        chord2notes['7#5#11'] = ('C4', 'E4', 'G#4', 'Bb4', 'F#5')

    # F#7#9#11 is WRONG in the database

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
        notes, chords = notes_and_chords(leadsheet)
        pitches = [n.pitch.midi for n in notes if n.isNote]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        return (min_pitch >= self.pitch_range[0]
                and max_pitch <= self.pitch_range[1])

    def random_leadsheet_tensor(self, sequence_length):
        lead_tensor = np.random.randint(len(self.symbol2index_dicts[self.NOTES]),
                                        size=sequence_length * self.subdivision)
        chords_tensor = np.random.randint(len(self.symbol2index_dicts[self.CHORDS]),
                                          size=sequence_length)
        lead_tensor = torch.from_numpy(lead_tensor).long()
        chords_tensor = torch.from_numpy(chords_tensor).long()

        return lead_tensor, chords_tensor


if __name__ == '__main__':
    leadsheet_it_gen = LeadsheetIteratorGenerator(num_elements=5)
    dataset = LsdbDataset(corpus_it_gen=leadsheet_it_gen,
                          sequences_size=8)
    dataset.initialize()
# dataset.make_score_dataset()
# dataset.test()
