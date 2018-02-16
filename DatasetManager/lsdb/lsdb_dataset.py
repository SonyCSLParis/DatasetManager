import pickle

import music21
import re
from bson import ObjectId

import numpy as np
from DatasetManager.helpers import SLUR_SYMBOL
from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import altered_pitches_music21_to_dict, REST, \
	getUnalteredPitch, getAccidental, getOctave, note_duration, \
	is_tied_left, general_note, FakeNote, assert_no_time_signature_changes, NC, \
	exclude_list_ids, set_metadata
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.lsdb.lsdb_exceptions import *



class LsdbDataset(MusicDataset):
	def __init__(self):
		super(LsdbDataset, self).__init__()
		music21.harmony.addNewChordSymbol(
			'seventh-suspended-fourth', '1,4,5,-7', ['7sus4', '7sus'])
		self.tick_values = [0.,
		                    1 / 4,
		                    1 / 3,
		                    1 / 2,
		                    2 / 3,
		                    3 / 4]
		self.number_of_beats = 4
		self.chord_to_notes, self.notes_to_chord = self.compute_chord_dicts()

	def make_score_dataset(self, database_name='lsdb.pickle'):
		"""

		:param database_name:
		:param query: default query is debug lstm leadsheet
		:return:
		"""
		# todo add query
		with LsdbMongo() as client:
			db = client.get_db()
			leadsheets = db.leadsheets.find({'_id': {
				'$nin': exclude_list_ids
			}})
			pieces = []

			for leadsheet in leadsheets:
				print(leadsheet['title'])
				print(leadsheet['_id'])
				try:
					pieces.append(self.leadsheet_to_music21(leadsheet))
				except (KeySignatureException, TimeSignatureException) as e:
					print(e)

		export_file_name = 'tmp_database/' + database_name + '.pickle'
		pickle.dump(pieces, open(export_file_name, 'wb'))
		print(str(len(pieces)) + ' pieces written in ' + export_file_name)
		return pieces

	def __repr__(self):
		# TODO
		return f'LsdbDataset(' \
		       f')'

	def make_tensor_dataset(self):
		pass

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
				# will raise an error if the first chord is a FakeNote
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
			if json_note["time_modification"] == '3/2':
				time_modification = 2 / 3
			else:
				raise UnknownTimeModification
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

		all_notes = self.chord_to_notes[json_chord_type]
		all_notes = [note.replace('b', '-')
		             for note in all_notes]

		interval = music21.interval.Interval(
			noteStart=music21.note.Note('C4'),
			noteEnd=music21.note.Note(json_chord_root))
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
				print(json_chord_root + json_chord_type)
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

			return chord2notes, notes2chord

	def test(self):
		with LsdbMongo() as client:
			db = client.get_db()
			leadsheets = db.leadsheets.find(
				{'_id': ObjectId('512c7f4758e338b31f000000')})
			leadsheet = next(leadsheets)
			print(leadsheet['title'])
			score = self.leadsheet_to_music21(leadsheet)
			score.show()


if __name__ == '__main__':
	dataset = LsdbDataset()
	dataset.test()
	dataset.make_score_dataset()

# To obtain the chord representation of the in the score, change the music21.harmony.ChordSymbol.writeAsChord to True. Unless otherwise specified, the duration of this chord object will become 1.0. If you have a leadsheet, run music21.harmony.realizeChordSymbolDurations() on the stream to assign the correct (according to offsets) duration to each harmony object.)
