import pickle

import music21
import re
from bson import ObjectId

import numpy as np
from DatasetManager.helpers import SLUR_SYMBOL
from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import altered_pitches_music21_to_dict, REST, \
	getUnalteredPitch, getAccidental, getOctave, note_duration, retain_altered_pitches_if_tied
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.lsdb.lsdb_exceptions import *
from music21 import key


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

	def make_tensor_dataset(self,
	                        query={'_id': ObjectId("57e4f5de58e338a57215d906")},
	                        database_name='tmp'):
		"""

		:param database_name:
		:param query: default query is debug lstm leadsheet
		:return:
		"""
		with LsdbMongo() as client:
			db = client.get_db()
			leadsheets = db.leadsheets.find()
			pieces = []

			for leadsheet in leadsheets:
				print(leadsheet['title'])
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


	def leadsheet_to_music21(self, leadsheet):
		# must convert b to -
		if 'keySignature' not in leadsheet:
			raise KeySignatureException(f'Leadsheet {leadsheet["title"]} '
			                            f'has no keySignature')
		key_signature = leadsheet['keySignature'].replace('b', '-')

		altered_pitches_at_key = altered_pitches_music21_to_dict(
			key.Key(key_signature).alteredPitches)

		if leadsheet["time"] != '4/4':
			raise TimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
			                             str(leadsheet['_id']) +
			                             ' is not in 4/4')
		if "changes" not in leadsheet:
			raise LeadsheetParsingException('Leadsheet ' + leadsheet['title'] + ' ' +
			                                str(leadsheet['_id']) +
			                                ' do not contain "changes" attribute')

		chords = []
		melody = []
		pitch = None
		score = music21.stream.Score()
		part = music21.stream.Part()
		# current_altered_pitches is a dict
		current_altered_pitches = altered_pitches_at_key.copy()

		for section_index, section in enumerate(leadsheet["changes"]):
			for bar_index, bar in enumerate(section["bars"]):

				# We consider only 4/4 pieces
				# Chords in bar
				chords_in_bar = self.chords_in_bar(bar)

				### Notes in bar ###
				note_index = -1
				bar_melody = bar["melody"]

				start = 0
				end = 0
				# add alterations if first note is an altered tied note

				first_note_in_bar = bar_melody[0]
				if ("tie" in first_note_in_bar
						and
						"stop" in first_note_in_bar["tie"]):
					current_altered_pitches = retain_altered_pitches_if_tied(
						current_altered_pitches,
						first_note_in_bar)
				else:
					current_altered_pitches = {}

				tmp = altered_pitches_at_key.copy()
				tmp.update(current_altered_pitches)
				current_altered_pitches = tmp

				for beat in range(4):
					for tick in self.tick_values:
						# we are afraid of precision issues
						assert beat + tick >= start - 1e-4
						assert beat + tick <= end + 1e-4
						# todo 16th notes ?!

						# continuation
						if beat + tick < end - 1e-4:
							melody.append(SLUR_SYMBOL)
						# new note
						if abs(beat + tick - end) < 1e-4:
							note_index += 1
							current_note = bar_melody[note_index]

							if pitch != SLUR_SYMBOL:
								preceding_note = pitch
							# pitch is Natural pitch + accidental alteration
							# do not take into account key signatures and previous alterations
							displayed_pitch = (REST
							                   if current_note["duration"][-1] == 'r'
							                   else current_note["keys"][0])
							value = (current_note["duration"][:-1]
							         if current_note["duration"][-1] == 'r'
							         else current_note["duration"])
							dot = (int(current_note["dot"])
							       if "dot" in current_note
							       else 0)

							# put real alterations
							if displayed_pitch != REST:
								unaltered_pitch = getUnalteredPitch(current_note)
								displayed_accidental = getAccidental(current_note)
								octave = getOctave(current_note)
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

							else:
								pitch = REST

							if "tie" in current_note:
								if 'stop' in current_note["tie"].split('_'):
									if preceding_note == pitch:
										pitch = SLUR_SYMBOL
									else:
										raise TieException('Tie between notes '
										                   'with different pitches in '
										                   + leadsheet['title'] + ' ' +
										                   str(leadsheet[
											                       '_id']) + ' at section '
										                   + str(section_index) + ', bar '
										                   + str(bar_index))

							time_modification = 1.
							if "time_modification" in current_note:
								if current_note["time_modification"] == '3/2':
									time_modification = 2 / 3
								else:
									raise UnknownTimeModification
							start = end
							end = start + note_duration(value, dot, time_modification)

							melody.append(pitch)
		return melody, chords

	def chords_in_bar(self, bar):
		json_chords = bar['chords']
		chord_durations = self.chords_duration(bar=bar)

		chords = []

		# add Rest chord if there are no chord on the first beat
		rest_duration = chord_durations[0]
		if rest_duration > 0:
			rest_chord = music21.note.Rest(duration=rest_duration)
			chords.append(rest_chord)

		for json_chord, duration in zip(json_chords, chord_durations[1:]):
			chord = self.music21_chord_from_json_chord(json_chord)
			chord.duration = music21.duration.Duration(duration)
			chords.append(chord)

		return chords

	def music21_chord_from_json_chord(self, json_chord):
		assert 'p' in json_chord
		json_chord_root = json_chord['p']
		# add chord type if exists
		if 'ch' in json_chord:
			json_chord_type = json_chord['ch']
		else:
			json_chord_type = ''

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
				# A7b13, m69 not handled
				print(e)
				print(json_chord_root + json_chord_type)
				print(chord_relative, chord_relative.root())
				print(chord, chord.root())
				print('========')
				skip_notes += 1


	def chords_duration(self, bar):
		"""

		:param bar:
		:return: np.array of durations in beats (float) of each chord in bar
		it is of length num_chords_in_bar + 1
		the first element indicates the duration of a possible
		__ chord (if there are no chords on the first beat)

		Example:
		if bar has chords on beats 1 and 3
		self.chords_duration(bar) = [0, 2, 2]

		if bar has one chord on beat 3
		self.chords_duration(bar) = [2, 2]
		"""
		json_chords = bar['chords']
		chord_durations = [json_chord['beat']
		                   for json_chord in json_chords]
		chord_durations += [self.number_of_beats + 1]
		chord_durations = np.array(chord_durations, dtype=np.float)

		# beat starts at 1...
		chord_durations -= 1
		chord_durations[1:] -= chord_durations[:-1]
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

			return chord2notes, notes2chord


if __name__ == '__main__':
	dataset = LsdbDataset()
	dataset.make_tensor_dataset()

# To obtain the chord representation of the in the score, change the music21.harmony.ChordSymbol.writeAsChord to True. Unless otherwise specified, the duration of this chord object will become 1.0. If you have a leadsheet, run music21.harmony.realizeChordSymbolDurations() on the stream to assign the correct (according to offsets) duration to each harmony object.)