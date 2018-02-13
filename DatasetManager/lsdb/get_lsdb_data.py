import pickle
import re

# bson is installed with pymongo
from bson import ObjectId
from music21 import key

from DatasetManager.lsdb.LsdbMongo import LsdbMongo

TIE = "_"
REST = 'R'

# dictionary
note_values = {
	'q':  1.,
	'h':  2.,
	'w':  4.,
	'8':  0.5,
	'16': 0.25
}

music21_alterations_to_json = {
	'-': 'b',
	'#': '#',
	'':  'n'
}

tick_values = [0., 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4]


class TieException(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


class ParsingException(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


class LeadsheetParsingException(ParsingException):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


class UnknownTimeModification(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


class TimeSignatureException(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)


def note_duration(note_value, dots, time_modification):
	"""

	:param time_modification:
	:type note_value: str
	:param note_value: duration of the note regardless of the dots
	:param dots: number of dots (0, 1 or 2)
	:return: the actual duration in beats
	"""
	duration = note_values[note_value]
	for dot in range(dots):
		duration *= 1.5
	return duration * time_modification


def create_chord_dicts():
	# Search LSDB for chord names
	modes = db.modes
	cursor_modes = modes.find({})
	chord2notes = {}  # Chord to notes dictionary
	notes2chord = {}  # Notes to chord dictionary
	for chord in cursor_modes:
		notes = ''
		# Remove white spaces from notes string
		for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
			notes = notes + note + ' '
		# Enter entries in dictionaries
		chord2notes[chord["mode"]] = notes
		if notes in notes2chord:
			notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
		else:
			notes2chord[notes] = [chord["mode"]]

	# Add missing chords
	notes2chord[u'C4 E4 Gb4 '] = notes2chord[u'C4 E4 Gb4 '] + [u'b5']
	chord2notes[u'b5'] = u'C4 E4 Gb4 '
	notes2chord[u'C4 E4 G#4 Bb4 D#5 '] = u'b9#5'
	chord2notes[u'b9#5'] = u'C4 E4 G#4 Bb4 D#5 '

	# Create dict to tokenize tone names. All alterations will be flats b (F# -> Gb).
	tone_names_uniq = {'A#': 'Bb',
	                   'B#': 'C',
	                   'C#': 'Db',
	                   'D#': 'Eb',
	                   'E#': 'F',
	                   'F#': 'Gb',
	                   'G#': 'Ab',
	                   'A':  'A',
	                   'B':  'B',
	                   'C':  'C',
	                   'D':  'D',
	                   'E':  'E',
	                   'F':  'F',
	                   'G':  'G',
	                   'Ab': 'Ab',
	                   'Bb': 'Bb',
	                   'Cb': 'B',
	                   'Db': 'Db',
	                   'Eb': 'Eb',
	                   'Fb': 'E',
	                   'Gb': 'Gb',
	                   '':   ''
	                   }

	# Dictionary use for transposing chord seqs. Returns the next tone in the circle of fifths
	cof_next = {'C':  'G',
	            'G':  'D',
	            'D':  'A',
	            'A':  'E',
	            'E':  'B',
	            'B':  'Gb',
	            'Gb': 'Db',
	            'Db': 'Ab',
	            'Ab': 'Eb',
	            'Eb': 'Bb',
	            'Bb': 'F',
	            'F':  'C',
	            '':   ''
	            }

	return chord2notes, notes2chord


def getAccidental(json_note):
	"""

	:param json_note:
	:return:
	"""
	# Pas plus de bémols ni dièses
	if '##' in json_note['keys'][0]:
		return '##'
	if 'bb' in json_note['keys'][0]:
		return '--'
	if 'n' in json_note['keys'][0]:
		return 'becarre'
	if 'b' in json_note['keys'][0]:
		return '-'
	if '#' in json_note['keys'][0]:
		return '#'
	return ''


def getOctave(json_note):
	"""

	:param json_note:
	:return: octave as string
	"""
	return json_note['keys'][0][-1]


def getUnalteredPitch(json_note):
	"""

	:param json_note:
	:return: 'Bb/4' -> B
	"""
	return json_note['keys'][0][0]


def retain_altered_pitches_if_tied(altered_pitches, json_note):
	"""

	:param altered_pitches: dict
	:param note: json note
	:return:
	"""
	pitch = getUnalteredPitch(json_note)
	if pitch in altered_pitches.keys():
		return {pitch: altered_pitches[pitch]}
	else:
		return {}


def altered_pitches_music21_to_dict(alteredPitches):
	"""

	:param alteredPitches:
	:return: dictionary {'B': 'b', 'C': ''}
	"""
	d = {}
	# todo natural ?
	for pitch in alteredPitches:
		d.update({pitch.name[0]: music21_alterations_to_json[pitch.name[1]]})
	return d


def parse_leadsheet(leadsheet):
	# must convert b to -
	key_signature = leadsheet['keySignature'].replace('b', '-')

	altered_pitches_at_key = altered_pitches_music21_to_dict(key.Key(key_signature).alteredPitches)

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
	# current_altered_pitches is a dict
	current_altered_pitches = altered_pitches_at_key.copy()

	for section_index, section in enumerate(leadsheet["changes"]):
		for bar_index, bar in enumerate(section["bars"]):

			# We consider only 4/4 pieces

			# Chords in bar
			chord_index = 0
			for beat in range(1, 5):
				bar_chords = bar["chords"]

				if chord_index < len(bar_chords) and bar_chords[chord_index]["beat"] == beat:
					chord = bar_chords[chord_index]
					# print(chord)
					assert "p" in chord
					p = chord['p']
					bp = chord["bp"] if "bp" in chord else ''
					ch = chord["ch"] if "ch" in chord else ''
					chords.append({"p":  p,
					               'bp': bp,
					               'ch': ch})
					chord_index += 1
				else:
					chords.append({"p":  TIE,
					               'bp': '',
					               'ch': ''})

			### Notes in bar ###
			note_index = -1
			bar_melody = bar["melody"]

			start = 0
			end = 0
			# add alterations if first note is an altered tied note

			first_note_in_bar = bar_melody[0]
			if "tie" in first_note_in_bar and "stop" in first_note_in_bar["tie"]:
				current_altered_pitches = retain_altered_pitches_if_tied(current_altered_pitches,
				                                                         first_note_in_bar)
			else:
				current_altered_pitches = {}

			tmp = altered_pitches_at_key.copy()
			tmp.update(current_altered_pitches)
			current_altered_pitches = tmp

			for beat in range(4):
				for tick in tick_values:
					# we are afraid of precision issues
					assert beat + tick >= start - 1e-4
					assert beat + tick <= end + 1e-4
					# todo 16th notes ?!

					# continuation
					if beat + tick < end - 1e-4:
						melody.append(TIE)
					# new note
					if abs(beat + tick - end) < 1e-4:
						note_index += 1
						current_note = bar_melody[note_index]

						if pitch != TIE:
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
									pitch = TIE
								else:
									raise TieException('Tie between notes '
									                   'with different pitches in '
									                   + leadsheet['title'] + ' ' +
									                   str(leadsheet['_id']) + ' at section '
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


def create_leadsheet_database(query={'_id': ObjectId("57e4f5de58e338a57215d906")},
                              database_name='tmp'):
	"""

	:param database_name:
	:param query: default query is debug lstm leadsheet
	:return:
	"""
	with LsdbMongo() as client:
		db = client.get_db()
		leadsheets = db.leadsheets.find(query)
		# 519377a058e3387864000001
		# 519377a058e3387864000005
		# pieces stores all leadsheets in matrix form
		pieces = []

		for leadsheet in leadsheets:
			try:
				print(leadsheet['title'])
				pieces.append(parse_leadsheet(leadsheet))
			except Exception as e:
				print(e)

		export_file_name = 'tmp_database/' + database_name + '.pickle'
		pickle.dump(pieces, open(export_file_name, 'wb'))
		print(str(len(pieces)) + ' pieces written in ' + export_file_name)
		return pieces


if __name__ == '__main__':
	# chord2notes, notes2chord = create_chord_dicts()
	# pieces = create_leadsheet_database(query={'composer': 'Michel Legrand'},
	#                                    database_name='michel_legrand')
	# pieces = create_leadsheet_database(query={'source': '519377a058e3387864000005'},
	#                                    database_name='bill_evans_fake_book')

	pieces = create_leadsheet_database(
		query={'source': "519377a058e3387864000001"},
		database_name='anthologie')
	# pieces = create_leadsheet_database(query={"_id": ObjectId("5120bab558e338a76c000001")})
	# for k, note in enumerate(pieces[0][0]):
	#     print(note + '\t', end='')
	#     if k % 6 == 5:
	#         print('\t \t', end='')
	#     if k % 24 == 23:
	#         print('')
