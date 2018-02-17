import glob
import os
from DatasetManager.helpers import SLUR_SYMBOL
import music21
from DatasetManager.lsdb.lsdb_exceptions import TimeSignatureException
from bson import ObjectId

REST = 'R'
NC = 'N.C.'

# dictionary
note_values = {
	'q':  1.,
	'h':  2.,
	'w':  4.,
	'8':  0.5,
	'16': 0.25,
	'32': 0.125,
}

music21_alterations_to_json = {
	'-': 'b',
	'#': '#',
	'':  'n'
}


class FakeNote:
	"""
	Class used to have SLUR_SYMBOLS with a duration
	"""

	def __init__(self, symbol, duration):
		self.symbol = symbol
		self.duration = duration

	def __repr__(self):
		return f'<FakeNote {self.symbol}>'


def general_note(pitch: str, duration: float):
	duration = music21.duration.Duration(duration)

	if pitch == SLUR_SYMBOL:
		return FakeNote(symbol=pitch,
		                duration=duration)
	elif pitch == REST:
		f = music21.note.Rest()
		f.duration = duration
		return f
	else:
		f = music21.note.Note(pitch=pitch)
		f.duration = duration
		return f


def is_tied_left(json_note):
	"""

	:param json_note:
	:return: True is the json_note is tied FROM the left
	"""
	return ('tie' in json_note
	        and
	        'stop' in json_note["tie"].split('_'))


def is_tied_right(json_note):
	"""

	:param json_note:
	:return: True is the json_note is tied FROM the right
	"""
	return ('tie' in json_note
	        and
	        'start' in json_note["tie"].split('_'))


def note_duration(note_value, dots, time_modification):
	"""

	:param time_modification:
	:type note_value: str
	:param note_value: duration of the note regardless of the dots
	:param dots: number of dots (0, 1 or 2)
	:return: the actual duration in beats (float)
	"""
	duration = note_values[note_value]
	for dot in range(dots):
		duration *= 1.5
	return duration * time_modification


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


def assert_no_time_signature_changes(leadsheet):
	changes = leadsheet['changes']
	for change in changes:
		if ('(timeSig' in change or
				('timeSignature' in change
				 and
				 not change['timeSignature'] == '')
		):
			raise TimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
			                             str(leadsheet['_id']) +
			                             ' has multiple time changes')


def leadsheet_on_ticks(leadsheet, tick_values):
	notes, chords = notes_and_chords(leadsheet)
	eps = 1e-5
	for n in notes:
		i, d = divmod(n.offset, 1)
		flag = False
		for tick_value in tick_values:
			if tick_value - eps < d < tick_value + eps:
				flag = True
		if not flag:
			return False

	return True


def set_metadata(score, lsdb_leadsheet):
	"""
	
	:param score: 
	:param lsdb_leadsheet: 
	:return: 
	"""
	score.insert(0, music21.metadata.Metadata())

	if 'title' in lsdb_leadsheet:
		score.metadata.title = lsdb_leadsheet['title']
	if 'composer' in lsdb_leadsheet:
		score.metadata.composer = lsdb_leadsheet['composer']


def notes_and_chords(leadsheet):
	"""

	:param leadsheet: music21 score
	:return:
	"""
	notes = leadsheet.parts[0].flat.notesAndRests
	notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
	chords = leadsheet.parts[0].flat.getElementsByClass(
		[music21.harmony.ChordSymbol,
		 music21.expressions.TextExpression
		 ])
	return notes, chords


class LeadsheetIteratorGenerator:
	"""
	Object that returns a iterator over leadsheet (as music21 scores)
	when called
	:return:
	"""

	def __init__(self, num_elements=None):
		self.num_elements = num_elements

	def __call__(self, *args, **kwargs):
		it = (
			leadsheet
			for leadsheet in self.leadsheet_generator()
		)
		return it

	def leadsheet_generator(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		leadsheet_paths = glob.glob(
			os.path.join(dir_path, 'xml/*.xml'))
		if self.num_elements is not None:
			leadsheet_paths = leadsheet_paths[:self.num_elements]
		for leadsheet_path in leadsheet_paths:
			try:
				yield music21.converter.parse(leadsheet_path)
			except ZeroDivisionError:
				print(f'{leadsheet_path} is not parsable')


# list of badly-formatted leadsheets
exclude_list_ids = [
	ObjectId('512dbeca58e3380f1c000000'),  # And on the third day
]
