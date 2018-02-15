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



def set_metadata(score, leadsheet):
	score.insert(0, music21.metadata.Metadata())

	if 'title' in leadsheet:
		score.metadata.title = leadsheet['title']
	if 'composer' in leadsheet:
		score.metadata.title = leadsheet['composer']

# list of badly-formatted leadsheets
exclude_list_ids = [
	ObjectId('512dbeca58e3380f1c000000'), # And on the third day
]
