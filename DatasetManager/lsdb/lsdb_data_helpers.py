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
