import pickle
import re

# bson is installed with pymongo
from bson import ObjectId


from DatasetManager.lsdb.LsdbMongo import LsdbMongo


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


	def parse_leadsheet(self, leadsheet):
		# must convert b to -
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
		# current_altered_pitches is a dict
		current_altered_pitches = altered_pitches_at_key.copy()

		for section_index, section in enumerate(leadsheet["changes"]):
			for bar_index, bar in enumerate(section["bars"]):

				# We consider only 4/4 pieces

				# Chords in bar
				chord_index = 0
				for beat in range(1, 5):
					bar_chords = bar["chords"]

					if (chord_index < len(bar_chords)
							and
							bar_chords[chord_index]["beat"] == beat):
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
						chords.append({"p":  SLUR_SYMBOL,
						               'bp': '',
						               'ch': ''})

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
