import os
import music21
import sys
import numpy as np

from tqdm import tqdm
from glob2 import glob
from fractions import Fraction
from music21 import interval, meter
from music21.abcFormat import ABCHandlerException

from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, PAD_SYMBOL
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetTimeSignatureException
from bson import ObjectId

# dictionary
note_values = {
    'q': 1.,
    'h': 2.,
    'w': 4.,
    '8': 0.5,
    '16': 0.25,
    '32': 0.125
}

tick_values = [
	0,
    Fraction(1, 4),
	Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
	Fraction(3, 4)
]

class FakeNote:
	"""
	Class used to have SLUR_SYMBOLS with a duration
	"""

	def __init__(self, symbol, duration):
		self.symbol = symbol
		self.duration = duration

	def __repr__(self):
		return f'<FakeNote {self.symbol}>'


def score_on_ticks(score, tick_values):
	notes, _ = notes_and_chords(score)
	eps = 1e-5
	for n in notes:
		_, d = divmod(n.offset, 1)
		flag = False
		for tick_value in tick_values:
			if tick_value - eps < d < tick_value + eps:
				flag = True
		if not flag:
			return False

	return True


def get_notes_in_measure(measure):
	"""

	:param leadsheet: music21 measure object
	:return:
	"""
	notes = measure.flat.notesAndRests
	notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
	return notes

def notes_and_chords(score):
	"""

	:param score: music21 score
	:return:
	"""
	notes = score.parts[0].flat.notesAndRests
	notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
	chords = score.parts[0].flat.getElementsByClass(
		[music21.harmony.ChordSymbol,
		 music21.expressions.TextExpression
		 ])
	return notes, chords

def get_notes(score):
    """
    Analyzes the
	:param score: music21 score
	:return:
	"""
    s = score.parts[0]
    #score.show()

    measures = s.recurse().getElementsByClass(music21.stream.Measure)
    num_measures = len(measures)
    # check for pick-up measures
    if measures[0].barDurationProportion() != 1.0:
        offset = measures[0].paddingLeft
        measures[0].insertAndShift(
            0.0, music21.note.Rest(quarterLength=offset))
    for m in measures:
        # check for pick-up measure
        notes = get_notes_in_measure(m)
    # offset_map = leadsheet.parts[0].measureOffsetMap()
    # num_measures = len(offset_map)
    # offest_list = list(offset_map.keys())
    # for i in np.arange(num_measures):
    #    m = offset_map[offest_list[i]][0]
    #leadsheet.show()
    notes = s.flat.notesAndRests
    notes = [n for n in notes if not isinstance(
        n, music21.harmony.ChordSymbol)]
    return notes


class FolkIteratorGenerator:
	"""
	Object that returns a iterator over folk dataset (as music21 scores)
	when called
	:return:
	"""

	def __init__(self, num_elements=None):
		if num_elements == None:
			self.num_elements = 25000
		else:
			self.num_elements = num_elements
		self.package_dir = os.path.dirname(os.path.realpath(__file__))
		self.raw_dataset_dir = os.path.join(
            self.package_dir,
            'raw_data',
        )
		
		self.raw_dataset_url = 'https://raw.githubusercontent.com/IraKorshunova/' \
                               'folk-rnn/master/data/' \
	                           'sessions_data_clean.txt'

		if not os.path.exists(self.raw_dataset_dir):
			os.mkdir(self.raw_dataset_dir)

		self.full_raw_dataset_filepath = os.path.join(
            self.raw_dataset_dir,
            'raw_dataset_full.txt'
        )

		if not os.path.exists(self.full_raw_dataset_filepath):
			self.download_raw_dataset()
			self.split_raw_dataset()

		self.valid_files_list = os.path.join(
            self.raw_dataset_dir,
            'valid_tune_filepaths.txt'
        )
		self.valid_tune_filepaths = []

	def download_raw_dataset(self):
		if os.path.exists(self.full_raw_dataset_filepath):
		    print('The Session dump already exists')
		else:
		    print('Downloading The Session dump')
		    os.system(
		        f'wget -L {self.raw_dataset_url} -O {self.full_raw_dataset_filepath}')

	def split_raw_dataset(self):
		print('Splitting raw dataset')
		with open(self.full_raw_dataset_filepath) as full_raw_dataset_file:
		    tune_index = 0

		    current_song_filepath = os.path.join(self.raw_dataset_dir,
                                                 f'tune_{tune_index}.abc')
		    current_song_file = open(current_song_filepath, 'w+')
		    for line in full_raw_dataset_file:
		        if line == '\n':
		            tune_index += 1
		            current_song_file.flush()
		            current_song_file.close()
		            current_song_filepath = os.path.join(self.raw_dataset_dir,
                                                         f'tune_{tune_index}.abc')
		            current_song_file = open(current_song_filepath, 'w+')
		        else:
		            current_song_file.write(line)

	def __call__(self, *args, **kwargs):
		it = (
			score
			for score in self.score_generator()
		)
		return it

	def score_generator(self):
		self.get_valid_tune_filepaths()
		for score_index, score_path in enumerate(self.valid_tune_filepaths):
			if score_index > self.num_elements:
				continue
			try:
				yield self.get_score_from_path(score_path)
			except ZeroDivisionError:
				print(f'{score_path} is not parsable')

	def get_valid_tune_filepaths(self):
		"""
        Stores a list of filepaths for all valid tunes in dataset
        """
		if os.path.exists(self.valid_files_list):
			print('List already exists. Reading it now')
			f = open(self.valid_files_list, 'r')
			self.valid_tune_filepaths = [line.rstrip('\n') for line in f]
			print(f'Number of file: {len(self.valid_tune_filepaths)}')
			return

		print('Checking dataset for valid files')
		tune_filepaths = glob(f'{self.raw_dataset_dir}/tune*')
		count = 0
		num_chords = 0
		num_multivoice = 0
		self.valid_file_indices = []
		self.valid_tune_filepaths = []
		for tune_index, tune_filepath in tqdm(enumerate(tune_filepaths)):
		    title = self.get_title(tune_filepath)
		    if title is None:    
		        continue
		    if self.tune_contains_chords(tune_filepath):
		        num_chords += 1
		        continue
		    if self.tune_is_multivoice(tune_filepath):
		        num_multivoice += 1
		        continue
		    try:
		        score = self.get_score_from_path(tune_filepath)
		        ts = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
		        # ignore files where notes are not on ticks
		        if not score_on_ticks(score, tick_values):
		            continue    
		        # ignore files with no notes 
		        notes, _ = notes_and_chords(score)
                
		        pitches = [n.pitch.midi for n in notes if n.isNote]
		        if pitches == []:
		            continue
		        # ignore files with too few or too high notes
		        MAX_NOTES = 140
		        MIN_NOTES = 40
		        if len(notes) < MIN_NOTES or len(notes) > MAX_NOTES:
		            continue
		        # ignotre files with non 4/4 and 3/4 time signatures
		        if len(ts) > 1:
		            continue
		        else:
		            ts_num = ts[0].numerator
		            ts_den = ts[0].denominator
		            if ts_den != 4:
		                continue
		            else:
		                if ts_num != 4 and ts_num != 3:
		                    continue
		                else:
		                    # ignore files with 32nd and 64th notes
		                    dur_list = [n.duration for n in notes if n.isNote] 
		                    for dur in dur_list:
		                        d = dur.type
		                        if d == '32nd':
		                            break
		                        elif d == '64th':
		                            break
		                        elif d == 'complex':
		                            # TODO: bad hack. fix this !!!
		                            if len(dur.components) > 2:
		                                break
		                    else:
		                        self.valid_file_indices.append(tune_index)
		                        self.valid_tune_filepaths.append(tune_filepath)
		                        count += 1
		    except (music21.abcFormat.ABCHandlerException,
		            music21.abcFormat.ABCTokenException,
                    music21.duration.DurationException,
                    music21.pitch.AccidentalException,
                    music21.meter.MeterException,
                    UnboundLocalError,
                    ValueError,
                    ABCHandlerException) as e:
		        print('Error when parsing ABC file')
		        print(e)

		f = open(self.valid_files_list, 'w')
		for tune_filepath in self.valid_tune_filepaths:
		    f.write("%s\n" % tune_filepath)
		f.close()
		self.num_valid_files = count
		self.num_with_chords = num_chords
		self.num_multivoice = num_multivoice
		# print(count, num_chords, num_multivoice)
        # print(count / len(tune_filepaths) * 100, 
        #        num_chords / len(tune_filepaths) * 100,
        #        num_multivoice / len(tune_filepaths) * 100)

	def get_score_from_path(self, tune_filepath):
	    """
	    Extract music21 score from provided path to the tune
        
	    :param tune_filepath: path to tune in .abc format
	    :return: music21 score object
	    """ 
	    score = music21.converter.parse(tune_filepath, format='abc')
	    score = self.fix_pick_up_measure_offset(score)
	    return score
	
	def fix_pick_up_measure_offset(self, score):
	    """
	    Adds rests to the pick-up measure (if-any)
	    : param score: music21 score object
	    """
	    measures = score.recurse().getElementsByClass(music21.stream.Measure)
	    num_measures = len(measures)
	    # add rests in pick-up measures
	    if num_measures > 0:
	        m0_dur = measures[0].barDurationProportion()
	        m1_dur = measures[0].barDurationProportion()
	        if m0_dur != 1.0:
	            if m0_dur + m1_dur != 1.0:
	                offset = measures[0].paddingLeft
	                measures[0].insertAndShift(0.0, music21.note.Rest(quarterLength=offset))
	                for i,m in enumerate(measures):
	                    # shift the offset of all other measures
	                    if i != 0:
	                        m.offset += offset
	    return score

	def scan_dataset(self):
	    num_files = len(self.valid_tune_filepaths)
	    num_notes = np.zeros(num_files, dtype=int)
	    num_4_4 = 0
	    num_3_4 = 0
	    num_6_8 = 0
	    num_other_ts = 0
	    num_multi_ts = 0
	    pitch_dist = np.zeros(128)
	    min_pitch = 127
	    max_pitch = 0
	    dur_dist = np.zeros(8)
	    num_fast_note_files = 0
	
	    for i in tqdm(range(num_files)):
	        tune_filepath = self.valid_tune_filepaths[i]
	        score = self.get_score_from_path(tune_filepath)
        
	        fast_note_flag = False
	        # get number of notes, pitch range and distribution
	        notes, _ = notes_and_chords(score)
	        pitches = [n.pitch.midi for n in notes if n.isNote]
	        if pitches == []:
	            continue
	        num_notes[i] = len(notes)
	        min_p = min(pitches)
	        max_p = max(pitches)
	        if min_p < min_pitch:
	            min_pitch = min_p
	        if max_p > max_pitch:
	            max_pitch = max_p
	        for p in pitches:
	            pitch_dist[p] += 1
	
	        # get duration distribution
	        dur_list = [n.duration for n in notes if n.isNote]
	        for dur in dur_list:
	            d = dur.type
	            if d == 'quarter':
	                dur_dist[0] += 1
	            elif d == 'eighth':
	                dur_dist[1] += 1
	            elif d == 'half':
	                dur_dist[2] += 1
	            elif d == '16th':
	                dur_dist[3] += 1
	            elif d == 'whole':
	                dur_dist[4] += 1
	            elif d == '32nd':
	                dur_dist[5] += 1
	            if not fast_note_flag:
                        num_fast_note_files += 1
                        fast_note_flag = True
	            elif d == '64th':
		            dur_dist[6] += 1
	            if not fast_note_flag:
	                num_fast_note_files += 1
	                fast_note_flag = True
	            else:
	                if d == 'complex':
	                    dur_dist[7] += 1
	                    print('**')
	                    print(dur.components)
	                    if not fast_note_flag:
	                        num_fast_note_files += 1
	                        fast_note_flag = True


            # get time signature
	        ts = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
	        if len(ts) > 1:
	            num_multi_ts += 1
	        else:
	            ts_num = ts[0].numerator
	            ts_den = ts[0].denominator
	            if ts_den == 4:
	                if ts_num == 4:
	                    num_4_4 += 1
	                elif ts_num == 3:
	                    num_3_4 += 1
	                else:
	                    num_other_ts += 1
	            elif ts_den == 8:
	                if ts_num == 6:
	                    num_6_8 += 1
	                else: 
	                    num_other_ts += 1
	    print(f'Num Files: {num_files}')
	    print(f'4/4: {num_4_4}')
	    print(f'3/4: {num_3_4}')
	    print(f'6/8: {num_6_8}')
	    print(f'Others: {num_other_ts}')
	    print(f'Multi: {num_multi_ts}')
	    print(f'Min and Max Pitch: {min_pitch, max_pitch}') 
	    print(f'Num files with complex notes: {num_fast_note_files}')
	    return num_notes, pitch_dist, dur_dist

	@staticmethod	
	def get_title(tune_filepath):
	    for line in open(tune_filepath):
	        if line[:2] == 'T:':
	            return line[2:]
	    return None
	
	@staticmethod
	def tune_contains_chords(tune_filepath):
	    # todo write it correctly
	    for line in open(tune_filepath):
	        if '"' in line:
	            return True
	    return False

	@staticmethod
	def tune_is_multivoice(tune_filepath):
	    for line in open(tune_filepath):
	        if line[:3] == 'V:2':
	            return True
	        if line[:4] == 'V: 2':
	            return True
	        if line[:4] == 'V :2':
	            return True
	        if line[:5] == 'V : 2':
	            return True
	    return False