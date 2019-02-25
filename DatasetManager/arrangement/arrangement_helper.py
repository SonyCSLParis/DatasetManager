import math
import shutil

import torch
import numpy as np
import os
import glob
import re
import music21


def note_to_midiPitch(note):
    """
    music21 note to number
    :param note:
    :return:
    """
    # +1 on octave is needed to obtain midi pitch
    octave = (note.octave + 1)
    pc = note.pitch.pitchClass
    return octave * 12 + pc


def midiPitch_to_octave_pc(number):
    """
    number to pc octave decomposition
    :param note:
    :return:
    """
    octave = number // 12
    pitch_class = number % 12
    return octave, pitch_class


def orchestral_tensor_to_pianoroll(tensor):
    #  A ECRIRE
    # Dict containing pianoroll frame for each instrument
    pianoroll_frame = {}
    return pianoroll_frame


def quantize_and_filter_music21_element(element, subdivision):
    frame_start = int(round(element.offset * subdivision))
    if abs((element.offset * subdivision) - frame_start) > 0.1:
        # Avoid elements not on fixed subdivision of quarter notes
        return None, None
    frame_end = int(round((element.offset + element.duration.quarterLength)*subdivision))
    if frame_start == frame_end:
        #TODO What do we do with very short events ?
        # Perhaps keep them...
        return frame_start, frame_end+1
    return frame_start, frame_end


def score_to_pianoroll(score, subdivision, simplify_instrumentation, transpose_to_sounding_pitch=False):
    # Transpose the score at sounding pitch. Simplify when transposing instruments are in the score
    if transpose_to_sounding_pitch:
        score_soundingPitch = score.toSoundingPitch()
    else:
        score_soundingPitch = score
    # Get start/end offsets
    start_offset = int(score.flat.lowestOffset)
    end_offset = 1 + int(score.flat.highestOffset)
    # Output
    pianoroll = dict()
    number_frames = (end_offset - start_offset) * subdivision
    for part in score_soundingPitch.parts:
        # Parse file
        elements_iterator = part.flat.getElementsByOffset(start_offset, end_offset,
                                                          classList=[music21.note.Note,
                                                                     music21.chord.Chord])
        this_pr = np.zeros((number_frames, 128))

        def add_note_to_pianoroll(note, note_start, note_end, pr):
            note_velocity = note.volume.velocity
            if note_velocity is None:
                note_velocity = 128
            note_pitch = note_to_midiPitch(note)

            pr[note_start:note_end, note_pitch] = note_velocity

        for element in elements_iterator:
            # Start at stop at previous frame. Problem: we loose too short events
            note_start, note_end = quantize_and_filter_music21_element(element, subdivision)
            if note_start is None:
                continue
            if element.isChord:
                for note in element._notes:
                    add_note_to_pianoroll(note, note_start, note_end, this_pr)
            else:
                add_note_to_pianoroll(element, note_start, note_end, this_pr)

        # Sometimes, typically for truncated files or when thick subdivisions are used
        # We might end up with instrument pr only files with zeros.
        # We ignore them
        if this_pr.sum() == 0:
            continue

        # Instrument name
        instrument_names = separate_instruments_names(simplify_instrumentation[part.partName])
        for instrument_name in instrument_names:
            if instrument_name in pianoroll.keys():
                pianoroll[instrument_name] = np.maximum(pianoroll[instrument_name], this_pr)
            else:
                pianoroll[instrument_name] = this_pr

    return pianoroll, number_frames


def pianoroll_to_score(pianoroll):
    score = None
    return score


def separate_instruments_names(instrument_names):
    return re.split(' and ', instrument_names)


class ArrangementIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    # todo redo
    def __init__(self, arrangement_path, subsets, num_elements=None):
        self.arrangement_path = arrangement_path  # Root of the database
        self.subsets = subsets
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.arrangement_generator()
        )
        return it

    def arrangement_generator(self):
        arrangement_paths = []
        for subset in self.subsets:
            # Should return pairs of files
            arrangement_paths += (glob.glob(
                os.path.join(self.arrangement_path, subset, '[0-9]*')))
        if self.num_elements is not None:
            arrangement_paths = arrangement_paths[:self.num_elements]
        for arrangement_path in arrangement_paths:
            try:
                xml_files = glob.glob(arrangement_path + '/*.xml')
                midi_files = glob.glob(arrangement_path + '/*.mid')
                if not ((len(xml_files) == 2) != (len(midi_files) == 2)):
                    raise Exception(f'There should be 2 midi or xml files in {arrangement_path}')
                if len(xml_files) == 2:
                    music_files = xml_files
                else:
                    music_files = midi_files
                print(music_files)
                # Here parse files and return as a dict containing matrices for piano and orchestra
                # arrangement_pair = process(xml_files)
                arrangement_pair = music21.converter.parse(music_files[0]), \
                                   music21.converter.parse(music_files[1])
                yield (arrangement_pair[0], arrangement_pair[1])
            except Exception as e:
                print(f'{music_files} is not parsable')
                print(e)
