import numpy as np
import os
import glob
import re
import music21
from DatasetManager.helpers import MAX_VELOCITY


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
    # Not used...
    pianoroll_frame = {}
    return pianoroll_frame


def quantize_and_filter_music21_element(element, subdivision):
    frame_start = int(round(element.offset * subdivision))
    if abs((element.offset * subdivision) - frame_start) > 0.1:
        # Avoid elements not on fixed subdivision of quarter notes
        return None, None
    frame_end = int(round((element.offset + element.duration.quarterLength) * subdivision))
    if frame_start == frame_end:
        # TODO What do we do with very short events ?
        # Perhaps keep them...
        return frame_start, frame_end + 1
    return frame_start, frame_end


def quantize_velocity_pianoroll_frame(frame, velocity_quantization):
    # This formula maps 0 -> 0
    # Then everything above 0 is mapped in [1, q-1] "uniformly"
    # resulting in q bins
    quantized_piano_frame = np.ceil((frame / MAX_VELOCITY) * (velocity_quantization - 1))
    return quantized_piano_frame


def unquantize_velocity(q_vel, velocity_quantization):
    u_vel = (q_vel / (velocity_quantization-1)) * (MAX_VELOCITY - 1)
    return int(u_vel)


def shift_pr_along_pitch_axis(matrix, shift):
    ret = np.zeros_like(matrix)
    if shift < 0:
        ret[:, :shift] = matrix[:, -shift:]
    elif shift > 0:
        ret[:, shift:] = matrix[:, :-shift]
    else:
        ret = matrix
    return ret


def flatten_dict_pr(dic):
    matrix = None
    for _, mat in dic.items():
        if matrix is None:
            matrix = mat
        else:
            matrix = matrix + mat
    return matrix


def new_events(pr_dict, onsets_dict):
    #  Sum them in one matrix
    pr = flatten_dict_pr(pr_dict)
    onsets = flatten_dict_pr(onsets_dict)

    #  Get new events indices (diff matrices)
    delta_flat = (np.abs(pr[1:] - pr[:-1])).sum(1)
    new_events_indices = list(np.where(delta_flat != 0)[0] + 1)
    # Note that this actually gives us the end of events, which is okay (miss first event and add a last note_off event)
    onsets_flat = onsets.sum(1)
    repeated_event_indices = list(np.where(onsets_flat > 0)[0])

    events = sorted(list(set(repeated_event_indices + new_events_indices)))
    return events


def score_to_pianoroll(score, subdivision, simplify_instrumentation,
                       instrument_grouping, transpose_to_sounding_pitch):
    # TODO COmpute also duration matrix
    # Transpose the score at sounding pitch. Simplify when transposing instruments are in the score
    if transpose_to_sounding_pitch:
        score_soundingPitch = score.toSoundingPitch()
    else:
        score_soundingPitch = score
    # Get start/end offsets
    start_offset = int(score.flat.lowestOffset)
    end_offset = 1 + int(score.flat.highestTime)
    # Output
    pianoroll = dict()
    onsets = dict()
    number_frames = (end_offset - start_offset) * subdivision
    for part in score_soundingPitch.parts:

        elements_iterator = part.flat.notes

        this_pr = np.zeros((number_frames, 128))
        this_onsets = np.zeros((number_frames, 128))

        def add_note_to_pianoroll(note, note_start, note_end, pr, onsets):
            note_velocity = note.volume.velocity
            if note_velocity is None:
                note_velocity = 128
            note_pitch = note_to_midiPitch(note)

            pr[note_start:note_end, note_pitch] = note_velocity

            if (note.tie is None) or (note.tie.type == 'start'):
                onsets[note_start, note_pitch] = 1
            # else:
            #     if :
            #         onsets[note_start, note_pitch] = 1
            return

        for element in elements_iterator:
            # Start at stop at previous frame. Problem: we loose too short events
            note_start, note_end = quantize_and_filter_music21_element(element, subdivision)

            if note_start is None:
                continue

            if element.isChord:
                for note in element._notes:
                    add_note_to_pianoroll(note, note_start, note_end, this_pr, this_onsets)
            else:
                add_note_to_pianoroll(element, note_start, note_end, this_pr, this_onsets)

        # Sometimes, typically for truncated files or when thick subdivisions are used
        # We might end up with instrument pr only files with zeros.
        # We ignore them
        if this_pr.sum() == 0:
            continue

        # print(part.partName)
        # Instrument name
        if simplify_instrumentation is None:
            instrument_names = ["Piano"]
        else:
            instrument_names = [instrument_grouping[e] for e in separate_instruments_names(simplify_instrumentation[part.partName])]

        for instrument_name in instrument_names:
            if instrument_name in pianoroll.keys():
                pianoroll[instrument_name] = np.maximum(pianoroll[instrument_name], this_pr)
                onsets[instrument_name] = np.maximum(onsets[instrument_name], this_onsets)
            else:
                pianoroll[instrument_name] = this_pr
                onsets[instrument_name] = this_onsets

    return pianoroll, onsets, number_frames


def pianoroll_to_score(pianoroll):
    score = None
    return score


def separate_instruments_names(instrument_names):
    return re.split(' and ', instrument_names)


def list_instru_score(score):
    list_instru = []
    for part in score.parts:
        list_instru.append(part.partName)
    return list_instru


def sort_arrangement_pairs(arrangement_pair):
    # Find which score is piano and which is orchestral
    if len(list_instru_score(arrangement_pair[0])) > len(list_instru_score(arrangement_pair[1])):
        return {'Orchestra': arrangement_pair[0], 'Piano': arrangement_pair[1]}
    elif len(list_instru_score(arrangement_pair[0])) < len(list_instru_score(arrangement_pair[1])):
        return {'Piano': arrangement_pair[0], 'Orchestra': arrangement_pair[1]}
    else:
        print(f'# SKIP!!')
        return None


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
                arr_pair = sort_arrangement_pairs(arrangement_pair)

                name = '-'.join(re.split('/', arrangement_path)[-2:])
                arr_pair['name'] = name
                yield arr_pair
            except Exception as e:
                print(f'{music_files} is not parsable')
                print(e)


class OrchestraIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    # todo redo
    def __init__(self, folder_path, subsets, process_file):
        self.folder_path = folder_path  # Root of the database
        self.process_file = process_file
        self.subsets = subsets

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.generator()
        )
        return it

    def generator(self):

        folder_paths = []
        for subset in self.subsets:
            folder_paths += glob.glob(f'{self.folder_path}/{subset}/**')

        for folder_path in folder_paths:
            xml_files = glob.glob(folder_path + '/*.xml')
            midi_files = glob.glob(folder_path + '/*.mid')
            if len(xml_files) == 1:
                music_files = xml_files
            elif len(midi_files) == 1:
                music_files = midi_files
            else:
                raise Exception(f"No or too much files in {folder_path}")
            print(music_files)
            # Here parse files and return as a dict containing matrices for piano and orchestra
            if self.process_file:
                ret = music21.converter.parse(music_files[0])
            else:
                ret = music_files[0]

            name = '-'.join(re.split('/', folder_path)[-2:])

            yield {'Piano': None, 'Orchestra': ret, 'name': name}
