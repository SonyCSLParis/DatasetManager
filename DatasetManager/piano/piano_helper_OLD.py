import collections
import copy
import glob
import itertools
import os

import numpy as np
from pretty_midi import PrettyMIDI, Note, Instrument

# ==================================================================================
# Parameters
# ==================================================================================

# NoteSeq -------------------------------------------------------------------------

DEFAULT_SAVING_PROGRAM = 1
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)
DEFAULT_VELOCITY_RANGE = range(21, 109)
DEFAULT_NORMALIZATION_BASELINE = 60  # C4

# EventSeq ------------------------------------------------------------------------

USE_VELOCITY = True
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65
DEFAULT_VELOCITY_STEPS = 32
# DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
DEFAULT_NOTE_LENGTH = BEAT_LENGTH / 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

# ControlSeq ----------------------------------------------------------------------

DEFAULT_WINDOW_SIZE = BEAT_LENGTH * 4
DEFAULT_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1


class PianoIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    def __init__(self, subsets, num_elements=None):
        self.path = f'{os.path.expanduser("~")}/Data/databases/Piano'
        self.subsets = subsets
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.generator()
        )
        return it

    def generator(self):
        midi_files = []
        for subset in self.subsets:
            # Should return pairs of files
            midi_files += (glob.glob(
                os.path.join(self.path, subset, '*.mid')))
            midi_files += (glob.glob(
                os.path.join(self.path, subset, '*.MID')))
        if self.num_elements is not None:
            midi_files = midi_files[:self.num_elements]
        for midi_file in midi_files:
            print(midi_file)
            yield midi_file


#  These come from Github "Performance RNN - PyTorch"
# https://github.com/djosix/Performance-RNN-PyTorch
def preprocess_midi(path, excluded_features, insert_zero_time_token):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq, excluded_features, insert_zero_time_token)
    ret = event_seq.to_array(excluded_features, insert_zero_time_token)
    return ret


# ==================================================================================
# Notes
# ==================================================================================
class NoteSeq:

    @staticmethod
    def from_midi(midi, programs=DEFAULT_LOADING_PROGRAMS):
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        return NoteSeq(list(notes))

    @staticmethod
    def from_midi_file(path, *kargs, **kwargs):
        midi = PrettyMIDI(path)
        return NoteSeq.from_midi(midi, *kargs, **kwargs)

    @staticmethod
    def merge(*note_seqs):
        notes = itertools.chain(*[seq.notes for seq in note_seqs])
        return NoteSeq(list(notes))

    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                assert isinstance(note, Note)
            notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))

    def copy(self):
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
        midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        inst = Instrument(program, False, 'NoteSeq')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *kargs, **kwargs):
        self.to_midi(*kargs, **kwargs).write(path)

    def add_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)

    def adjust_pitches(self, offset):
        for note in self.notes:
            pitch = note.pitch + offset
            pitch = 0 if pitch < 0 else pitch
            pitch = 127 if pitch > 127 else pitch
            note.pitch = pitch

    def adjust_velocities(self, offset):
        for note in self.notes:
            velocity = note.velocity + offset
            velocity = 0 if velocity < 0 else velocity
            velocity = 127 if velocity > 127 else velocity
            note.velocity = velocity

    def adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
        last_notes = {}
        for i, note in enumerate(self.notes):
            if note.pitch in last_notes:
                last_note = last_notes[note.pitch]
                if note.start - last_note.start <= min_interval:
                    last_note.end = max(note.end, last_note.end)
                    last_note.velocity = max(note.velocity, last_note.velocity)
                    del self.notes[i]
                elif note.start < last_note.end:
                    last_note.end = note.start
            else:
                last_notes[note.pitch] = note


# ==================================================================================
# Events
# ==================================================================================

class Event:

    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value

    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)


class EventSeq:
    pitch_range = DEFAULT_PITCH_RANGE
    velocity_range = DEFAULT_VELOCITY_RANGE
    velocity_steps = DEFAULT_VELOCITY_STEPS

    @staticmethod
    def time_shift_bins(insert_zero_time_token):

        #  From Chris:
        # ∆T
        # for 1–100 ticks (short) 1–100
        # ∆T
        # for 100–1000 ticks (medium) 101–190
        # ∆T
        # for > 10000 ticks (long) 191–370

        # Use long ??
        # smallest_time_shift = 0.001
        # if insert_zero_time_token:
        #     short_time_shifts = np.arange(0, 1.0, smallest_time_shift)
        # else:
        #     short_time_shifts = np.arange(smallest_time_shift, 1.0, smallest_time_shift)
        # medium_time_shifts = np.arange(1.0, 5.0, 5.0 * smallest_time_shift)
        # time_shift_bins = np.concatenate((short_time_shifts, medium_time_shifts))

        if insert_zero_time_token:
            time_shift_bins = np.arange(0, 5.0, 0.01)
        else:
            time_shift_bins = np.arange(0.01, 5.0, 0.01)

        return time_shift_bins

    @staticmethod
    def from_note_seq(note_seq, excluded_features, insert_zero_time_token):
        note_events = []

        if USE_VELOCITY:
            velocity_bins = EventSeq.get_velocity_bins()

        for note in note_seq.notes:
            if note.pitch in EventSeq.pitch_range:
                if USE_VELOCITY:
                    velocity = note.velocity
                    velocity = max(velocity, EventSeq.velocity_range.start)
                    velocity = min(velocity, EventSeq.velocity_range.stop - 1)
                    velocity_index = np.searchsorted(velocity_bins, velocity)
                    note_events.append(Event('velocity', note.start, velocity_index))

                pitch_index = note.pitch - EventSeq.pitch_range.start
                note_events.append(Event('note_on', note.start, pitch_index))
                note_events.append(Event('note_off', note.end, pitch_index))

        #  sort by time
        note_events.sort(key=lambda event: event.time)  # stable

        #  remove exlcuded types here
        note_events = [e for e in note_events if e.type not in excluded_features]

        events = []

        for i, event in enumerate(note_events):
            events.append(event)

            if event is note_events[-1]:
                break

            interval = note_events[i + 1].time - event.time

            # shift = 0
            # if insert_zero_time_token:
            #     minimum_time_interval = EventSeq.time_shift_bins(insert_zero_time_token)[1]
            # else:
            #     minimum_time_interval = EventSeq.time_shift_bins(insert_zero_time_token)[0]
            # first = True
            # while interval - shift >= minimum_time_interval:
            #     if not first:
            #         print(interval)
            #     index = np.searchsorted(EventSeq.time_shift_bins(insert_zero_time_token),
            #                             interval - shift, side='right') - 1
            #     events.append(Event('time_shift', event.time + shift, index))
            #     shift += EventSeq.time_shift_bins(insert_zero_time_token)[index]
            #     first = False

            # if interval > EventSeq.time_shift_bins(insert_zero_time_token)[-1]:
            #     print('yoyoyo')

            index = np.searchsorted(EventSeq.time_shift_bins(insert_zero_time_token),
                                    interval, side='right') - 1

            if not insert_zero_time_token and index == -1:
                continue

            events.append(Event('time_shift', event.time, index))

        return EventSeq(events, insert_zero_time_token)

    @staticmethod
    def from_array(event_indeces, excluded_features, insert_zero_time_token):
        # notes: old original version
        #  
        #     time = 0
        #     events = []
        #     for event_index in event_indeces:
        #         for event_type, feat_range in EventSeq.feat_ranges().items():
        #             if feat_range.start <= event_index < feat_range.stop:
        #                 event_value = event_index - feat_range.start
        #                 events.append(Event(event_type, time, event_value))
        #                 if event_type == 'time_shift':
        #                     time += EventSeq.time_shift_bins[event_value]
        #                 break
        #
        #     return EventSeq(events)

        time = 0
        events = []
        first_note_on_encountered = False
        for event_index in event_indeces:
            for event_type, feat_range in EventSeq.feat_ranges(excluded_features, insert_zero_time_token).items():
                if feat_range.start <= event_index < feat_range.stop:
                    if not first_note_on_encountered:
                        if event_type == 'note_on':
                            first_note_on_encountered = True
                        elif event_type not in ['velocity', 'note_on']:
                            break
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += EventSeq.time_shift_bins(insert_zero_time_token)[event_value]
                    break

        return EventSeq(events, insert_zero_time_token)

    @staticmethod
    def dim(insert_zero_time_token):
        return sum(EventSeq.feat_dims(insert_zero_time_token).values())

    @staticmethod
    def feat_dims(insert_zero_time_token):
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(EventSeq.pitch_range)
        feat_dims['note_off'] = len(EventSeq.pitch_range)
        if USE_VELOCITY:
            feat_dims['velocity'] = EventSeq.velocity_steps
        feat_dims['time_shift'] = len(EventSeq.time_shift_bins(insert_zero_time_token))
        return feat_dims

    @staticmethod
    def feat_ranges(excluded_features, insert_zero_time_token):
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in EventSeq.feat_dims(insert_zero_time_token).items():
            if feat_name in excluded_features:
                continue
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
        n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
        return np.arange(
            EventSeq.velocity_range.start,
            EventSeq.velocity_range.stop,
            n / (EventSeq.velocity_steps - 1))

    def __init__(self, events, insert_zero_time_token):
        for event in events:
            assert isinstance(event, Event)

        self.events = copy.deepcopy(events)

        # compute event times again
        time = 0
        for event in self.events:
            event.time = time
            if event.type == 'time_shift':
                time += EventSeq.time_shift_bins(insert_zero_time_token)[event.value]

    def to_note_seq(self, insert_zero_time_token):
        time = 0
        notes = []

        velocity = DEFAULT_VELOCITY
        velocity_bins = EventSeq.get_velocity_bins()

        last_notes = {}

        for event in self.events:
            if event.type == 'note_on':
                pitch = event.value + EventSeq.pitch_range.start
                note = Note(velocity, pitch, time, None)
                notes.append(note)
                last_notes[pitch] = note

            elif event.type == 'note_off':
                pitch = event.value + EventSeq.pitch_range.start

                # If note not in last_notes, it is just removed
                if pitch in last_notes:
                    note = last_notes[pitch]
                    note.end = max(time, note.start + MIN_NOTE_LENGTH)
                    del last_notes[pitch]

            elif event.type == 'velocity':
                index = min(event.value, velocity_bins.size - 1)
                velocity = velocity_bins[index]

            elif event.type == 'time_shift':
                time += EventSeq.time_shift_bins(insert_zero_time_token)[event.value]

        for note in notes:
            if note.end is None:
                note.end = note.start + DEFAULT_NOTE_LENGTH

            note.velocity = int(note.velocity)

        return NoteSeq(notes)

    def to_array(self, excluded_features, insert_zero_time_token):
        feat_idxs = EventSeq.feat_ranges(excluded_features=excluded_features,
                                         insert_zero_time_token=insert_zero_time_token)
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if EventSeq.dim(insert_zero_time_token) <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)


# ==================================================================================
# Controls
# ==================================================================================

class Control:

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram  # list
        self.note_density = note_density  # int

    def __repr__(self):
        return 'Control(pitch_histogram={}, note_density={})'.format(
            self.pitch_histogram, self.note_density)

    def to_array(self):
        feat_dims = ControlSeq.feat_dims()
        ndens = np.zeros([feat_dims['note_density']])
        ndens[self.note_density] = 1.  # [dens_dim]
        phist = np.array(self.pitch_histogram)  # [hist_dim]
        return np.concatenate([ndens, phist], 0)  # [dens_dim + hist_dim]


class ControlSeq:
    note_density_bins = DEFAULT_NOTE_DENSITY_BINS
    window_size = DEFAULT_WINDOW_SIZE

    @staticmethod
    def from_event_seq(event_seq):
        events = list(event_seq.events)
        start, end = 0, 0

        pitch_count = np.zeros([12])
        note_count = 0

        controls = []

        def _rel_pitch(pitch):
            return (pitch - 24) % 12

        for i, event in enumerate(events):

            while start < i:
                if events[start].type == 'note_on':
                    abs_pitch = events[start].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] -= 1.
                    note_count -= 1.
                start += 1

            while end < len(events):
                if events[end].time - event.time > ControlSeq.window_size:
                    break
                if events[end].type == 'note_on':
                    abs_pitch = events[end].value + EventSeq.pitch_range.start
                    rel_pitch = _rel_pitch(abs_pitch)
                    pitch_count[rel_pitch] += 1.
                    note_count += 1.
                end += 1

            pitch_histogram = (
                pitch_count / note_count
                if note_count
                else np.ones([12]) / 12
            ).tolist()

            note_density = max(np.searchsorted(
                ControlSeq.note_density_bins,
                note_count, side='right') - 1, 0)

            controls.append(Control(pitch_histogram, note_density))

        return ControlSeq(controls)

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        note_density_dim = len(ControlSeq.note_density_bins)
        return collections.OrderedDict([
            ('pitch_histogram', 12),
            ('note_density', note_density_dim)
        ])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def recover_compressed_array(array):
        feat_dims = ControlSeq.feat_dims()
        assert array.shape[1] == 1 + feat_dims['pitch_histogram']
        ndens = np.zeros([array.shape[0], feat_dims['note_density']])
        ndens[np.arange(array.shape[0]), array[:, 0]] = 1.  # [steps, dens_dim]
        phist = array[:, 1:].astype(np.float64) / 255  # [steps, hist_dim]
        return np.concatenate([ndens, phist], 1)  # [steps, dens_dim + hist_dim]

    def __init__(self, controls):
        for control in controls:
            assert isinstance(control, Control)
        self.controls = copy.deepcopy(controls)

    def to_compressed_array(self):
        ndens = [control.note_density for control in self.controls]
        ndens = np.array(ndens, dtype=np.uint8).reshape(-1, 1)
        phist = [control.pitch_histogram for control in self.controls]
        phist = (np.array(phist) * 255).astype(np.uint8)
        return np.concatenate([
            ndens,  # [steps, 1] density index
            phist  # [steps, hist_dim] 0-255
        ], 1)  # [steps, hist_dim + 1]


def get_midi_type(midi, midi_ranges):
    for feat_name, feat_range in midi_ranges.items():
        if midi in feat_range:
            midi_type = feat_name
            return midi_type


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


if __name__ == '__main__':
    import pickle
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/midi/ecomp_piano_dataset/BLINOV02.mid'

    print('Converting MIDI to EventSeq')
    note_seq = NoteSeq.from_midi_file(path)
    es = EventSeq.from_note_seq(note_seq)

    print('Converting EventSeq to MIDI')
    EventSeq.from_array(es.to_array()).to_note_seq().to_midi_file('/tmp/test.mid')

    print('Converting EventSeq to ControlSeq')
    cs = ControlSeq.from_event_seq(es)

    print('Saving compressed ControlSeq')
    pickle.dump(cs.to_compressed_array(), open('/tmp/cs-compressed.data', 'wb'))

    print('Loading compressed ControlSeq')
    c = ControlSeq.recover_compressed_array(pickle.load(open('/tmp/cs-compressed.data', 'rb')))

    print('Done')

