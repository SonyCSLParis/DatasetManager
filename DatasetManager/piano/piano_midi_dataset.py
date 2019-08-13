import json
import os
import pickle
import re
import shutil

import music21
import numpy as np
import torch
from tqdm import tqdm

import DatasetManager
from DatasetManager.arrangement.arrangement_helper import shift_pr_along_pitch_axis, note_to_midiPitch, \
    score_to_pianoroll, new_events
from DatasetManager.config import get_config
from DatasetManager.helpers import REST_SYMBOL, SLUR_SYMBOL, END_SYMBOL, START_SYMBOL, \
    YES_SYMBOL, NO_SYMBOL, TIME_SHIFT, PAD_SYMBOL, STOP_SYMBOL, MASK_SYMBOL
from DatasetManager.music_dataset import MusicDataset

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class PianoMidipianoDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 mean_number_messages_per_time_frame,
                 subdivision,
                 sequence_size,
                 max_transposition,
                 integrate_discretization):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()

        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.subdivision = subdivision  # We use only on beats notes so far
        assert sequence_size % 2 == 1
        self.sequence_size = sequence_size
        self.velocity_quantization = velocity_quantization
        self.max_transposition = max_transposition
        self.integrate_discretization = integrate_discretization

        config = get_config()

        arrangement_path = config["arrangement_path"]
        reference_tessitura_path = f'{arrangement_path}/reference_tessitura.json'
        self.dump_folder = config["dump_folder"]

        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = tessitura['Piano']

        # Only piano is overwritten
        self.max_number_messages_piano = self.sequence_size * mean_number_messages_per_time_frame
        self.smallest_padding_length = self.max_number_messages_piano
        self.symbol2index = {}
        self.index2symbol = {}

        self.precomputed_vectors_piano = {
            START_SYMBOL: None,
            END_SYMBOL: None,
            PAD_SYMBOL: None,
            MASK_SYMBOL: None,
            REST_SYMBOL: None,
        }

        return

    def __repr__(self):
        name = f'PianosMidipianoDataset-' \
            f'{self.name}-' \
            f'{self.subdivision}-' \
            f'{self.sequence_size}-' \
            f'{self.velocity_quantization}-' \
            f'{self.max_transposition}'
        return name

    def iterator_gen(self):
        return (score for score in self.corpus_it_gen())

    def compute_index_dicts(self, index_dict_path):
        lowest_note, highest_note = self.reference_tessitura['Piano']
        lowest_pitch = note_to_midiPitch(lowest_note)
        highest_pitch = note_to_midiPitch(highest_note)
        list_midiPitch = sorted(list(range(lowest_pitch, highest_pitch + 1)))
        list_velocity = list(range(self.velocity_quantization))
        list_duration =

        # From performance rnn
        # 100 duration from 10ms to 1s
        # 32 vellocities
        # dimension = 388
        # 30 seconds ~ 1200 frames

        index = 0
        for midi_pitch in list_midiPitch:
            for velocity in list_velocity:
                symbol = f'{midi_pitch}_{velocity}'
                self.symbol2index[symbol] = index
                self.index2symbol[index] = symbol
                index += 1

        # TODO: build mapping duration


        # Mask (for nade like inference schemes)
        self.symbol2index[MASK_SYMBOL] = index
        self.index2symbol[index] = MASK_SYMBOL
        index += 1
        # Pad
        self.symbol2index[PAD_SYMBOL] = index
        self.index2symbol[index] = PAD_SYMBOL
        index += 1
        # Start
        self.symbol2index[START_SYMBOL] = index
        self.index2symbol[index] = START_SYMBOL
        index += 1
        # End
        self.symbol2index[END_SYMBOL] = index
        self.index2symbol[index] = END_SYMBOL
        index += 1
        return

    # def make_tensor_dataset(self, frame_orchestra=None):

    # def get_score_tensor(self, scores, offsets):

    # def transposed_score_and_metadata_tensors(self, score, semi_tone):

    # def get_metadata_tensor(self, score):

    # def score_to_list_pc(self, score, datatype):

    # def align_score(self, piano_score, orchestra_score):

    # def get_allowed_transpositions_from_pr(self, pr, frames, instrument_name):

    # def prepare_chunk_from_corresponding_frames(self, corresponding_frames):

    def transpose_loop_iteration(self, pianorolls_pair, onsets_pair, transposition_semi_tone,
                                 chunks_piano_indices, chunks_orchestra_indices,
                                 minimum_transposition_allowed, maximum_transposition_allowed,
                                 piano_tensor_dataset, orchestra_tensor_dataset,
                                 orchestra_instruments_presence_tensor_dataset,
                                 total_chunk_counter, too_many_instruments_frame, impossible_transposition):

        ############################################################
        # Transpose pianorolls
        this_pr_piano = shift_pr_along_pitch_axis(pianorolls_pair["Piano"], transposition_semi_tone)
        this_onsets_piano = shift_pr_along_pitch_axis(onsets_pair["Piano"], transposition_semi_tone)

        this_pr_orchestra = {}
        this_onsets_orchestra = {}
        for instrument_name in pianorolls_pair["Orchestra"].keys():
            # Pr
            pr = pianorolls_pair["Orchestra"][instrument_name]
            shifted_pr = shift_pr_along_pitch_axis(pr, transposition_semi_tone)
            this_pr_orchestra[instrument_name] = shifted_pr
            # Onsets
            onsets = onsets_pair["Orchestra"][instrument_name]
            shifted_onsets = shift_pr_along_pitch_axis(onsets, transposition_semi_tone)
            this_onsets_orchestra[instrument_name] = shifted_onsets
        ############################################################

        if minimum_transposition_allowed is None:
            if transposition_semi_tone != 0:
                raise Exception("Possible transpositions should be computed on non transposed pianorolls")
            # We have to construct the possible transpose
            build_allowed_transposition_flag = True
            minimum_transposition_allowed = []
            maximum_transposition_allowed = []
        else:
            build_allowed_transposition_flag = False

        for chunk_index in range(len(chunks_piano_indices)):
            this_chunk_piano_indices = chunks_piano_indices[chunk_index]
            this_chunk_orchestra_indices = chunks_orchestra_indices[chunk_index]
            avoid_this_chunk = False
            total_chunk_counter += 1

            ############################################################
            #  Get list of allowed transpositions
            if build_allowed_transposition_flag:
                min_transposition = -self.max_transposition
                max_transposition = self.max_transposition

                # Observe tessitura for each instrument for this chunk. Use non transposed pr of course
                this_min_transposition, this_max_transposition = \
                    self.get_allowed_transpositions_from_pr(this_pr_piano,
                                                            this_chunk_piano_indices,
                                                            "Piano")

                if (this_min_transposition is None) or (this_max_transposition is None):
                    min_transposition = None
                    max_transposition = None
                else:
                    min_transposition = max(this_min_transposition, min_transposition)
                    max_transposition = min(this_max_transposition, max_transposition)

                # Use reference tessitura or compute tessitura directly on the files ?
                if min_transposition is not None:
                    for instrument_name, pr in this_pr_orchestra.items():
                        this_min_transposition, this_max_transposition = \
                            self.get_allowed_transpositions_from_pr(pr,
                                                                    this_chunk_orchestra_indices,
                                                                    instrument_name)
                        if this_min_transposition is not None:  # If instrument not in this chunk, None was returned
                            min_transposition = max(this_min_transposition, min_transposition)
                            max_transposition = min(this_max_transposition, max_transposition)

                    this_minimum_transposition_allowed = min(0, min_transposition)
                    this_maximum_transposition_allowed = max(0, max_transposition)
                else:
                    this_minimum_transposition_allowed = None
                    this_maximum_transposition_allowed = None
                minimum_transposition_allowed.append(this_minimum_transposition_allowed)
                maximum_transposition_allowed.append(this_maximum_transposition_allowed)
            else:
                this_minimum_transposition_allowed = minimum_transposition_allowed[chunk_index]
                this_maximum_transposition_allowed = maximum_transposition_allowed[chunk_index]

            ############################################################
            #  Test if the transposition is possible
            if (this_maximum_transposition_allowed is None) or (this_minimum_transposition_allowed is None):
                impossible_transposition += 1
                continue
            if (this_minimum_transposition_allowed > transposition_semi_tone) \
                    or (this_maximum_transposition_allowed < transposition_semi_tone):
                impossible_transposition += 1
                continue

            ############################################################
            #  Extract representations
            local_piano_tensor = []
            local_orchestra_tensor = []
            local_orchestra_instruments_presence_tensor = []
            previous_notes_orchestra = None
            for index_frame in range(len(this_chunk_piano_indices)):
                if index_frame != 0:
                    previous_frame_piano = this_chunk_piano_indices[index_frame - 1]
                    if previous_frame_piano in [START_SYMBOL, END_SYMBOL]:
                        previous_frame_piano = None
                else:
                    previous_frame_piano = None
                frame_piano = this_chunk_piano_indices[index_frame]
                frame_orchestra = this_chunk_orchestra_indices[index_frame]

                if frame_orchestra in [START_SYMBOL, MASK_SYMBOL, PAD_SYMBOL, END_SYMBOL]:
                    if frame_piano == END_SYMBOL:
                        #  Remove TS (after asserting there is one...)
                        assert local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT], "Weirdness"
                        local_piano_tensor.pop()
                        # Add end symbol, then the stop symbol
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                        local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])
                    else:
                        local_piano_tensor.append(self.symbol2index_piano[frame_piano])
                    orchestra_t_encoded = self.precomputed_vectors_orchestra[frame_orchestra].clone().detach()
                    orchestra_instruments_presence_t_encoded = self.precomputed_vectors_orchestra_instruments_presence[
                        PAD_SYMBOL].clone().detach()
                else:
                    local_piano_tensor = self.pianoroll_to_piano_tensor(
                        pr=this_pr_piano,
                        onsets=this_onsets_piano,
                        previous_frame_index=previous_frame_piano,
                        frame_index=frame_piano,
                        piano_vector=local_piano_tensor)

                    #  Time shift piano
                    local_piano_tensor.append(self.symbol2index_piano[TIME_SHIFT])

                    orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = self.pianoroll_to_orchestral_tensor(
                        this_pr_orchestra,
                        this_onsets_orchestra,
                        previous_notes=previous_notes_orchestra,
                        frame_index=frame_orchestra)

                if orchestra_t_encoded is None:
                    avoid_this_chunk = True
                    break

                local_orchestra_tensor.append(orchestra_t_encoded)
                local_orchestra_instruments_presence_tensor.append(orchestra_instruments_presence_t_encoded)

            # If last symbol is not a PAD  Replace the last TIME_SHIFT added by a STOP
            if local_piano_tensor[-1] == self.symbol2index_piano[TIME_SHIFT]:
                local_piano_tensor.pop()
                local_piano_tensor.append(self.symbol2index_piano[STOP_SYMBOL])

            #  Pad piano vector to max length
            padding_length = self.max_number_messages_piano - len(local_piano_tensor)
            if padding_length < 0:
                raise Exception(
                    f"Padding length is equal to {padding_length}, consider increasing the max length of the encoding vector")
            self.smallest_padding_length = min(self.smallest_padding_length, padding_length)
            for _ in range(padding_length):
                local_piano_tensor.append(self.symbol2index_piano[PAD_SYMBOL])
            ############################################################

            if avoid_this_chunk:
                too_many_instruments_frame += 1
                continue

            assert len(local_piano_tensor) == self.max_number_messages_piano
            assert len(local_orchestra_tensor) == self.sequence_size

            local_piano_tensor = torch.LongTensor(local_piano_tensor)
            local_orchestra_tensor = torch.stack(local_orchestra_tensor)
            local_orchestra_instruments_presence_tensor = torch.stack(local_orchestra_instruments_presence_tensor)

            piano_tensor_dataset.append(
                local_piano_tensor[None, :].int())
            orchestra_tensor_dataset.append(
                local_orchestra_tensor[None, :, :].int())
            orchestra_instruments_presence_tensor_dataset.append(
                local_orchestra_instruments_presence_tensor[None, :, :].int())

        return minimum_transposition_allowed, maximum_transposition_allowed, \
               piano_tensor_dataset, orchestra_tensor_dataset, orchestra_instruments_presence_tensor_dataset, \
               total_chunk_counter, too_many_instruments_frame, impossible_transposition

    def pianoroll_to_piano_tensor(self, pr, onsets, previous_frame_index, frame_index, piano_vector):

        notes = [(e, True) for e in list(np.where(onsets[frame_index])[0])]
        if previous_frame_index is not None:
            delta_pr = pr[previous_frame_index] - pr[frame_index]
            notes += [(e, False) for e in list(np.where(delta_pr)[0])]

        # Sort notes from lowest to highest
        #  Todo check register individually for each voice for smaller categorical representations
        notes = sorted(notes, key=lambda e: e[0])

        #  Build symbols and append corresponding index in midi_messages
        previous_note = None
        for (note, on) in notes:
            #  Filter out note_off for repeated notes (uselessely crowds representation)
            if (previous_note == note) and (not on):
                continue
            if on:
                symbol = str(note) + '_on'
            else:
                symbol = str(note) + '_off'

            if symbol in self.symbol2index_piano.keys():
                index = self.symbol2index_piano[symbol]
            else:
                print(f'OOR: Piano - {symbol}')
                continue
            piano_vector.append(index)
            previous_note = note
        return piano_vector

    def init_generation_filepath(self, batch_size, context_length, filepath, banned_instruments=[],
                                 unknown_instruments=[], subdivision=None):

        ts_ind = self.symbol2index_piano[TIME_SHIFT]
        start_ind = self.symbol2index_piano[START_SYMBOL]
        pad_ind = self.symbol2index_piano[PAD_SYMBOL]
        end_ind = self.symbol2index_piano[END_SYMBOL]
        meta_inds = [ts_ind, start_ind, end_ind, pad_ind]
        stop_ind = self.symbol2index_piano[STOP_SYMBOL]

        # Get pianorolls
        score_piano = music21.converter.parse(filepath)

        if subdivision is None:
            subdivision = self.subdivision
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(
            score=score_piano,
            subdivision=self.subdivision,
            simplify_instrumentation=None,
            instrument_grouping=self.instrument_grouping,
            transpose_to_sounding_pitch=self.transpose_to_sounding_pitch,
            integrate_discretization=self.integrate_discretization,
            binarize=True
        )

        rhythm_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        piano_init_list = []
        previous_frame_index = None
        for frame_index in rhythm_piano:
            piano_init_list = self.pianoroll_to_piano_tensor(
                pr=pianoroll_piano["Piano"],
                onsets=onsets_piano,
                previous_frame_index=previous_frame_index,
                frame_index=frame_index,
                piano_vector=piano_init_list)
            previous_frame_index = frame_index

            #  Time shift piano
            piano_init_list.append(ts_ind)

        #  Remove the last time shift added
        piano_init_list.pop()

        # Prepend rests frames at the beginning and end of the piano score
        piano_init_list = [pad_ind] * (context_length - 1) + \
                          [start_ind] + \
                          piano_init_list + \
                          [end_ind] + \
                          [pad_ind] * (context_length - 1)

        # Use a non-sliced version for writing back the piano file
        piano_write = torch.tensor(piano_init_list).unsqueeze(0).repeat(batch_size, 1)

        #  Prepare chunks with padding for piano input
        def write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag):
            piano_init[chunk_index, :len(this_chunk)] = torch.tensor(this_chunk)
            if writing_stop_flag:
                piano_init[chunk_index, len(this_chunk)] = stop_ind
                piano_init[chunk_index, len(this_chunk) + 1:] = pad_ind
            else:
                piano_init[chunk_index, len(this_chunk):] = pad_ind
            return piano_init

        piano_init = torch.zeros(len(rhythm_piano), self.max_number_messages_piano)
        this_chunk = []
        chunk_index = 0
        frame_counter = 0
        writing_stop_flag = True

        for piano_symbol in piano_init_list:
            if piano_symbol in meta_inds:
                if frame_counter >= self.sequence_size - 1:
                    #  Write in piano_init
                    piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)
                    #  Clean this_chunk
                    while this_chunk[0] not in meta_inds:
                        this_chunk.pop(0)
                    this_chunk.pop(0)
                    chunk_index += 1
                frame_counter += 1

            if piano_symbol == end_ind:
                this_chunk.append(end_ind)
                this_chunk.append(stop_ind)
                writing_stop_flag = False
            else:
                this_chunk.append(piano_symbol)

        #  Don't forget last chunk
        piano_init = write_piano_chunk(piano_init, this_chunk, chunk_index, writing_stop_flag)

        # Orchestra
        padding_length = context_length * 2
        num_frames = len(rhythm_piano) + padding_length
        orchestra_silences, orchestra_unknown, instruments_presence, orchestra_init = \
            self.init_orchestra(num_frames, context_length, banned_instruments, unknown_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)
        instruments_presence_init = instruments_presence.unsqueeze(0).repeat(batch_size, 1, 1)

        return piano_init.long().cuda(), piano_write.long(), rhythm_piano, \
               orchestra_init.long().cuda(), \
               instruments_presence_init.long().cuda(), orchestra_silences, orchestra_unknown

    def piano_tensor_to_score(self, tensor_score, durations=None, writing_tempo="adagio", subdivision=None):
        """

        :param durations:
        :param tensor_score: one-hot encoding with dimensions (time, instrument)
        :return:
        """
        # (batch, num_parts, notes_encoding)
        elems = list(tensor_score.numpy())

        if subdivision is None:
            subdivision = self.subdivision

        if durations is None:
            # Note: using len(elems) yields a duration vector which is too long, but fuck it
            durations = np.ones((len(elems))) * subdivision

        #  Init score
        stream = music21.stream.Stream()
        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(0, music21_instrument)

        # Keep track of the duration of notes on
        notes_on_duration = {}
        frame_index = 0
        global_offset = 0
        for elem in elems:
            #  DEBUG
            if elem == -1:
                # This is for debugging, to separate between batches
                f = music21.note.Rest()
                f.quarterLength = 1
                this_part.insert(global_offset / subdivision, f)
                global_offset += subdivision
                continue

            #  Get symbol corresponding to one-hot encoding
            symbol = self.index2symbol_piano[elem]

            if symbol in [START_SYMBOL, END_SYMBOL, MASK_SYMBOL, PAD_SYMBOL]:
                global_offset += durations[frame_index]
                # frame_index += 1
            elif symbol in [TIME_SHIFT, STOP_SYMBOL]:
                # print(f'TS')
                # Increment duration in notes_on_duration
                global_offset += durations[frame_index]
                for pitch, _ in notes_on_duration.items():
                    notes_on_duration[pitch]['duration'] += durations[frame_index]
                #  increment frame counter
                frame_index += 1

                if symbol == STOP_SYMBOL:
                    # Write the last notes_on
                    for pitch, values in notes_on_duration.items():
                        duration = values['duration']
                        offset = values['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    break
            else:
                pitch, on = re.split('_', symbol)

                # print(f'{pitch} {on}')

                #  Repeat case
                if on == 'on':
                    if pitch in notes_on_duration.keys():
                        #  Write this pitch with its associated duration
                        duration = notes_on_duration[pitch]['duration']
                        offset = notes_on_duration[pitch]['offset']
                        f = music21.note.Note(int(pitch))
                        f.volume.velocity = 60.
                        f.quarterLength = duration / subdivision
                        this_part.insert((offset / subdivision), f)
                    #  Either it's a repeat or not, add in notes_on_duration
                    #  Write 0, as durations are all updated when a time shift event is met
                    else:
                        #  If it's not a note repeat need to instantiate the note in the dict
                        notes_on_duration[pitch] = {}
                    notes_on_duration[pitch]['duration'] = 0
                    notes_on_duration[pitch]['offset'] = global_offset
                # Note off
                elif on == 'off':
                    # Write this pitch with its associated duration
                    if pitch not in notes_on_duration.keys():
                        continue
                    duration = notes_on_duration[pitch]['duration']
                    offset = notes_on_duration[pitch]['offset']
                    f = music21.note.Note(int(pitch))
                    f.volume.velocity = 60.
                    f.quarterLength = duration / subdivision
                    this_part.insert((offset / subdivision), f)
                    #  Remove from notes_on_duration
                    notes_on_duration.pop(pitch, None)
                else:
                    raise Exception("should not happen")

        this_part.atSoundingPitch = self.transpose_to_sounding_pitch
        stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano=None, writing_dir=None,
                        filepath=None, writing_tempo='adagio', subdivision=None):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{self.dump_folder}/arrangement"

        if len(piano_pianoroll.size()) == 1:
            piano_pianoroll = torch.unsqueeze(piano_pianoroll, dim=0)
            orchestra_pianoroll = torch.unsqueeze(orchestra_pianoroll, dim=0)

        num_batches = len(piano_pianoroll)

        for batch_ind in range(num_batches):
            piano_part = self.piano_tensor_to_score(piano_pianoroll[batch_ind], durations_piano,
                                                    writing_tempo=writing_tempo,
                                                    subdivision=subdivision)
            orchestra_stream = self.orchestra_tensor_to_score(orchestra_pianoroll[batch_ind], durations_piano,
                                                              writing_tempo=writing_tempo,
                                                              subdivision=subdivision)

            piano_part.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_piano.mid", fmt='midi')
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_orchestra.mid", fmt='midi')
            # Both in the same score
            orchestra_stream.append(piano_part)
            orchestra_stream.write(fp=f"{writing_dir}/{filepath}_{batch_ind}_both.mid", fmt='midi')


if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    config = get_config()

    # parameters
    sequence_size = 5
    max_transposition = 12
    velocity_quantization = 2
    subdivision = 16
    integrate_discretization = True

    transposition_semi_tone = 0

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path=f'{config["database_path"]}/Orchestration/arrangement',
        subsets=[
            # 'liszt_classical_archives',
            'debug',
        ],
        num_elements=None
    )

    dataset = ArrangementMidipianoDataset(corpus_it_gen,
                                          corpus_it_gen_instru_range=None,
                                          name="shit",
                                          mean_number_messages_per_time_frame=10,
                                          subdivision=subdivision,
                                          sequence_size=sequence_size,
                                          max_transposition=max_transposition,
                                          integrate_discretization=True,
                                          alignement_type='complete',
                                          transpose_to_sounding_pitch=True,
                                          compute_statistics_flag=None)

    dataset.load_index_dicts()

    writing_dir = f'{config["dump_folder"]}/arrangement_voice/reconstruction_midi'
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)

    num_frames = 500

    for arr_pair in dataset.iterator_gen():
        ######################################################################
        # Piano
        # Tensor
        arr_id = arr_pair['name']

        pr_piano, onsets_piano, _ = score_to_pianoroll(
            score=arr_pair['Piano'],
            subdivision=subdivision,
            simplify_instrumentation=None,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=True
        )

        events_piano = new_events(pr_piano, onsets_piano)
        events_piano = events_piano[:num_frames]
        piano_tensor = []
        previous_frame_index = None
        for frame_index in events_piano:
            piano_tensor = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano['Piano'],
                onsets=onsets_piano['Piano'],
                previous_frame_index=previous_frame_index,
                frame_index=frame_index,
                piano_vector=piano_tensor)
            previous_frame_index = frame_index

            #  Time shift piano
            piano_tensor.append(dataset.symbol2index_piano[TIME_SHIFT])

        #  Remove the last time shift added
        piano_tensor.pop()

        piano_tensor = torch.LongTensor(piano_tensor)

        # Reconstruct
        piano_cpu = piano_tensor.cpu()
        duration_piano = list(np.asarray(events_piano)[1:] - np.asarray(events_piano)[:-1]) + [subdivision]

        piano_part = dataset.piano_tensor_to_score(piano_cpu, duration_piano, subdivision=subdivision)
        # piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.xml", fmt='musicxml')
        piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.mid", fmt='midi')

        ######################################################################
        #  Orchestra
        pr_orchestra, onsets_orchestra, _ = score_to_pianoroll(
            score=arr_pair['Orchestra'],
            subdivision=subdivision,
            simplify_instrumentation=dataset.simplify_instrumentation,
            instrument_grouping=dataset.instrument_grouping,
            transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch,
            integrate_discretization=dataset.integrate_discretization,
            binarize=True
        )

        events_orchestra = new_events(pr_orchestra, onsets_orchestra)
        events_orchestra = events_orchestra[:num_frames]
        orchestra_tensor = []
        previous_notes_orchestra = None
        for frame_counter, frame_index in enumerate(events_orchestra):
            orchestra_t_encoded, previous_notes_orchestra, _ = dataset.pianoroll_to_orchestral_tensor(
                pr=pr_orchestra,
                onsets=onsets_orchestra,
                previous_notes=previous_notes_orchestra,
                frame_index=frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)

        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = list(np.asarray(events_orchestra)[1:] - np.asarray(events_orchestra)[:-1]) + [
            subdivision]
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra,
                                                           subdivision=subdivision)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.mid", fmt='midi')

        ######################################################################
        # Original
        try:
            arr_pair["Orchestra"].write(fp=f"{writing_dir}/{arr_id}_original.mid", fmt='midi')
            arr_pair["Piano"].write(fp=f"{writing_dir}/{arr_id}_original_piano.mid", fmt='midi')
        except:
            print("Can't write original")

        ######################################################################
        # Aligned version
        corresponding_frames = dataset.align_score(piano_pr=pr_piano,
                                                   piano_onsets=onsets_piano,
                                                   orchestra_pr=pr_orchestra,
                                                   orchestra_onsets=onsets_orchestra)

        corresponding_frames = corresponding_frames[:num_frames]

        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_notes_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, previous_notes_orchestra, orchestra_instruments_presence_t_encoded = \
                dataset.pianoroll_to_orchestral_tensor(
                    pr=pr_orchestra,
                    onsets=onsets_orchestra,
                    previous_notes=previous_notes_orchestra,
                    frame_index=frame_orchestra
                )
            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue
            orchestra_tensor_event.append(orchestra_t_encoded)

            #######
            # Piano
            piano_tensor_event = dataset.pianoroll_to_piano_tensor(
                pr=pr_piano['Piano'],
                onsets=onsets_piano['Piano'],
                previous_frame_index=previous_frame_index,
                frame_index=frame_piano,
                piano_vector=piano_tensor_event
            )
            previous_frame_index = frame_piano
            #  Time shift piano
            piano_tensor_event.append(dataset.symbol2index_piano[TIME_SHIFT])

        #  Remove the last time shift added
        piano_tensor_event.pop()

        piano_tensor_event = torch.LongTensor(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, durations=None, subdivision=subdivision)
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu, durations=None, subdivision=subdivision)
        orchestra_part.append(piano_part)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.mid", fmt='midi')
