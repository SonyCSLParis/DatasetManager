import torch
import numpy as np
from DatasetManager.arrangement.arrangement_midiPiano_dataset import ArrangementMidipianoDataset
from DatasetManager.config import get_config
from DatasetManager.helpers import REST_SYMBOL, TIME_SHIFT

if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, score_to_pianoroll, \
        new_events, flatten_dict_pr

    config = get_config()

    # parameters
    subdivision = 4
    sequence_size = 3
    max_transposition = 3

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path=f'{config["database_path"]}/Orchestration/arrangement',
        subsets=[
            # 'bouliane',
            # 'imslp',
            'liszt_classical_archives',
            # 'hand_picked_Spotify',
            # 'debug',
        ],
        num_elements=None
    )
    group_instrument_per_section = True

    dataset = ArrangementMidipianoDataset(corpus_it_gen,
                                          corpus_it_gen_instru_range=None,
                                          name="shit",
                                          mean_number_messages_per_time_frame=10,
                                          subdivision=subdivision,
                                          sequence_size=sequence_size,
                                          max_transposition=max_transposition,
                                          transpose_to_sounding_pitch=True,
                                          cache_dir=None,
                                          compute_statistics_flag=None)

    dataset.compute_index_dicts()

    writing_dir = f'{config["dump_folder"]}/reconstruction_midi'

    num_frames = 500

    for arr_pair in dataset.iterator_gen():
        ######################################################################
        # Piano
        # Tensor
        arr_id = arr_pair['name']

        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'],
                                                              subdivision,
                                                              simplify_instrumentation=None,
                                                              instrument_grouping=dataset.instrument_grouping,
                                                              transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch)
        events_piano = new_events(pianoroll_piano, onsets_piano)
        events_piano = events_piano[:num_frames]
        onsets_piano = onsets_piano["Piano"]
        piano_tensor = []
        previous_frame_index = None
        for frame_index in events_piano:
            piano_tensor = dataset.pianoroll_to_piano_tensor(
                pr=pianoroll_piano["Piano"],
                onsets=onsets_piano,
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
        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(arr_pair["Orchestra"],
                                                                      subdivision,
                                                                      dataset.simplify_instrumentation,
                                                                      dataset.instrument_grouping,
                                                                      dataset.transpose_to_sounding_pitch)
        events_orchestra = new_events(pianoroll_orchestra, onsets_orchestra)
        flat_pr_orch = np.where(flatten_dict_pr(pianoroll_orchestra) > 1, 1, 0)
        events_orchestra = events_orchestra[:num_frames]
        orchestra_tensor = []
        previous_frame_orchestra = None
        for frame_counter, frame_index in enumerate(events_orchestra):
            orchestra_t_encoded, _ = dataset.pianoroll_to_orchestral_tensor(
                pianoroll_orchestra,
                onsets_orchestra,
                previous_frame=previous_frame_orchestra,
                frame_index=frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)
            previous_frame_orchestra = orchestra_t_encoded

        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = list(np.asarray(events_orchestra)[1:] - np.asarray(events_orchestra)[:-1]) + [subdivision]
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra, subdivision=subdivision)
        # orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.xml", fmt='musicxml')
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
        corresponding_frames, this_scores = dataset.align_score(arr_pair['Piano'], arr_pair['Orchestra'])
        corresponding_frames = corresponding_frames[:num_frames]

        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        piano_tensor_event = []
        orchestra_tensor_event = []
        previous_frame_index = None
        previous_frame_orchestra = None
        for frame_counter, (frame_piano, frame_orchestra) in enumerate(zip(piano_frames, orchestra_frames)):

            #  IMPORTANT:
            #  Compute orchestra first to know if the frame has to be skipped or not
            #  (typically if too many instruments are played in one section)

            #######
            # Orchestra
            orchestra_t_encoded, orchestra_instruments_presence_t_encoded = dataset.pianoroll_to_orchestral_tensor(
                pianoroll_orchestra,
                onsets_orchestra,
                previous_frame=previous_frame_orchestra,
                frame_index=frame_orchestra)
            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue
            orchestra_tensor_event.append(orchestra_t_encoded)
            previous_frame_orchestra = orchestra_t_encoded

            #######
            # Piano
            piano_tensor_event = dataset.pianoroll_to_piano_tensor(
                pr=pianoroll_piano["Piano"],
                onsets=onsets_piano,
                previous_frame_index=previous_frame_index,
                frame_index=frame_piano,
                piano_vector=piano_tensor_event)
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
        # orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.xml", fmt='musicxml')
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.mid", fmt='midi')
