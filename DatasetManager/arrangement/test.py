import glob
import shutil

import torch
import numpy as np
from DatasetManager.arrangement.arrangement_categorical_dataset import ArrangementCategoricalDataset
from DatasetManager.config import get_config
from helpers import REST_SYMBOL

if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, score_to_pianoroll, \
        new_events

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
    group_instrument_per_section = False

    dataset = ArrangementCategoricalDataset(corpus_it_gen,
                                            corpus_it_gen_instru_range=None,
                                            name="shit",
                                            subdivision=subdivision,
                                            sequence_size=sequence_size,
                                            max_transposition=max_transposition,
                                            transpose_to_sounding_pitch=True,
                                            cache_dir=None,
                                            compute_statistics_flag=None,
                                            group_instrument_per_section=group_instrument_per_section)

    dataset.compute_index_dicts()

    writing_dir = f'{config["dump_folder"]}/reconstruction'

    for arr_id, arr_pair in enumerate(dataset.iterator_gen()):
        ######################################################################
        # Piano
        # Tensor
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(arr_pair['Piano'],
                                                              subdivision,
                                                              simplify_instrumentation=None,
                                                              instrument_grouping=dataset.instrument_grouping,
                                                              transpose_to_sounding_pitch=dataset.transpose_to_sounding_pitch)
        events_piano = new_events(pianoroll_piano, onsets_piano)
        onsets_piano = onsets_piano["Piano"]
        piano_tensor = []
        for frame_index in events_piano:
            piano_t_encoded = dataset.pianoroll_to_piano_tensor(
                pianoroll_piano["Piano"],
                onsets_piano,
                frame_index)
            if piano_t_encoded is None:
                piano_t_encoded = dataset.precomputed_vectors_piano[REST_SYMBOL]
            piano_tensor.append(piano_t_encoded)

        piano_tensor = torch.stack(piano_tensor)

        # Reconstruct
        piano_cpu = piano_tensor.cpu()
        duration_piano = list(np.asarray(events_piano)[1:] - np.asarray(events_piano)[:-1]) + [subdivision]

        piano_part = dataset.piano_tensor_to_score(piano_cpu, duration_piano, subdivision=subdivision)
        piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.xml", fmt='musicxml')

        ######################################################################
        #  Orchestra
        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(arr_pair["Orchestra"],
                                                                      subdivision,
                                                                      dataset.simplify_instrumentation,
                                                                      dataset.instrument_grouping,
                                                                      dataset.transpose_to_sounding_pitch)
        events_orchestra = new_events(pianoroll_orchestra, onsets_orchestra)
        orchestra_tensor = []
        for frame_index in events_orchestra:
            orchestra_t_encoded, _ = dataset.pianoroll_to_orchestral_tensor(
                pianoroll_orchestra,
                onsets_orchestra,
                frame_index)
            if orchestra_t_encoded is None:
                orchestra_t_encoded = dataset.precomputed_vectors_orchestra[REST_SYMBOL]
            orchestra_tensor.append(orchestra_t_encoded)

        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = list(np.asarray(events_orchestra)[1:] - np.asarray(events_orchestra)[:-1]) + [subdivision]
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra, subdivision=subdivision)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.xml", fmt='musicxml')

        ######################################################################
        # Original
        try:
            arr_pair["Orchestra"].write(fp=f"{writing_dir}/{arr_id}_original.xml", fmt='musicxml')
        except:
            print("Can't write original")

        ######################################################################
        # Aligned version
        corresponding_frames, this_scores = dataset.align_score(arr_pair['Piano'], arr_pair['Orchestra'])

        piano_frames = [e[0][0] for e in corresponding_frames]
        orchestra_frames = [e[1][0] for e in corresponding_frames]

        piano_tensor_event = []
        orchestra_tensor_event = []
        for frame_piano, frame_orchestra in zip(piano_frames, orchestra_frames):
            piano_t_encoded = dataset.pianoroll_to_piano_tensor(
                pianoroll_piano["Piano"],
                onsets_piano,
                frame_piano)
            orchestra_t_encoded, orchestra_instruments_presence_t_encoded = dataset.pianoroll_to_orchestral_tensor(
                pianoroll_orchestra,
                onsets_orchestra,
                frame_orchestra)

            if orchestra_t_encoded is None:
                avoid_this_chunk = True
                continue

            if piano_t_encoded is None:
                avoid_this_chunk = True
                continue

            piano_tensor_event.append(piano_t_encoded)
            orchestra_tensor_event.append(orchestra_t_encoded)

        piano_tensor_event = torch.stack(piano_tensor_event)
        orchestra_tensor_event = torch.stack(orchestra_tensor_event)
        # Reconstruct
        orchestra_cpu = orchestra_tensor_event.cpu()
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, durations=None, subdivision=subdivision)
        piano_cpu = piano_tensor_event.cpu()
        piano_part = dataset.piano_tensor_to_score(piano_cpu, durations=None, subdivision=subdivision)
        orchestra_part.append(piano_part)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_both_aligned.xml", fmt='musicxml')
