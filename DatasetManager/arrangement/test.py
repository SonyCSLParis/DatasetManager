import glob
import shutil

import torch

import music21
import numpy as np
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.config import get_config

if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, score_to_pianoroll, \
        quantize_velocity_pianoroll_frame

    config = get_config()

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

    dataset = ArrangementDataset(corpus_it_gen,
                                 corpus_it_gen_instru_range=None,
                                 name="shit",
                                 subdivision=2,
                                 sequence_size=3,
                                 velocity_quantization=8,
                                 max_transposition=3,
                                 transpose_to_sounding_pitch=True,
                                 cache_dir=None,
                                 compute_statistics_flag=None)


    dataset.compute_index_dicts()

    subdivision = 4

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

        quantized_pianoroll_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                                      dataset.velocity_quantization)

        onsets_piano = onsets_piano["Piano"]
        flat_onsets = onsets_piano.sum(1)
        rhythm_piano = np.where(flat_onsets > 0)[0]
        rhythm_piano = rhythm_piano[:100]

        piano_tensor = []
        for frame_index in rhythm_piano:
            piano_t_encoded = dataset.pianoroll_to_piano_tensor(
                quantized_pianoroll_piano,
                onsets_piano,
                frame_index)
            piano_tensor.append(piano_t_encoded)
        piano_tensor = torch.stack(piano_tensor)

        # Reconstruct
        piano_cpu = piano_tensor.cpu()
        duration_piano = np.asarray(list(rhythm_piano[1:]) + [subdivision]) - np.asarray(list(rhythm_piano[:-1]) + [0])

        piano_part = dataset.piano_tensor_to_score(piano_cpu, duration_piano, subdivision=subdivision)
        piano_part.write(fp=f"{writing_dir}/{arr_id}_piano.xml", fmt='musicxml')

        ######################################################################
        #  Orchestra
        pianoroll_orchestra, onsets_orchestra, _ = score_to_pianoroll(arr_pair["Orchestra"],
                                                                      subdivision,
                                                                      dataset.simplify_instrumentation,
                                                                      dataset.instrument_grouping,
                                                                      dataset.transpose_to_sounding_pitch)

        quantized_pianoroll_orchestra = {k: quantize_velocity_pianoroll_frame(v, dataset.velocity_quantization)
                                         for k, v in pianoroll_orchestra.items()}
        #  New events orchestra
        onsets_cumulated = None
        for k, v in onsets_orchestra.items():
            if onsets_cumulated is None:
                onsets_cumulated = v.sum(1)
            else:
                onsets_cumulated += v.sum(1)
        rhythm_orchestra = np.where(onsets_cumulated > 0)[0]
        rhythm_orchestra = rhythm_orchestra[:100]

        orchestra_tensor = []
        rhythm_orchestra_clean = []
        for frame_index in rhythm_orchestra:
            orchestra_t_encoded, _ = dataset.pianoroll_to_orchestral_tensor(
                quantized_pianoroll_orchestra,
                onsets_orchestra,
                frame_index)
            if orchestra_t_encoded is not None:
                orchestra_tensor.append(orchestra_t_encoded)
                rhythm_orchestra_clean.append(frame_index)
        orchestra_tensor = torch.stack(orchestra_tensor)

        # Reconstruct
        orchestra_cpu = orchestra_tensor.cpu()
        duration_orchestra = np.asarray(list(rhythm_orchestra_clean[1:]) + [subdivision]) - np.asarray(
            list(rhythm_orchestra_clean[:-1]) + [0])
        orchestra_part = dataset.orchestra_tensor_to_score(orchestra_cpu, duration_orchestra, subdivision=subdivision)
        orchestra_part.write(fp=f"{writing_dir}/{arr_id}_orchestra.xml", fmt='musicxml')
        try:
            arr_pair["Orchestra"].write(fp=f"{writing_dir}/{arr_id}_original.xml", fmt='musicxml')
        except:
            print("Can't write original")

