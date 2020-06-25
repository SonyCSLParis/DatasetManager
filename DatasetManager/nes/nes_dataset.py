from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import shutil
import time
from tqdm import tqdm

import pretty_midi

pretty_midi.pretty_midi.MAX_TICK = 1e16

import torch
from torch.utils.data import Dataset


class NESDataset(Dataset):
    def __init__(self, phase='train', voices=None):
        self.root = (Path(__file__) / '../../../../data/nesmdb_midi').resolve()
        self.voices = voices if voices is not None else [0, 1, 2, 3]

        self.train = phase == 'train'

        if not self.root.exists():
            raise ValueError(
                f"The dataset could not be found in this folder: {self.root}.\n"
                "Move it to that folder or create a link if you already have it elsewhere.\n"
                "Otherwise run ./DatasetManager/nes/init_nes_dataset.sh to download and extract it (72 MB)."
            )

        self.processed = self.root / 'processed'
        self.processed.mkdir(exist_ok=True)

        # assign an id to each instrument
        self.instrument_to_id = {
            80: 0,
            81: 1,
            38: 2,
            121: 3
        }
        self.id_to_instrument = {v: k for k, v in self.instrument_to_id.items()}

        # Loading index dicts
        self.dict_path = self.root / 'pitches.json'
        if self.dict_path.exists():
            with open(self.dict_path, 'r') as f:
                pitches_dict = json.load(f)
                # JSON only supports type string for dict keys
                self.pitches_dict = {int(k): v for k, v in pitches_dict.items()}

        else:
            self.pitches_dict = defaultdict(lambda: len(self.pitches_dict.keys()),
                                            {-1: -1, 0: 0})  # enable automatic updating of keys

            shutil.rmtree(self.processed)

            self.processed.mkdir(exist_ok=True)

            # preprocessing
            print(f'Preprocessing {self}...')
            t0 = time.time()
            self.preprocess_dataset()
            t1 = time.time()
            d = t1 - t0
            print('Done. Time elapsed:',
                  '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))

            self.pitches_dict = dict(self.pitches_dict)  # remove defaultdict behaviour
            with open(self.dict_path, 'w') as f:
                json.dump(self.pitches_dict, f)

        self.paths = list((self.processed / phase).glob('*.npy'))

        self.data_augmentation = lambda x: x

    def __getitem__(self, idx):

        path = self.paths[idx]
        # print(path)
        score = np.load(path, allow_pickle=False)[:, self.voices, :]

        ## data augmentation
        # score = self.data_augmentation(score)
        if self.train:
            padding_mask = (score == -1)

            # 1. transpose melodic voices by a random number of semitones between -6 and 5
            actual_voices = [i for i in range(score.shape[1]) if score[0, i, 0] >= 0]
            melodic_voices = [i for i in actual_voices if self.id_to_instrument[self.voices[i]] < 112]  # TODO:

            if melodic_voices != []:
                melodic_pitches = score[:, melodic_voices, 0] % 128
                pitch_shift = np.random.randint(
                    -min(6, melodic_pitches[melodic_pitches > 0].min()),
                    min(6, 128 - melodic_pitches.max())
                )
                score[:, melodic_voices, 0] += pitch_shift

            # 2. adjust the speed of the piece by a random percentage between +/- 5%
            time_speed = 1 + (np.random.random() - 0.5) / 10

            score[:, :, 1] *= time_speed
            score[:, :, 3] *= time_speed
            #
            # actual_num_voices = sum(data[0,:,0] == 0)
            # if actual_num_voices > 1 and np.random.random() < 0.5:
            #     # 3. half of the time, remove one of the instruments from the ensemble
            #     score[:,np.random.randint(actual_num_voices)] = 0
            #
            #     # 4. half of the time, shuffle the score-to-instrument alignment for the melodic instruments only
            #     melodic_voices = max(actual_num_voices, 3)
            #     # v1 = np.random.randint(melodic_voices)
            #     # v2 = (v1 + 1) % melodic_voices
            #     # # score[:,v1] # TODO:
            score[padding_mask] = -1

        # replace midi pitches by an id so that every id between 0 and id_max are used
        score[:, :, 0] = np.vectorize(self.pitches_dict.__getitem__)(score[:, :, 0].astype(np.int32))
        return torch.from_numpy(score)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return f'<NESDataset(dir={self.processed})>'

    def __str__(self):
        return 'NES-MDB Dataset'

    # PREPROCESSING

    def preprocess_dataset(self):
        r"""Recursively explore the dataset and generate for each MIDI file found
        a numpy array containing its data
        """
        bar = tqdm(sorted(list(self.root.glob('**/*.mid'))), leave=False)
        for src_file in bar:
            fname = src_file.name
            bar.set_description(fname[:29] + '...' if len(fname) > 32 else fname + (32 - len(fname)) * ' ')

            target_dir = self.processed / src_file.relative_to(self.root).parent
            target_name = src_file.stem + '.npy'
            target_file = target_dir / target_name

            # skip already preprocessed files
            if target_file.exists():
                continue

            target_dir.mkdir(exist_ok=True)

            # compute musical events as tensor
            midi = pretty_midi.PrettyMIDI(str(src_file))
            tensor = self.parse_midi(midi)

            # save the computed tensor
            np.save(target_file, tensor, allow_pickle=False)

    def parse_midi(self, midi):
        r"""Read a MIDI file and return all notes in it in a numpy.ndarray

        Arguments:
            midi (pretty_midi.PrettyMIDI): MIDI file to be read

        Returns:
            numpy.ndarray: padded content of the MIDI
        """
        voices = []
        num_notes_per_voice = []
        instrument_ids = []

        for instrument in midi.instruments:
            try:
                instrument_id = self.instrument_to_id[instrument.program]
            except KeyError:
                raise KeyError(
                    f"Instrument {instrument.program} has not been registered in NESDataset.instruments_id. "
                    f"As of now the registered instruments are {', '.join(self.instruments_ids.keys())} "
                    f"whereas the instruments of the current MIDI file are {', '.join([i.program for i in midi.instruments])}."
                )
            voice = []

            # add notes in the requested format
            for n in instrument.notes:
                pitch = 128 * instrument_id + n.pitch
                assert pitch > 0, str(n)
                # add the note to the list of notes
                voice.append([pitch, n.duration, n.velocity, n.start])

                # associate to the pitch a unique id if not done yet
                for p in range(max(128 * instrument_id, pitch - 6), min(128 * (instrument_id + 1), pitch + 6)):
                    _ = self.pitches_dict[p]

            # ensure that notes are sorted by increasing starts
            voice.sort(key=lambda n: n[3])
            # add the voice to the list of voices
            voices.append(np.array(voice, dtype=np.float32))
            num_notes_per_voice.append(len(voice))
            instrument_ids.append(instrument_id)

        # pad voices # NOTE: padding value MUST ABSOLUTELY be -1 so that the trick in split_sequence works
        padded_voices = -np.ones(
            (max(num_notes_per_voice), len(self.id_to_instrument), 4),
            dtype=np.float32
        )
        for i, voice, num_notes in zip(instrument_ids, voices, num_notes_per_voice):
            padded_voices[:num_notes, i, :] = voice

        return padded_voices

    def generate_midi(self, notes_tensor):
        r"""
        """
        notes_per_instrument = defaultdict(list)

        if torch.is_tensor(notes_tensor):
            notes_tensor = notes_tensor.cpu().numpy()

        voice_pitch, duration, velocity, start = notes_tensor.T
        voice_pitch = voice_pitch.astype(np.int64)
        velocity = velocity.astype(np.int64)

        instrument_id, pitch = np.divmod(voice_pitch, 128)

        end = start + duration

        for i, v, p, s, e in zip(instrument_id, velocity, pitch, start, end):
            notes_per_instrument[i].append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))

        midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)

        for instrument_id, notes in notes_per_instrument.items():
            prog = self.id_to_instrument[instrument_id]
            instrument = pretty_midi.Instrument(program=prog, is_drum=(prog >= 112))

            instrument.notes = notes

            midi.instruments.append(instrument)

        ts = pretty_midi.TimeSignature(4, 4, 0)
        eos = pretty_midi.TimeSignature(1, 1, float(max(end) - min(start)))
        midi.time_signature_changes.extend([ts, eos])
        # print(midi.instruments[0].notes)
        # print(midi.instruments)
        # print(midi.time_signature_changes)
        # raise IndexError
        return midi


if __name__ == '__main__':
    pass
