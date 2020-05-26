from collections import defaultdict
import json
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

import music21

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class NESDataset(Dataset):
    def __init__(self, phase='train', num_blocks=None, time_interval=1, num_voices=4):
        self.root = (Path(__file__) / '../../../../data/nesmdb_midi').resolve()
        self.num_voices = num_voices
        self.train = phase == 'train'

        if not self.root.exists():
            raise ValueError(
                f"The dataset could not be found in this folder: {self.root}.\n"
                "Move it to that folder or create a link if you already have it elsewhere.\n"
                "Otherwise run ./DatasetManager/datasets/init_nes_dataset.sh to download and extract it (72 MB)."
            )

        self.processed = self.root / 'processed'

        self.processed.mkdir(exist_ok=True)


        self.fields = ['pitch', 'duration', 'velocity', 'offset']

        # Loading index dicts
        self.dict_path = self.root / 'pitches.json'
        if self.dict_path.exists():
            with open(self.dict_path, 'r') as f:
                pitches_dict = json.load(f)
        else:
            pitches_dict = {0: 0}
        # JSON only supports type string for dict keys
        self.pitches_dict = {int(k): v for k, v in pitches_dict.items()}

        # preprocessing
        if not (self.processed / '.done').exists():
            self.pitches_dict = defaultdict(lambda: len(self.pitches_dict.keys()), self.pitches_dict) # enable automatic updating of keys
            print(f'Preprocessing {self}...')
            t0 = time.time()
            self.preprocess_dataset()
            t1 = time.time()
            d = t1 - t0
            print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))

            with open(self.processed / '.done', 'wb') as f:     # indicates that the dataset has been fully preprocessed
                f.write(bytes(0))

        self.paths = list((self.processed / phase).glob('*.npy'))

        self.data_augmentation = lambda x: x

        self.min_duration = None if num_blocks is None else num_blocks * time_interval

    def __getitem__(self, idx):
        path = self.paths[idx]
        # print(path)
        score = np.load(path, allow_pickle=False)

        # skip too short sequences
        if self.min_duration is not None and score[...,3].max() < self.min_duration:
            return None

        ## data augmentation
        score = self.data_augmentation(score)

            # # 1. transpose melodic voices by a random number of semitones between -6 and 5
            # pitch_shift = np.random.randint(-6, 6)
            # score[:,:3,0] += pitch_shift        # TODO: correct pitch -> skip rests
            #
            # # 2. adjust the speed of the piece by a random percentage between +/- 5%
            # time_speed = 1 + (np.random.random()-0.5) / 10
            #
            # score[:,:,1] *= time_speed
            # score[:,:,3] *= time_speed
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

        # replace midi pitches by an id so that every id between 0 and id_max are used
        score[:,:,0] = np.vectorize(self.pitches_dict.get)(score[:,:,0].astype(np.int32), -1)
        return torch.from_numpy(score)

    def __len__(self):
        return len(self.paths)

    def __repr__(self):
        return f'<NESDataset(dir={self.processed})>'

    def __str__(self):
        return 'NES-MDB Dataset'



    # PREPROCESSING

    def preprocess_dataset(self):
        for src_file in tqdm(sorted(list(self.root.glob('**/*.mid'))), leave=False):
            target_dir = self.processed / src_file.relative_to(self.root).parent
            target_name = src_file.stem + '.npy'
            target_file = target_dir / target_name

            # skip already preprocessed files
            if target_file.exists():
                continue

            target_dir.mkdir(exist_ok=True)

            # compute musical events as tensor
            score = music21.converter.parse(src_file)
            tensor = self.parse_score(score)


            # save the computed tensor
            np.save(target_file, tensor, allow_pickle=False)

            with open(self.dict_path, 'w') as f:
                json.dump(self.pitches_dict, f, indent=2) # NOTE: pitches_dict must be saved after each step so that preprocessing can be interrupted

    def parse_score(self, score, padding_value=-1):
        r"""Computes the per-voice list of musical events of a midi input

        Args:
            score: music21.stream.Score

        Returns:
            torch.tensor, shape (num_events, num_voices, 4)
        """
        # get tempo
        metronome = score.flat.getElementsByClass(music21.tempo.MetronomeMark)
        try: bpm = next(metronome).getQuarterBPM()
        except StopIteration: bpm = 120

        # computes features (pitch, duration, velocity, timeshift) of notes
        events_per_voice = self.num_voices * [torch.zeros(1,4)+padding_value]
        for i, part in enumerate(score.parts):
            events = [self.compute_event(n, i) for n in part.flat.notesAndRests]

            # # add END_SYMBOL
            # last_event = events[-1][-1]
            # end_event = (0, 0, 0, last_event[1] + last_event[3])
            # events.append([end_event])

            events = torch.tensor([e for event in events for e in event]) # flatten

            # rescale to seconds instead of beats
            events[:,1] *= 60 / bpm
            events[:,3] *= 60 / bpm

            events_per_voice[i] = events

        return pad_sequence(events_per_voice, padding_value=padding_value)




    def compute_event(self, n, i):
        """Returns the (pitch, duration, velocity, offset) of a note and fills
        pitches_dict on the fly

        Args:
            n: music21.GeneralNote
                note to extract features
            i: int
                voice index

        Returns:
            tuple
                features of the note (pitch, duration, velocity, offset)
        """
        d, o = float(n.quarterLength), float(n.offset)

        if n.isRest:
            k = 129*i + 128
            _ = self.pitches_dict[k]    # add key to dict if not exists
            return [(k, d, 0, o)]

        notes = []
        v = n.volume.velocity
        for p in n.pitches:
            k = 129*i + p.midi
            _ = self.pitches_dict[k]
            notes.append((k, d, v, o))
        return notes



























if __name__ == '__main__':
    # compute statistics about the dataset
    from collections import Counter
    import matplotlib.pyplot as plt

    def split_sequence(sequence, interval=1):
        r"""Splits a sequence of events into a sequence of blocks, where each block
        contains all events for one voice during a time interval

        Args:
            sequences: torch.Tensor, shape (seq_length, num_voices, num_channels), slices indices for building blocks
                batch of musical events sequences
            interval: float (default=1)
                interval of sequences
            max_length: int
                maximal number of blocks per voice

        Returns:
            torch.Tensor, list of total_num_blocks torch.Tensors, shape (block_size, num_channels)
                sequences of blocks
        """
        all_blocks = []

        for voice in sequence.transpose(0,1):
            # compute the index of block for each event
            indices = voice[:,-1] / interval
            indices[indices < 0] = -1
            indices = indices.long()

            # compute the number of events per block
            bins, counts = torch.unique_consecutive(indices, return_counts=True)
            slices = torch.zeros(bins.max()+2, dtype=torch.int64)
            slices[bins] = counts

            # split the sequence into blocks
            blocks = list(torch.split(voice, slices.tolist()))

            blocks.pop()

            all_blocks.extend(blocks)

        return all_blocks

    def plot_hist(counters, titles=None):
        imax = int(np.sqrt(len(counters)))
        jmax = len(counters) // imax
        if titles is None:
            titles = len(counters)*['']
        for i, (c, title) in enumerate(zip(counters, titles)):
            # remove outliers since it causes problems at display
            to_remove = []
            for k, v in c.items():
                if v < 5: to_remove.append(k)
            for k in to_remove: del c[k]


            plt.subplot(imax, jmax, i+1)
            plt.bar(c.keys(), c.values(), color='b')
            # plt.xlim(0, 600)
            plt.yscale('log')
            plt.title(title)
        plt.show()

    dataset = NESDataset()


    # raise IndexError
    c_pitch = Counter()
    c_duration = Counter()
    c_velocity = Counter()
    c_timeshift = Counter()
    c_length = Counter()
    c_voices = Counter()
    c_nevents = Counter()

    for score in tqdm(dataset):
        blocks = split_sequence(score)
        lengths = [len(block) for block in blocks]
        c_nevents += Counter(lengths)
        # if max(lengths) > 100:
        #     # print(blocks)
        #     # print(lengths, len(lengths))
        #     # print(blocks[lengths.index(max(lengths))])
        #     break
    # # remove null values
    # c_pitch[0] = 0
    # c_velocity[0] = 0
    print(max(c_nevents.keys()))
    plot_hist([c_nevents])
    # plot_hist(
    #     [c_pitch, c_duration, c_velocity, c_timeshift, c_length, c_voices],
    #     [
    #         'pitch', 'duration', 'velocity', 'timeshift',
    #         'Number of events',
    #         'Number of voices'
    #     ]
    # )
