from glob2 import glob
from music21.abcFormat import ABCHandlerException

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.music_dataset import MusicDataset
import os
import music21


class FolkDataset(MusicDataset):
    def __init__(self, cache_dir=None):
        super(FolkDataset, self).__init__(cache_dir=cache_dir)
        self.raw_dataset_dir = os.path.join(
            self.cache_dir,
            'raw_dataset',
        )
        if not os.path.exists(self.raw_dataset_dir):
            os.mkdir(self.raw_dataset_dir)

        self.full_raw_dataset_filepath = os.path.join(
            self.raw_dataset_dir,
            'raw_dataset_full.txt'
        )

        self.raw_dataset_url = 'https://raw.githubusercontent.com/IraKorshunova/' \
                               'folk-rnn/master/data/' \
                               'sessions_data_clean.txt'

    def download_raw_dataset(self):
        if os.path.exists(self.full_raw_dataset_filepath):
            print('The Session dump already exists')
        else:
            print('Downloading The Session dump')
            os.system(f'wget -L {self.raw_dataset_url} -O {self.full_raw_dataset_filepath}')

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

    def make_tensor_dataset(self):
        pass

    def get_abc_song(self, tune_index):
        tune_filepath = os.path.join(self.raw_dataset_dir,
                                     f'tune_{tune_index}.abc')
        if os.path.exists(tune_filepath):
            return None
        else:
            raise ValueError

    def find_tune_as_leadsheet(self):
        # todo remove add
        offset = 43434
        tune_filepaths = glob(f'{self.raw_dataset_dir}/tune*')
        # tune_filepaths = [f'{self.raw_dataset_dir}/tune_29777.abc']
        count = 0
        for tune_index, tune_filepath in enumerate(tune_filepaths):
            if tune_index < offset:
                continue
            title = self.get_title(tune_filepath)
            print()
            print('*******************************')
            print(f'{tune_index}: {title}')
            print(f'{tune_filepath}')
            if title is None:
                print('No title')
                continue
            if not self.tune_contains_chords(tune_filepath):
                print('No chords')
                continue
            if self.tune_is_multivoice(tune_filepath):
                print('Multivoice')
                continue
            try:
                score = music21.converter.parse(tune_filepath, format='abc')
                score.show()
                count += 1
            except (music21.abcFormat.ABCHandlerException,
                    music21.abcFormat.ABCTokenException,
                    music21.duration.DurationException,
                    UnboundLocalError,
                    ValueError,
                    ABCHandlerException) as e:
                print('Error when parsing ABC file')
                print(e)

        print(count)
        print(count / len(tune_filepaths) * 100)

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

    @staticmethod
    def get_title(tune_filepath):
        for line in open(tune_filepath):
            if line[:2] == 'T:':
                return line[2:]
        return None


if __name__ == '__main__':
    # dataset_manager = DatasetManager()
    folk_dataset = FolkDataset(cache_dir='../dataset_cache')
    folk_dataset.download_raw_dataset()
    # folk_dataset.split_raw_dataset()
    folk_dataset.find_tune_as_leadsheet()
