from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.music_dataset import MusicDataset
import os


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


if __name__ == '__main__':
    # dataset_manager = DatasetManager()
    folk_dataset = FolkDataset(cache_dir='../dataset_cache')
    folk_dataset.download_raw_dataset()
    folk_dataset.split_raw_dataset()
