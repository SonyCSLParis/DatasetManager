import shutil
from abc import ABC, abstractmethod
import os
from torch.utils.data import TensorDataset, DataLoader
import torch


class MusicDataset(ABC):
    """
    Abstract Base Class for music data sets
    Must return
    """

    def __init__(self):
        self.tensor_dataset = None

    @abstractmethod
    def iterator_gen(self):
        """

        return: Iterator over the dataset
        """
        pass

    @abstractmethod
    def make_tensor_dataset(self):
        """

        :return: TensorDataset
        """
        pass

    @abstractmethod
    def get_score_tensor(self, score):
        """

        :param score: music21 score object
        :return: torch tensor, with the score representation
                 as a tensor
        """
        pass

    @abstractmethod
    def get_metadata_tensor(self, score):
        """

        :param score: music21 score object
        :return: torch tensor, with the metadata representation
                 as a tensor
        """
        pass

    @abstractmethod
    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        """

        :param score: music21 score object
        :param semi-tone: int, +12 to -12, semitones to transpose 
        :return: Transposed score shifted by the semi-tone
        """
        pass

    @abstractmethod
    def extract_score_tensor_with_padding(self,
                                          tensor_score,
                                          start_tick,
                                          end_tick):
        """

        :param tensor_score: torch tensor containing the score representation
        :param start_tick:
        :param end_tick:
        :return: tensor_score[:, start_tick: end_tick]
        with padding if necessary
        i.e. if start_tick < 0 or end_tick > tensor_score length
        """
        pass

    @abstractmethod
    def extract_metadata_with_padding(self,
                                      tensor_metadata,
                                      start_tick,
                                      end_tick):
        """

        :param tensor_metadata: torch tensor containing metadata
        :param start_tick:
        :param end_tick:
        :return:
        """
        pass

    @abstractmethod
    def empty_score_tensor(self, score_length):
        """
        
        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with start indices 
        """
        pass

    @abstractmethod
    def random_score_tensor(self, score_length):
        """

        :param score_length: int, length of the score in ticks
        :return: torch long tensor, initialized with random indices
        """
        pass

    @abstractmethod
    def tensor_to_score(self, tensor_score):
        """

        :param tensor_score: torch tensor, tensor representation
                             of the score
        :return: music21 score object
        """
        pass

    def get_tensor_dataset(self, cache_dir):
        """
        Loads or computes TensorDataset
        :return: TensorDataset
        """
        if self.tensor_dataset is None:
            if self.tensor_dataset_is_cached(cache_dir):
                print(f'Loading TensorDataset for {self.__repr__()}')
                self.tensor_dataset = torch.load(self.tensor_dataset_filepath(cache_dir))
            else:
                print(f'Creating {self.__repr__()} TensorDataset'
                      f' since it is not cached')
                #  Create dataset dir
                dataset_dir = os.path.join(cache_dir, self.__repr__())
                if os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)
                os.makedirs(dataset_dir)
                #  Build database
                self.tensor_dataset = self.make_tensor_dataset()
                #  Store
                torch.save(self.tensor_dataset, self.tensor_dataset_filepath(cache_dir))
                print(f'TensorDataset for {self.__repr__()} '
                      f'saved in {self.tensor_dataset_filepath(cache_dir)}')
        return self.tensor_dataset

    def tensor_dataset_is_cached(self, cache_dir):
        return os.path.exists(self.tensor_dataset_filepath(cache_dir))

    def tensor_dataset_filepath(self, cache_dir):
        return os.path.join(cache_dir, self.__repr__(), 'tensor_dataset')

    def filepath(self, cache_dir):
        return os.path.join(cache_dir, self.__repr__(), 'dataset')

    def data_loaders(self, batch_size, cache_dir, split=(0.85, 0.10), DEBUG_BOOL_SHUFFLE=True):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1

        dataset = self.get_tensor_dataset(cache_dir)
        num_examples = len(dataset)
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                             int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=DEBUG_BOOL_SHUFFLE,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl
