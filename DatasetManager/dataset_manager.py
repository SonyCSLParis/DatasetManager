import os

import music21
import torch
from DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
from DatasetManager.helpers import ShortChoraleIteratorGen
from DatasetManager.metadata import TickMetadata, BeatMarkerMetadata
from DatasetManager.lsdb.lsdb_data_helpers import LeadsheetIteratorGenerator
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.the_session.folk_dataset import FolkDataset
from DatasetManager.the_session.folk_data_helpers import FolkIteratorGenerator

# Basically, all you have to do to use an existing dataset is to
# add an entry in the all_datasets variable
# and specify its base class and which music21 objects it uses
# by giving an iterator over music21 scores

all_datasets = {
    'bach_chorales':
        {
            'dataset_class_name': ChoraleDataset,
            'corpus_it_gen':      music21.corpus.chorales.Iterator
        },
    'bach_chorales_beats':
        {
            'dataset_class_name': ChoraleBeatsDataset,
            'corpus_it_gen':      music21.corpus.chorales.Iterator
        },
    'bach_chorales_test':
        {
            'dataset_class_name': ChoraleDataset,
            'corpus_it_gen':      ShortChoraleIteratorGen()
        },
    'lsdb_test':
        {
            'dataset_class_name': LsdbDataset,
            'corpus_it_gen':      LeadsheetIteratorGenerator(
                num_elements=10)
        },
    'lsdb':
        {
            'dataset_class_name': LsdbDataset,
            'corpus_it_gen':      LeadsheetIteratorGenerator(
                num_elements=None)
        },
    'folk':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None
            )
        }
}


class DatasetManager:
    def __init__(self):
        self.package_dir = os.path.dirname(os.path.realpath(__file__))
        self.cache_dir = os.path.join(self.package_dir,
                                      'dataset_cache')
        # create cache dir if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def get_dataset(self, name: str, **dataset_kwargs) -> MusicDataset:
        if name in all_datasets:
            return self.load_if_exists_or_initialize_and_save(
                name=name,
                **all_datasets[name],
                **dataset_kwargs
            )
        else:
            print('Dataset with name {name} is not registered in all_datasets variable')
            raise ValueError

    def load_if_exists_or_initialize_and_save(self,
                                              dataset_class_name,
                                              corpus_it_gen,
                                              name,
                                              **kwargs):
        """

        :param dataset_class_name:
        :param corpus_it_gen:
        :param name:
        :param kwargs: parameters specific to an implementation
        of MusicDataset (ChoraleDataset for instance)
        :return:
        """
        kwargs.update(
            {'name':          name,
             'corpus_it_gen': corpus_it_gen,
             'cache_dir': self.cache_dir
             })
        dataset = dataset_class_name(**kwargs)
        if os.path.exists(dataset.filepath):
            print(f'Loading {dataset.__repr__()} from {dataset.filepath}')
            dataset = torch.load(dataset.filepath)
            print(f'(the corresponding TensorDataset is not loaded)')
        else:
            print(f'Creating {dataset.__repr__()}, '
                  f'both tensor dataset and parameters')
            # initialize and force the computation of the tensor_dataset
            # first remove the cached data if it exists
            if os.path.exists(dataset.tensor_dataset_filepath):
                os.remove(dataset.tensor_dataset_filepath)
            # recompute dataset parameters and tensor_dataset
            # this saves the tensor_dataset in dataset.tensor_dataset_filepath
            tensor_dataset = dataset.tensor_dataset
            # save all dataset parameters EXCEPT the tensor dataset
            # which is stored elsewhere
            dataset.tensor_dataset = None
            torch.save(dataset, dataset.filepath)
            print(f'{dataset.__repr__()} saved in {dataset.filepath}')
            dataset.tensor_dataset = tensor_dataset
        return dataset


# Usage example
if __name__ == '__main__':
    dataset_manager = DatasetManager()
    # BACH

    # subdivision = 4
    # metadatas = [
    # 	TickMetadata(subdivision=subdivision),
    # 	FermataMetadata(),
    # 	KeyMetadata()
    #              ]
    #
    # bach_chorales_dataset: ChoraleDataset = datataset_manager.get_dataset(
    # 	name='bach_chorales',
    # 	voice_ids=[0, 1, 2, 3],
    # 	metadatas=metadatas,
    # 	sequences_size=8,
    # 	subdivision=subdivision
    # )
    # (train_dataloader,
    #  val_dataloader,
    #  test_dataloader) = bach_chorales_dataset.data_loaders(
    # 	batch_size=128,
    # 	split=(0.85, 0.10)
    # )
    # print(next(train_dataloader.__iter__()))

    # LSDB
    #lsdb_dataset: LsdbDataset = datataset_manager.get_dataset(
    #    name='lsdb_test',
    #    sequences_size=64,
    #)
    #(train_dataloader,
    # val_dataloader,
    # test_dataloader) = lsdb_dataset.data_loaders(
    #    batch_size=128,
    #    split=(0.85, 0.10)
    #)
    #print(next(train_dataloader.__iter__()))

    # Folk Dataset  
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
    folk_dataset_kwargs = {
        'metadatas':        metadatas,
        'sequences_size':   32
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name ='folk',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=128,
        split=(0.85, 0.10)
    )
    print(next(train_dataloader.__iter__()))