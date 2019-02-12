import os

import music21
import torch
from DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
from DatasetManager.helpers import ShortChoraleIteratorGen
from DatasetManager.metadata import TickMetadata, \
                                    BeatMarkerMetadata, \
                                    FermataMetadata, \
                                    KeyMetadata
from DatasetManager.lsdb.lsdb_data_helpers import LeadsheetIteratorGenerator
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.music_dataset import MusicDataset
from DatasetManager.the_session.folk_dataset import FolkDataset, \
                                                    FolkMeasuresDataset, \
                                                    FolkMeasuresDatasetTranspose, \
                                                    FolkDatasetNBars
from DatasetManager.the_session.folk_data_helpers \
             import FolkIteratorGenerator

from DatasetManager.arrangement.arrangement_frame_dataset import ArrangementFrameDataset
from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

# Basically, all you have to do to use an existing dataset is to
# add an entry in the all_datasets variable
# and specify its base class and which music21 objects it uses
# by giving an iterator over music21 scores

all_datasets = {
    'arrangement_frame_test':
        {
            'dataset_class_name': ArrangementFrameDataset,
            'corpus_it_gen':      ArrangementIteratorGenerator(
                # arrangement_path='/home/leo/databases/Orchestration/LOP_database_06_09_17',
                arrangement_path='/home/leo/databases/Orchestration/LOP_database_mxml_clean',
                subsets=[
                    # 'bouliane',
                    # 'imslp',
                    # 'liszt_classical_archives',
                    # 'hand_picked_Spotify',
                    'debug'],
                num_elements=10
            )
        },
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
    'bach_chorales_beats_test':
        {
            'dataset_class_name': ChoraleBeatsDataset,
            'corpus_it_gen':      ShortChoraleIteratorGen()
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
                num_elements=10
            )
        },
    'lsdb':
        {
            'dataset_class_name': LsdbDataset,
            'corpus_it_gen':      LeadsheetIteratorGenerator(
                num_elements=None
            )
        },
    'folk':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(3, 4), (4, 4)]
            )
        },
    'folk_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=10,
                has_chords=False,
                time_sigs=[(3, 4), (4, 4)]
            )
        },
    'folk_4by4_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=100,
                has_chords=False,
                time_sigs=[(4, 4)]
            ) 
        },
    'folk_4by4':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(4, 4)]
            ) 
        },
    'folk_3by4_test':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=100,
                has_chords=False,
                time_sigs=[(3, 4)]
            ) 
        },
    'folk_3by4':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen':      FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(3, 4)]
            ) 
        },
    'folk_4by4chords':
        {
            'dataset_class_name': FolkDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                has_chords=True,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measures_test':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=100,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measures_test2':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=1,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measures':
        {
            'dataset_class_name': FolkMeasuresDataset,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measurestr_test':
        {
            'dataset_class_name': FolkMeasuresDatasetTranspose,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=1000,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4measurestr':
        {
            'dataset_class_name': FolkMeasuresDatasetTranspose,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4nbars_test':
        {
            'dataset_class_name': FolkDatasetNBars,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=100,
                has_chords=False,
                time_sigs=[(4, 4)]
            )
        },
    'folk_4by4nbars':
        {
            'dataset_class_name': FolkDatasetNBars,
            'corpus_it_gen': FolkIteratorGenerator(
                num_elements=None,
                has_chords=False,
                time_sigs=[(4, 4)]
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

    # Arrangement
    subdivision = 1
    metadatas = [
    ]

    arrangement_dataset: ArrangementFrameDataset = dataset_manager.get_dataset(
        name='arrangement_frame_test',
        metadatas=metadatas,
        subdivision=subdivision
    )

    # (train_dataloader,
    #  val_dataloader,
    #  test_dataloader) = bach_chorales_dataset.data_loaders(
    #     batch_size=128,
    #     split=(0.85, 0.10)
    # )
    # print('Num Train Batches: ', len(train_dataloader))
    # print('Num Valid Batches: ', len(val_dataloader))
    # print('Num Test Batches: ', len(test_dataloader))

    # BACH
    '''
    subdivision = 4
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]
    
    bach_chorales_dataset: ChoraleDataset = dataset_manager.get_dataset(
        name='bach_chorales_test',
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=subdivision
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = bach_chorales_dataset.data_loaders(
        batch_size=128,
        split=(0.85, 0.10)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))
    '''
    # LSDB
    '''
    lsdb_dataset: LsdbDataset = dataset_manager.get_dataset(
        name='lsdb_test',
        sequences_size=64,
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = lsdb_dataset.data_loaders(
        batch_size=128,
        split=(0.85, 0.10)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))
    '''

    # Folk Dataset
    '''
    metadatas = [
        BeatMarkerMetadata(subdivision=6),
        TickMetadata(subdivision=6)
    ]
    folk_dataset_kwargs = {
        'metadatas':        metadatas,
        'sequences_size':   32
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name ='folk_4by4nbars',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=256,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))'''