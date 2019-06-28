import os
import torch
from DatasetManager.music_dataset import MusicDataset
import DatasetManager.all_datasets as all_datasets

# Basically, all you have to do to use an existing dataset is to
# add an entry in the all_datasets variable
# and specify its base class and which music21 objects it uses
# by giving an iterator over music21 scores

all_datasets = all_datasets.get_all_datasets()


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
    from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset

    # # Arrangement
    # subdivision = 4
    # sequence_size = 3
    # arrangement_dataset: ArrangementDataset = dataset_manager.get_dataset(
    #     name='arrangement_test',
    #     transpose_to_sounding_pitch=True,
    #     subdivision=subdivision,
    #     sequence_size=sequence_size,
    #     velocity_quantization=2,
    #     max_transposition=3,
    #     compute_statistics_flag=False
    # )
    #
    # (train_dataloader,
    #  val_dataloader,
    #  test_dataloader) = arrangement_dataset.data_loaders(
    #     batch_size=16,
    #     split=(0.85, 0.10),
    #     DEBUG_BOOL_SHUFFLE=True
    # )
    # print('Num Train Batches: ', len(train_dataloader))
    # print('Num Valid Batches: ', len(val_dataloader))
    # print('Num Test Batches: ', len(test_dataloader))
    #
    # # Visualise a few examples
    # number_dump = 20
    # writing_dir = f"{arrangement_dataset.dump_folder}/arrangement/writing"
    # if os.path.isdir(writing_dir):
    #     shutil.rmtree(writing_dir)
    # os.makedirs(writing_dir)
    # for i_batch, sample_batched in enumerate(train_dataloader):
    #     piano_batch, orchestra_batch = sample_batched
    #     # Flatten matrices
    #     # piano_flat = piano_batch.view(-1, dataset.number_pitch_piano)
    #     # piano_flat_t = piano_flat[dataset.sequence_size - 1::dataset.sequence_size]
    #     # orchestra_flat = orchestra_batch.view(-1, dataset.number_instruments)
    #     # orchestra_flat_t = orchestra_flat[dataset.sequence_size - 1::dataset.sequence_size]
    #     if i_batch > number_dump:
    #         break
    #     arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}_seq")
    #     # dataset.visualise_batch(piano_flat_t, orchestra_flat_t, writing_dir, filepath=f"{i_batch}_t")

    # BACH
    from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata
    from DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
    subdivision = 4
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]

    bach_chorales_dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
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

    # LSDB
    # lsdb_dataset: LsdbDataset = dataset_manager.get_dataset(
    #     name='lsdb_test',
    #     sequences_size=64,
    # )
    # (train_dataloader,
    #  val_dataloader,
    #  test_dataloader) = lsdb_dataset.data_loaders(
    #     batch_size=128,
    #     split=(0.85, 0.10)
    # )
    # print('Num Train Batches: ', len(train_dataloader))
    # print('Num Valid Batches: ', len(val_dataloader))
    # print('Num Test Batches: ', len(test_dataloader))

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