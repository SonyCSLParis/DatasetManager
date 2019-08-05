import os
import shutil

from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.arrangement.arrangement_midiPiano_dataset import ArrangementMidipianoDataset
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata, BeatMarkerMetadata
from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.the_session.folk_dataset import FolkDataset

if __name__ == '__main__':

    database_to_run = "arrangement_voice"
    number_dump = 100
    batch_size = 32
    subdivision = 16
    sequence_size = 7
    integrate_discretization = True

    dataset_manager = DatasetManager()

    ###########################################################
    # Arrangement
    if database_to_run == 'arrangement':

        arrangement_dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement',
            transpose_to_sounding_pitch=True,
            subdivision=subdivision,
            sequence_size=sequence_size,
            max_transposition=12,
            velocity_quantization=2,
            compute_statistics_flag=False,
        )

        (train_dataloader,
         val_dataloader,
         test_dataloader) = arrangement_dataset.data_loaders(
            batch_size=batch_size,
            split=(0.85, 0.10),
            DEBUG_BOOL_SHUFFLE=True
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))

        # Visualise a few examples
        writing_dir = f"{arrangement_dataset.dump_folder}/{database_to_run}/writing"
        if os.path.isdir(writing_dir):
            shutil.rmtree(writing_dir)
        os.makedirs(writing_dir)
        for i_batch, sample_batched in enumerate(train_dataloader):
            piano_batch, orchestra_batch, instrumentation_batch = sample_batched
            if i_batch > number_dump:
                break
            arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}")

    ###########################################################
    # Arrangement voice piano
    elif database_to_run == 'arrangement_voice':
        arrangement_dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
            name='arrangement_voice_small',
            transpose_to_sounding_pitch=True,
            subdivision=subdivision,
            sequence_size=sequence_size,
            integrate_discretization=integrate_discretization,
            max_transposition=12,
            alignement_type='complete',
            compute_statistics_flag=False,
        )

        (train_dataloader,
         val_dataloader,
         test_dataloader) = arrangement_dataset.data_loaders(
            batch_size=batch_size,
            split=(0.85, 0.10),
            DEBUG_BOOL_SHUFFLE=True
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))

        # Visualise a few examples
        writing_dir = f"{arrangement_dataset.dump_folder}/{database_to_run}/writing"
        if os.path.isdir(writing_dir):
            shutil.rmtree(writing_dir)
        os.makedirs(writing_dir)
        for i_batch, sample_batched in enumerate(train_dataloader):
            piano_batch, orchestra_batch, instrumentation_batch = sample_batched
            if i_batch > number_dump:
                break
            arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir,
                                                filepath=f"{i_batch}")

    ###########################################################
    # Arrangement Midi piano
    elif database_to_run == 'arrangement_midi':
        mean_number_messages_per_time_frame = 14
        arrangement_dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
            name='arrangement_midiPiano_small',
            transpose_to_sounding_pitch=True,
            subdivision=subdivision,
            sequence_size=sequence_size,
            max_transposition=12,
            integrate_discretization=True,
            alignement_type='complete',
            compute_statistics_flag=False,
            mean_number_messages_per_time_frame=mean_number_messages_per_time_frame
        )

        (train_dataloader,
         val_dataloader,
         test_dataloader) = arrangement_dataset.data_loaders(
            batch_size=batch_size,
            split=(0.85, 0.10),
            DEBUG_BOOL_SHUFFLE=True
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))

        # Visualise a few examples
        writing_dir = f"{arrangement_dataset.dump_folder}/{database_to_run}/writing"
        if os.path.isdir(writing_dir):
            shutil.rmtree(writing_dir)
        os.makedirs(writing_dir)
        for i_batch, sample_batched in enumerate(train_dataloader):
            piano_batch, orchestra_batch, instrumentation_batch = sample_batched
            if i_batch > number_dump:
                break
            arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}")

    ###########################################################
    # BACH
    elif database_to_run == 'bach_beat':
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
            batch_size=batch_size,
            split=(0.85, 0.10)
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))

    # LSDB
    elif database_to_run == 'lsdb':
        lsdb_dataset: LsdbDataset = dataset_manager.get_dataset(
            name='lsdb_test',
            sequences_size=64,
        )
        (train_dataloader,
         val_dataloader,
         test_dataloader) = lsdb_dataset.data_loaders(
            batch_size=batch_size,
            split=(0.85, 0.10)
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))

    # Folk Dataset
    elif database_to_run == 'folk':
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
            batch_size=batch_size,
            split=(0.7, 0.2)
        )
        print('Num Train Batches: ', len(train_dataloader))
        print('Num Valid Batches: ', len(val_dataloader))
        print('Num Test Batches: ', len(test_dataloader))
