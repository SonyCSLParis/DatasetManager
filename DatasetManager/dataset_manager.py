import os

import music21
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DatasetManager.helpers import ShortChoraleIteratorGen
from DatasetManager.metadata import TickMetadata, FermataMetadata, KeyMetadata
from DatasetManager.music_dataset import MusicDataset

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
	'bach_chorales_test':
		{
			'dataset_class_name': ChoraleDataset,
			'corpus_it_gen':      ShortChoraleIteratorGen()
		},

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
			{'name':                  name,
			 'chorale_corpus_it_gen': corpus_it_gen
			 })
		dataset = dataset_class_name(**kwargs)
		filepath = os.path.join(
			self.cache_dir,
			dataset.__repr__()
		)
		if os.path.exists(filepath):
			dataset = torch.load(filepath)
		else:
			dataset.initialize()
			torch.save(dataset, filepath)
			print(f'{dataset.__repr__()} saved in {filepath}')
		return dataset


# Usage example
if __name__ == '__main__':
	datataset_manager = DatasetManager()
	subdivision = 4
	metadatas = [
		TickMetadata(subdivision=subdivision),
		FermataMetadata(),
		KeyMetadata()
	             ]

	bach_chorales_dataset: ChoraleDataset = datataset_manager.get_dataset(
		name='bach_chorales',
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
	print(next(train_dataloader.__iter__()))
