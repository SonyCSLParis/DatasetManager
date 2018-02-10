from abc import ABC, abstractmethod

from torch.utils.data import TensorDataset, DataLoader


class MusicDataset(ABC):
	"""
	Abstract Base Class for music data sets
	Must return
	"""

	def __init__(self):
		self._tensor_dataset = None

	def initialize(self):
		self._tensor_dataset = self.make_tensor_dataset()

	@abstractmethod
	def make_tensor_dataset(self):
		"""

		:return: TensorDataset
		"""
		pass

	@property
	def tensor_dataset(self):
		if self._tensor_dataset is None:
			self.initialize()
		return self._tensor_dataset

	def data_loaders(self, batch_size, split=(0.85, 0.10)):
		"""
		Returns three data loaders obtained by splitting
		self.tensor_dataset according to split
		:param batch_size:
		:param split:
		:return:
		"""
		assert sum(split) < 1

		dataset = self.tensor_dataset
		num_examples = dataset.data_tensor.size()[0]
		a, b = split
		train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
		val_dataset = TensorDataset(*dataset[int(a * num_examples):
		                                     int((a + b) * num_examples)])
		eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

		train_dl = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=False,
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
