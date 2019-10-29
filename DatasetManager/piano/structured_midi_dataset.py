import itertools
import math
import os
import pickle
import random
import shutil

import numpy as np
import pretty_midi
import torch
from torch.utils import data
from tqdm import tqdm

from DatasetManager.helpers import END_SYMBOL, START_SYMBOL, \
    PAD_SYMBOL
from DatasetManager.piano.harpsichord_midi_dataset import HarpsichordMidiDataset
from DatasetManager.piano.piano_helper import PianoIteratorGenerator, find_nearest_value, extract_cc

"""
Typical piano sequence:
p0 p1 TS p0 p1 p2 TS p0 STOP X X X X

If beginning: 
START p0 p1 TS p0 p1 p2 TS p0 STOP X X X

If end: 
p0 p1 TS p0 p1 p2 TS p0 END STOP X X X

"""


class StructuredMidiDataset(HarpsichordMidiDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 sequence_size,
                 max_transposition,
                 time_dilation_factor,
                 ):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__(corpus_it_gen,
                         sequence_size,
                         max_transposition,
                         time_dilation_factor)
        return

    def data_loaders(self, batch_size, split=(0.85, 0.10), DEBUG_BOOL_SHUFFLE=True):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1

        excluded_features = []
        # Just want this to be chosen when calling dataloaders, not before
        self.selected_features_indices = [self.index_order_dict[feat_name] for feat_name in self.index_order
                                          if feat_name not in excluded_features]

        num_examples = len(self)
        a, b = split
        train_ids = self.list_ids[: int(a * num_examples)]
        val_ids = self.list_ids[int(a * num_examples): int((a + b) * num_examples)]
        eval_ids = self.list_ids[int((a + b) * num_examples):]

        train_dataset = self.extract_subset(train_ids)
        val_dataset = self.extract_subset(val_ids)
        eval_dataset = self.extract_subset(eval_ids)

        train_dl = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=DEBUG_BOOL_SHUFFLE,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl