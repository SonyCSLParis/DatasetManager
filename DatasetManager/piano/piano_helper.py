import glob
import os

import numpy as np


class PianoIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    def __init__(self, subsets, num_elements=None):
        self.path = f'{os.path.expanduser("~")}/Data/databases/Piano'
        self.subsets = subsets
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.generator()
        )
        return it

    def generator(self):
        midi_files = []
        for subset in self.subsets:
            # Should return pairs of files
            midi_files += (glob.glob(
                os.path.join(self.path, subset, '*.mid')))
            midi_files += (glob.glob(
                os.path.join(self.path, subset, '*.MID')))
        if self.num_elements is not None:
            midi_files = midi_files[:self.num_elements]
        for midi_file in midi_files:
            print(midi_file)
            yield midi_file


def get_midi_type(midi, midi_ranges):
    for feat_name, feat_range in midi_ranges.items():
        if midi in feat_range:
            midi_type = feat_name
            return midi_type


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
