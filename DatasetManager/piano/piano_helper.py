import csv
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
                os.path.join(self.path, subset, '*.midi')))
            midi_files += (glob.glob(
                os.path.join(self.path, subset, '*.MID')))
        if self.num_elements is not None:
            midi_files = midi_files[:self.num_elements]
        for midi_file in midi_files:
            print(midi_file)
            yield midi_file


class MaestroIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    def __init__(self, composers_filter=[], num_elements=None):
        self.path = f'{os.path.expanduser("~")}/Data/databases/Piano/maestro-v2.0.0'
        self.composers_filter = composers_filter
        self.num_elements = num_elements

    def __str__(self):
        ret = 'Maestro'
        if self.num_elements is not None:
            ret += f'_{self.num_elements}'
        return ret

    def __call__(self, *args, **kwargs):
        it = (
            elem
            for elem in self.generator()
        )
        return it

    def generator(self):
        midi_files = []
        splits = []
        master_csv_path = f'{self.path}/maestro-v2.0.0.csv'
        with open(master_csv_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                if row['canonical_composer'] in self.composers_filter:
                    continue
                else:
                    midi_name = row["midi_filename"]
                    midi_files.append(f'{self.path}/{midi_name}')
                    splits.append(row['split'])
        if self.num_elements is not None:
            midi_files = midi_files[:self.num_elements]
            splits = splits[:self.num_elements]
        for split, midi_file in zip(splits, midi_files):
            print(f'{split}: {midi_file}')
            yield midi_file, split


def extract_cc(control_changes, channel, binarize):
    ret_time = []
    ret_value = []
    previous_value = -1
    for cc in control_changes:
        if cc.number == channel:
            if binarize:
                value = 1 if cc.value > 0 else 0
                if value == previous_value:
                    continue
            else:
                value = cc.value
            ret_time.append(cc.time)
            ret_value.append(value)
            previous_value = value

    if len(ret_time) == 0:
        ret_time = [0]
        ret_value = [0]
    elif ret_time[0] > 0:
        ret_time.insert(0, 0.)
        ret_value.insert(0, 0)

    return np.array(ret_time), np.array(ret_value)


def get_midi_type(midi, midi_ranges):
    for feat_name, feat_range in midi_ranges.items():
        if midi in feat_range:
            midi_type = feat_name
            return midi_type


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
