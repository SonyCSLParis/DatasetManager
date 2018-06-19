import music21
import torch
import numpy as np
import os
import sys

from glob2 import glob
from music21.abcFormat import ABCHandlerException

from music21 import interval, stream, meter
from torch.utils.data import TensorDataset
from tqdm import tqdm
from fractions import Fraction

from DatasetManager.music_dataset import MusicDataset
from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, \
    standard_name, PAD_SYMBOL, standard_note, OUT_OF_RANGE, \
    BEAT_SYMBOL, DOWNBEAT_SYMBOL
from DatasetManager.lsdb.lsdb_data_helpers import notes_and_chords, \
    leadsheet_on_ticks
from DatasetManager.lsdb.lsdb_exceptions import *
from DatasetManager.the_session.folk_data_helpers import get_notes, \
    get_notes_in_measure, tick_values

class FolkDataset(MusicDataset):
    def __init__(self,  
                 name,
                 corpus_it_gen = None, # TODO: NOT BEING USED RIGHT NOW
                 metadatas=None,
                 sequences_size=32,
                 subdivision = 4, # TODO: NOT BEING USED RIGHT NOW
                 cache_dir=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over the files (as music21 scores)
        :pram name
        :param sequences_size: in beats
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where the tensor_dataset is stored
        """
        super(FolkDataset, self).__init__(cache_dir=cache_dir)
        
        self.raw_dataset_dir = os.path.join(
            self.cache_dir,
            'raw_dataset',
        )
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.num_melodies = 10 ### Change this to increase / decrease the dataset size
        self.NOTES = 0
        self.num_voices = 1
        self.pitch_range = [55, 84]
        self.tick_values = tick_values
        self.subdivision = len(self.tick_values)
        self.tick_durations = self.compute_tick_durations()
        self.sequences_size = sequences_size
        self.metadatas = metadatas
        if self.metadatas:
            for metadata in self.metadatas:
                if metadata.name == 'beatmarker':
                    self.beat_index2symbol_dicts = metadata.beat_index2symbol_dicts
                    self.beat_symbol2index_dicts = metadata.beat_symbol2index_dicts
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.dict_path = os.path.join( 
            self.raw_dataset_dir,
            'dict_path.txt'
        )

    def __repr__(self):
        return f'FolkDataset(' \
               f'{self.name},' \
               f'{[metadata.name for metadata in self.metadatas]},' \
               f'{self.sequences_size},' \
               f'{self.subdivision})' \
               f'{self.num_melodies}'

    def chorale_iterator_gen(self):
        return (chorale
                for chorale in self.corpus_it_gen()
        )

    def compute_tick_durations(self):
        """
        Computes the tick durations
        """
        diff = [n - p
                for n, p in zip(self.tick_values[1:], self.tick_values[:-1])]
        diff = diff + [1 - self.tick_values[-1]]
        return diff

    def get_lead_tensor(self, score):
        """
        Extract the lead tensor from the lead sheet
        :param score: music21 score object
        :return: lead_tensor
        """
        eps = 1e-4
        notes, _ = notes_and_chords(score)
        if not leadsheet_on_ticks(score, self.tick_values):
            raise LeadsheetParsingException(
                f'Leadsheet {score.metadata.title} has notes not on ticks')

        # add entries to dictionaries if not present
        # should only be called by make_tensor_dataset when transposing
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
                                         for n in notes
                                         if n.isNote]
        note2index = self.note2index_dicts[self.NOTES]
        index2note = self.index2note_dicts[self.NOTES]
        pitch_range = self.pitch_range
        min_pitch, max_pitch = pitch_range
        for note_name, pitch in list_note_strings_and_pitches:
            # if out of range
            if pitch < min_pitch or pitch > max_pitch:
                note_name = OUT_OF_RANGE
            if note_name not in note2index:
                new_index = len(note2index)
                index2note.update({new_index: note_name})
                note2index.update({note_name: new_index})
                print('Warning: Entry ' + str(
                    {new_index: note_name}) + ' added to dictionaries')

        # construct sequence
        j = 0
        i = 0
        length = int(score.highestTime * self.subdivision)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0
        while i < length:
            if j < num_notes - 1:
                if notes[j + 1].offset > current_tick + eps:
                    t[i, :] = [note2index[standard_name(notes[j])],
                               is_articulated]
                    i += 1
                    current_tick += self.tick_durations[
                        (i - 1) % len(self.tick_values)]
                    is_articulated = False
                else:
                    j += 1
                    is_articulated = True
            else:
                t[i, :] = [note2index[standard_name(notes[j])],
                           is_articulated]
                i += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * note2index[SLUR_SYMBOL]
        # convert to torch tensor
        lead_tensor = torch.from_numpy(lead).long()[None, :]
        return lead_tensor #, chord_tensor

    def get_metadata_tensor(self, score):
        """
        Extract the metadata tensor (beat markers) from the lead sheet
        :param score: music21 score object
        :return: metadata_tensor
        """
        md = []
        if self.metadatas:
            for metadata in self.metadatas:
                sequence_metadata = torch.from_numpy(
                    metadata.evaluate(score, self.subdivision)).long().clone()
                square_metadata = sequence_metadata.repeat(self.num_voices, 1)
                md.append(
                    square_metadata[:, :, None]
                )
        # compute length
        lead_length = int(score.highestTime * self.subdivision)
        # add voice indexes
        voice_id_metada = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        square_metadata = torch.transpose(
            voice_id_metada.repeat(lead_length, 1),
            0, 
            1
            )
        md.append(square_metadata[:, :, None])

        all_metadata = torch.cat(md, 2)
        return all_metadata
        #all_metadata = torch.cat(md, 2)
        #return all_metadata

    def transposed_chorale_and_metadata_tensors(self, chorale, semi_tone):
        """
        Convert chorale to a couple (chorale_tensor, metadata_tensor),
        the original chorale is transposed semi_tone number of semi-tones

        :param chorale: music21 object
        :param semi_tone:
        :return: couple of tensors
        """
        # TODO: implement this properly. 
        if semi_tone != 0:
            raise NotImplementedError
        
        chorale_tensor = self.get_lead_tensor(chorale)
        metadata_tensor = self.get_metadata_tensor(chorale)
        return chorale_tensor, metadata_tensor

    def make_tensor_dataset(self):
        self.compute_index_dicts()
        print('Making tensor dataset')
        lead_tensor_dataset = []
        metadata_tensor_dataset = []
        count = 0
        for score_id, score in tqdm(enumerate(self.corpus_it_gen())):
            if not self.is_in_range(score):
                continue
            try:
                if count > self.num_melodies:
                    break
                count += 1
                lead_tensor = self.get_lead_tensor(score)
                metadata_tensor = self.get_metadata_tensor(score)
                # main loop - lead
                for offsetStart in range(-self.sequences_size + 1,
                                         int(score.highestTime)):
                    offsetEnd = offsetStart + self.sequences_size
                    local_lead_tensor = self.extract_lead_with_padding(
                        tensor=lead_tensor,
                        start_tick=offsetStart * self.subdivision,
                        end_tick=offsetEnd * self.subdivision,
                        symbol2index=self.note2index_dicts[self.NOTES]
                    )
                    local_metadata_tensor = self.extract_metadata_with_padding(
                        tensor_metadata=metadata_tensor,
                        start_tick=offsetStart * self.subdivision,
                        end_tick=offsetEnd * self.subdivision
                    )
                    # append and add batch dimension
                    # cast to int
                    lead_tensor_dataset.append(
                        local_lead_tensor.int()
                    )
                    metadata_tensor_dataset.append(
                        local_metadata_tensor.int()
                    )
            except LeadsheetParsingException as e:
                print(e)
                print(f'For score: {score_id}')
        lead_tensor_dataset = torch.cat(lead_tensor_dataset, 0)
        num_datapoints = lead_tensor_dataset.size()[0]
        lead_tensor_dataset = lead_tensor_dataset.view(
            num_datapoints, 1, -1
        )
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)
        num_datapoints, length, num_metadata = metadata_tensor_dataset.size()
        metadata_tensor_dataset = metadata_tensor_dataset.view(
            num_datapoints, 1, length, num_metadata
        )
        dataset = TensorDataset(lead_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {lead_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset

    def make_tensor_dataset_full_melody(self):
        """
        Creates tensor and metadata datasets with full length melodies
        Uses a packed padded sequence approach
        """
        self.compute_index_dicts()
        print('Making tensor dataset full melody')
        # TODO: Finish this method with packed padded sequence stuff
        local_lead_tensor = []
        local_metadata_tensor = []
        longest_seq = 0
        for score_id, score in tqdm(enumerate(self.corpus_it_gen())):
            if not self.is_in_range(score):
                continue
            try:
                lead_tensor = self.get_lead_tensor(score)
                metadata_tensor = self.get_metadata_tensor(score)
                # lead and metadata tensors should have same length
                assert(lead_tensor.size()[1] == metadata_tensor.size()[1])
                seq_len = lead_tensor.size()[1]
                if seq_len > longest_seq:
                    longest_seq = seq_len
                # add lead and metadata tesnors to respective lists
                local_lead_tensor.append(lead_tensor.int())
                local_metadata_tensor.append(metadata_tensor.int())
            except LeadsheetParsingException as e:
                print(e)
                print(f'For score: {score_id}')
        # pad the tensors appropriately and store the padding lengths
        lead_tensor_dataset = self.create_packed_pad_dataset(
            local_lead_tensor, longest_seq
        )
        metadata_tensor_dataset = self.create_packed_pad_dataset(
            local_metadata_tensor, longest_seq
        )
        # resize tensors to fit model expectations
        # TODO: complete this part 
        
        # create pytorch Tensor Dataset
        dataset = TensorDataset(lead_tensor_dataset, metadata_tensor_dataset)
        print(f'Sizes: {lead_tensor_dataset.size()}')
        print(f'Sizes: {metadata_tensor_dataset.size()}')
        return dataset

    def create_packed_pad_dataset(self, tensor_dataset, longest_seq_len):
        """
        Takes a list of variable length tensors and creates a dataset
        Uses pytorch packed padded sequence to zero-pad appropriately

        :param tensor_dataset: list of tensors differening in dim 1
        :param longest_seq_len: length of the longest sequence
        """
        # get the tensor dimensions

        # interate and pad zeros, keep track of number of zeros padded


    def transposed_lead_tensor(self, score, semi_tone):
        """
        Convert lead to a tensor,
        the original lead is transposed semi_tone number of semi-tones

        :param score: music21 object
        :param semi_tone: int, number of semi-tones to transpose by
        :return: transposed lead tensor
        """
        # transpose
        # compute the most "natural" interval given a number of semi-tones
        interval_type, interval_nature = interval.convertSemitoneToSpecifierGeneric(
            semi_tone)
        transposition_interval = interval.Interval(
            str(interval_nature) + interval_type)

        score_tranposed = score.transpose(transposition_interval)
        if not self.is_in_range(score_tranposed):
            return None
        lead_tensor = self.get_lead_tensor(score_tranposed)
        return lead_tensor


    def extract_lead_with_padding(self, tensor,
                             start_tick,
                             end_tick,
                             symbol2index):
        """

        :param tensor: (batch_size, length)
        :param start_tick:
        :param end_tick:
        :param symbol2index:
        :return: (batch_size, end_tick - start_tick)
        """
        assert start_tick < end_tick
        assert end_tick > 0
        batch_size, length = tensor.size()

        padded_tensor = []
        if start_tick < 0:
            start_symbols = np.array([symbol2index[START_SYMBOL]])
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            start_symbols = start_symbols.repeat(batch_size, -start_tick)
            #start_symbols[-1] = symbol2index[START_SYMBOL]
            padded_tensor.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length

        padded_tensor.append(tensor[:, slice_start: slice_end])

        if end_tick > length:
            end_symbols = np.array([symbol2index[END_SYMBOL]])
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            end_symbols = end_symbols.repeat(batch_size, end_tick - length)
            #end_symbols[0] = symbol2index[END_SYMBOL]
            padded_tensor.append(end_symbols)

        padded_tensor = torch.cat(padded_tensor, 1)
        return padded_tensor

    def extract_metadata_with_padding(self, tensor_metadata,
                                      start_tick, end_tick):
        """

        :param tensor_metadata: (num_voices, length, num_metadatas)
        last metadata is the voice_index
        :param start_tick:
        :param end_tick:
        :return:
        """
        assert start_tick < end_tick
        assert end_tick > 0
        num_voices, length, num_metadatas = tensor_metadata.size()
        padded_tensor_metadata = []

        if start_tick < 0:
            # TODO fix PAD symbol in the beginning and end
            start_symbols = np.zeros((self.num_voices, -start_tick, num_metadatas))
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            padded_tensor_metadata.append(start_symbols)

        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length
        padded_tensor_metadata.append(tensor_metadata[:, slice_start: slice_end, :])

        if end_tick > length:
            end_symbols = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            padded_tensor_metadata.append(end_symbols)

        padded_tensor_metadata = torch.cat(padded_tensor_metadata, 1)
        return padded_tensor_metadata

    def compute_index_dicts_for_score(self, score):
        """
        Computes the index dcitionaries specific to the symbols in the score
        For debugging only
        """
        #self.compute_beatmarker_dicts
        print('Computing note index dicts for score')
        self.index2note_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.note2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            #note_set.add(PAD_SYMBOL)
        # part is either lead or chords as lists
        for part_id, part in enumerate(notes_and_chords(score)):
            for n in part:
                note_sets[part_id].add(standard_name(n))
         # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2note_dicts,
                                                    self.note2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})
    
    def compute_index_dicts(self):
        if os.path.exists(self.dict_path):
            print('Dictionaries already exists. Reading them now')
            f = open(self.dict_path, 'r')
            dicts = [line.rstrip('\n') for line in f]
            assert(len(dicts) == 2) # must have 2 dictionaries
            self.index2note_dicts = eval(dicts[0])
            self.note2index_dicts = eval(dicts[1])
            return

        #self.compute_beatmarker_dicts()
        print('Computing note index dicts')
        self.index2note_dicts = [
            {} for _ in range(self.num_voices)
        ]
        self.note2index_dicts = [
            {} for _ in range(self.num_voices)
        ]

        # create and add additional symbols
        note_sets = [set() for _ in range(self.num_voices)]
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            #note_set.add(PAD_SYMBOL)

        # get all notes
        # iteratre through all scores and fill in the notes 
        #for tune_filepath in tqdm(self.valid_tune_filepaths):
        count = 0
        for _, score in tqdm(enumerate(self.corpus_it_gen())):
            #score = self.get_score_from_path(tune_filepath)
            # part is either lead or chords as lists
            if count > self.num_melodies:
                break
            count += 1
            for part_id, part in enumerate(notes_and_chords(score)):
                for n in part:
                    note_sets[part_id].add(standard_name(n))

        # create tables
        for note_set, index2note, note2index in zip(note_sets,
                                                    self.index2note_dicts,
                                                    self.note2index_dicts):
            for note_index, note in enumerate(note_set):
                index2note.update({note_index: note})
                note2index.update({note: note_index})
        
        # write as text file for use later
        f = open(self.dict_path, 'w')
        f.write("%s\n" % self.index2note_dicts)
        f.write("%s\n" % self.note2index_dicts)
        f.close()

    def is_in_range(self, score):
        """
        Checks if the pitches are within the min and max range

        :param score: music21 score object
        :return: boolean 
        """
        notes, _ = notes_and_chords(score)
        pitches = [n.pitch.midi for n in notes if n.isNote]
        if pitches == []:
            return False
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        return (min_pitch >= self.pitch_range[0]
                and max_pitch <= self.pitch_range[1])


    def tensor_chorale_to_score(self, tensor_chorale):
        """
        Converts lead given as tensor_lead to a true music21 score
        :param tensor_chorale:
        :return:
        """
        slur_index = self.note2index_dicts[self.NOTES][SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_chorale.numpy().flatten()
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.subdivision]
                f = standard_note(self.index2note_dicts[self.NOTES][note_index])
            else:
                dur += self.tick_durations[tick_index % self.subdivision]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    def empty_chorale(self, chorale_length):
        start_symbols = np.array([note2index[START_SYMBOL]
                                  for note2index in self.note2index_dicts])
        start_symbols = torch.from_numpy(start_symbols).long().clone()
        start_symbols = start_symbols.repeat(chorale_length, 1).transpose(0, 1)
        return start_symbols

if __name__ == '__main__':
    # dataset_manager = DatasetManager()
    folk_dataset = FolkDataset('folk', cache_dir='../dataset_cache')
    #folk_dataset.download_raw_dataset()
    folk_dataset.make_tensor_dataset
