import copy
import csv

from DatasetManager.config import get_config
from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator, note_to_midiPitch, separate_instruments_names
import music21
import numpy as np
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class ComputeStatistics:
    def __init__(self, database_path, subsets, subdivision, sounding_pitch_boolean=False):

        config = get_config()
        self.dump_folder = config['dump_folder']
        simplify_instrumentation_path = config['simplify_instrumentation_path']

        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)
        #  Histogram with range of notes per instrument
        self.tessitura = dict()
        # Histogram with number of simultaneous notes*
        self.simultaneous_notes = dict()
        # Histogram with the number of simultaneous notes per instrument
        self.stat_dict = dict()
        self.score_iterator = ArrangementIteratorGenerator(
            arrangement_path=database_path,
            subsets=subsets
        )
        self.subdivision = subdivision
        self.sounding_pitch_boolean = sounding_pitch_boolean

        # Write out paths here
        self.simultaneous_notes_details_path = f'{self.dump_folder}/arrangement/statistics/simultaneous_notes_details.txt'
        open(self.simultaneous_notes_details_path, 'w').close()

        return

    def get_statistics(self):
        for arrangement_pair in self.score_iterator():
            for score in arrangement_pair:
                self.histogram_tessitura(score)
                self.counting_simultaneous_notes(score)

        # Get reference tessitura for comparing when plotting
        with open('reference_tessitura.json') as ff:
            reference_tessitura = json.load(ff)
        reference_tessitura = {k: (note_to_midiPitch(music21.note.Note(v[0])), note_to_midiPitch(music21.note.Note(v[1]))) for
                               k, v in reference_tessitura.items()}

        stats = []
        this_stats = {}
        for instrument_names, histogram in self.tessitura.items():
            for instrument_name in separate_instruments_names(instrument_names):
                # Plot histogram for each instrument
                x = range(128)
                y = list(histogram.astype(int))
                plt.bar(x, y)
                plt.xlabel('Notes', fontsize=8)
                plt.ylabel('Frequency', fontsize=8)
                plt.title(instrument_name, fontsize=10)
                plt.xticks(np.arange(0, 128, 1))
                # Add reference tessitura
                if instrument_name != "Remove":
                    min_ref, max_ref = reference_tessitura[instrument_name]
                else:
                    min_ref = 0
                    max_ref = 128
                x = range(min_ref, max_ref)
                y = [0 for e in range(len(x))]
                plt.plot(x, y, 'ro')
                if self.sounding_pitch_boolean:
                    plt.savefig(f'{self.dump_folder}/arrangement/statistics/tessitura_sounding/' + instrument_name + '_tessitura.pdf')
                else:
                    plt.savefig(f'{self.dump_folder}/arrangement/statistics/tessitura/' + instrument_name + '_tessitura.pdf')
                plt.show()

                # Write stats in txt file
                total_num_notes = histogram.sum()
                non_zero_indices = np.nonzero(histogram)[0]
                lowest_pitch = non_zero_indices.min()
                highest_pitch = non_zero_indices.max()
                this_stats = {
                    'instrument_name': instrument_name,
                    'total_num_notes': total_num_notes,
                    'lowest_pitch': lowest_pitch,
                    'highest_pitch': highest_pitch
                }
                stats.append(this_stats)

        # Write number of co-occuring notes
        with open(f'{self.dump_folder}/arrangement/statistics/simultaneous_notes.txt', 'w') as ff:
            for instrument_name, simultaneous_counter in self.simultaneous_notes.items():
                ff.write(f"## {instrument_name}\n")
                for ind, simultaneous_occurences in enumerate(list(simultaneous_counter)):
                    ff.write('  {:d} : {:d}\n'.format(ind, int(simultaneous_occurences)))

        with open(f'{self.dump_folder}/arrangement/statistics/statistics.csv', 'w') as ff:
            fieldnames = this_stats.keys()
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for this_stats in stats:
                writer.writerow(this_stats)

    def histogram_tessitura(self, score):
        # Transpose to sounding pitch ?
        if self.sounding_pitch_boolean:
            score_processed = score.toSoundingPitch()
        else:
            score_processed = score

        for part in score_processed.parts:
            instrument_names = self.simplify_instrumentation[part.partName]
            for instrument_name in separate_instruments_names(instrument_names):
                part_flat = part.flat
                histogram = music21.graph.plot.HistogramPitchSpace(part_flat)
                histogram.extractData()
                histogram_data = histogram.data
                for (pitch, frequency, _) in histogram_data:
                    if instrument_name not in self.tessitura:
                        self.tessitura[instrument_name] = np.zeros((128,))
                    self.tessitura[instrument_name][int(pitch)] += frequency
        return

    def counting_simultaneous_notes(self, score):
        start_offset = score.flat.lowestOffset
        end_offset = score.flat.highestOffset

        with open(self.simultaneous_notes_details_path, 'a') as ff:
            ff.write(f'##### {score.filePath}\n')

        for part in score.parts:
            # Instrument name
            instrument_name = self.simplify_instrumentation[part.partName]
            # Parse file
            element_iterator = part.flat.getElementsByOffset(start_offset, end_offset,
                                                             classList=[music21.note.Note,
                                                                        music21.chord.Chord])
            # Count simultaneous notes
            notes_on = []  # Contains offset of note_off events
            for elem in element_iterator:
                current_offset = float(elem.offset)
                # Add the end of notes time to the list of notes on
                if elem.isChord:
                    notes_on.append((float(elem.offset + elem.duration.quarterLength), len(elem._notes)))
                else:
                    notes_on.append((elem.offset + elem.duration.quarterLength, 1))
                # Clean notes_on and count number of simultaneous notes
                this_simultaneous_notes = 0
                notes_on_OLD = copy.copy(notes_on)
                notes_on = []
                for note_off_offset, num_notes in notes_on_OLD:
                    if current_offset < note_off_offset:
                        notes_on.append((note_off_offset, num_notes))
                        this_simultaneous_notes += num_notes
                #  Update simultaneous notes counter
                if instrument_name not in self.simultaneous_notes.keys():
                    self.simultaneous_notes[instrument_name] = np.zeros((10,))

                if (instrument_name != 'Piano') and (this_simultaneous_notes > 2):
                    with open(self.simultaneous_notes_details_path, 'a') as ff:
                        ff.write('{} : {:f}\n'.format(instrument_name, float(current_offset)))
                self.simultaneous_notes[instrument_name][this_simultaneous_notes] += 1
        return


if __name__ == '__main__':
    database_path = '/home/leo/Recherche/Databases/Orchestration/arrangement_mxml/'
    subsets = ['liszt_classical_archives']
    simplify_instrumentation_path = 'simplify_instrumentation.json'
    sounding_pitch_boolean = True
    subdivision = 2
    computeStatistics = ComputeStatistics(database_path, subsets, subdivision, sounding_pitch_boolean)
    computeStatistics.get_statistics()
