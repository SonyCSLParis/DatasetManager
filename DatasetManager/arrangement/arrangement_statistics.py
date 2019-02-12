import math
import csv
from arrangement.arrangement_helper import ArrangementIteratorGenerator
import music21
import numpy as np
import json
import matplotlib.pyplot as plt


class ComputeStatistics:
    def __init__(self, database_path, subsets, simplify_instrumentation_path, subdivision):
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)
        #  Histogram with range of notes per instrument
        self.tessitura = dict()
        # Histogram with the number of simultaneous notes per instrument
        self.stat_dict = dict()
        self.score_iterator = ArrangementIteratorGenerator(
            arrangement_path=database_path,
            subsets=subsets
        )
        self.subdivision = subdivision
        return

    def get_statistics(self):
        for arrangement_pair in self.score_iterator():
            for score in arrangement_pair:
                self.histogram_tessitura(score)
                # self.notes_stats_per_instru(score)

        # Get reference tessitura for comparing when plotting
        with open('reference_tessitura.json') as ff:
            reference_tessitura = json.load(ff)

        stats = []
        this_stats = {}
        for instrument_name, histogram in self.tessitura.items():
            # Plot histogram for each instrument
            x = range(128)
            y = list(histogram.astype(int))
            plt.bar(x, y)
            plt.xlabel('Notes', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.title(instrument_name, fontsize=10)
            plt.xticks(np.arange(0, 128, 1))
            # Add reference tessitura

            plt.savefig('tessitura/' + instrument_name + '_tessitura.pdf')
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

        with open('statistics.csv', 'w') as ff:
            fieldnames = this_stats.keys()
            writer = csv.DictWriter(ff, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for this_stats in stats:
                writer.writerow(this_stats)

    def histogram_tessitura(self, score):
        for part in score.parts:
            instrument_name = self.simplify_instrumentation[part.partName]
            part_flat = part.flat
            histogram = music21.graph.plot.HistogramPitchSpace(part_flat)
            histogram.extractData()
            histogram_data = histogram.data
            for (pitch, frequency, _) in histogram_data:
                if instrument_name not in self.tessitura:
                    self.tessitura[instrument_name] = np.zeros((128,))
                self.tessitura[instrument_name][int(pitch)] += frequency
        return


# def notes_stats_per_instru(self, score):
#     # start_offset = score.flat.lowestOffset
#     # end_offset = score.flat.highestOffset
#     for part in score.parts:
#         instrument_name = self.simplify_instrumentation[part.partName]
#
#
#         THIS IS ONLY USEFUL FOR GETTING THE NUMBER OF SIMULTANEOUS NOTES
#         frames = [(off, [pc.pitch.pitchClass if pc.isNote else pc.pitchClasses
#                          for pc in part.flat.getElementsByOffset(off, mustBeginInSpan=False,
#                                                                  classList=[music21.note.Note,
#                                                                             music21.chord.Chord]).notes])
#                   for off in np.arange(start_offset, end_offset + 1, 1 / self.subdivision)
#                   ]
#         for frame in frames:
#
#
#         self.stat_dict[instrument_name]['number_notes'] +=


if __name__ == '__main__':
    database_path = '/home/leo/databases/Orchestration/arrangement_mxml/'
    # subsets = ['liszt_classical_archives']
    subsets = ['debug']
    simplify_instrumentation_path = 'simplify_instrumentation.json'
    subdivision = 4
    computeStatistics = ComputeStatistics(database_path, subsets, simplify_instrumentation_path, subdivision)
    computeStatistics.get_statistics()
    print("yo")
