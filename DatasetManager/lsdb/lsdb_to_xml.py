import os
import re

import music21
from music21.key import KeySignatureException
from music21.meter import TimeSignatureException

from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import exclude_list_ids, leadsheet_to_music21
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetParsingException

import numpy as np


# todo as method
class LsdbConverter:
    """
    Object to handle the creation of local xml databases from LSDB
    """

    # todo other mongodb queries?
    # todo num_elements ...
    def __init__(self, time_signature='4/4', composer=None):
        self.time_signature = time_signature
        self.composer = composer
        self.dataset_dir = os.path.join('xml',
                                        self.__repr__())

    def __repr__(self):
        return f'{self.time_signature.replace("/","_")}' \
               f'{f"_{composer}" if self.composer else ""}'

    def make_score_dataset(self):
        """
        Download all LSDB leadsheets, convert them into MusicXML and write them
        in xml folder

        :return:
        """

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        (lsdb_chord_to_notes,
         notes_to_chord_lsdb) = self.compute_lsdb_chord_dicts()

        # todo add query
        with LsdbMongo() as client:
            db = client.get_db()
            leadsheets = db.leadsheets.find({'_id': {
                '$nin': exclude_list_ids,
            }})
            # todo remove slicing
            for leadsheet in leadsheets[:10]:
                # discard leadsheet with no title
                if 'title' not in leadsheet:
                    continue
                # if os.path.exists(os.path.join(self.dataset_dir,
                #                                f'{leadsheet["title"]}.xml'
                #                                )):
                #     print(leadsheet['title'])
                #     print(leadsheet['_id'])
                #     print('exists!')
                #     continue
                print(leadsheet['title'])
                print(leadsheet['_id'])
                if not leadsheet['title'] == 'After The Rain':
                    continue
                try:
                    score = leadsheet_to_music21(leadsheet,
                                                 lsdb_chord_to_notes)
                    export_file_name = os.path.join(self.dataset_dir,
                                                    f'{score.metadata.title}.xml'
                                                    )

                    score.write('xml', export_file_name)

                except (KeySignatureException,
                        TimeSignatureException,
                        LeadsheetParsingException) as e:
                    print(e)

    def assert_leadsheet_in_dataset(self, leadsheet):
        if leadsheet['time'] != self.time_signature:
            raise TimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
                                         str(leadsheet['_id']) +
                                         f' is not in {self.time_signature}')
    @staticmethod
    def correct_chord_dicts(chord2notes, notes2chord):
        """
        Modifies chord2notes and notes2chord in place
        to correct errors in LSDB modes (dict of chord symbols with notes)
        :param chord2notes:
        :param notes2chord:
        :return:
        """
        # Add missing chords
        # b5
        notes2chord[('C4', 'E4', 'Gb4')] = notes2chord[('C4', 'E4', 'Gb4')] + ['b5']
        chord2notes['b5'] = ('C4', 'E4', 'Gb4')
        # b9#5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'D#5')] = 'b9#b'
        chord2notes['b9#5'] = ('C4', 'E4', 'G#4', 'Bb4', 'D#5')
        # 7#5#11 is WRONG in the database
        # C4 F4 G#4 B-4 D5 instead of C4 E4 G#4 B-4 D5
        notes2chord[('C4', 'E4', 'G#4', 'Bb4', 'F#5')] = '7#5#11'
        chord2notes['7#5#11'] = ('C4', 'E4', 'G#4', 'Bb4', 'F#5')
        # F#7#9#11 is WRONG in the database

    def compute_lsdb_chord_dicts(self):
        # Search LSDB for chord names
        with LsdbMongo() as mongo_client:
            db = mongo_client.get_db()
            modes = db.modes
            cursor_modes = modes.find({})
            chord2notes = {}  # Chord to notes dictionary
            notes2chord = {}  # Notes to chord dictionary
            for chord in cursor_modes:
                notes = []
                # Remove white spaces from notes string
                for note in re.compile("\s*,\s*").split(chord["chordNotes"]):
                    notes.append(note)
                notes = tuple(notes)

                # Enter entries in dictionaries
                chord2notes[chord['mode']] = notes
                if notes in notes2chord:
                    notes2chord[notes] = notes2chord[notes] + [chord["mode"]]
                else:
                    notes2chord[notes] = [chord["mode"]]

            self.correct_chord_dicts(chord2notes, notes2chord)

            return chord2notes, notes2chord


if __name__ == '__main__':
    LsdbConverter().make_score_dataset()
