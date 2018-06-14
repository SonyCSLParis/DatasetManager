import os

import music21
from music21.key import KeySignatureException
from music21.meter import TimeSignatureException

from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import exclude_list_ids, leadsheet_to_music21
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetParsingException

import numpy as np


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
        return f'{self.time_signature}{f"_{composer}" if self.composer else ""}'

    def make_score_dataset(self):
        """
        Download all LSDB leadsheets, convert them into MusicXML and write them
        in xml folder

        :return:
        """

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        # todo add query
        with LsdbMongo() as client:
            db = client.get_db()
            leadsheets = db.leadsheets.find({'_id': {
                '$nin': exclude_list_ids
            }})
            # todo remove slicing
            for leadsheet in leadsheets[:10]:
                # discard leadsheet with no title
                if 'title' not in leadsheet:
                    continue
                if os.path.exists(os.path.join(self.dataset_dir,
                                               f'{leadsheet["title"]}.xml'
                                               )):
                    print(leadsheet['title'])
                    print(leadsheet['_id'])
                    print('exists!')
                    continue
                print(leadsheet['title'])
                print(leadsheet['_id'])
                try:
                    score = leadsheet_to_music21(leadsheet)
                    export_file_name = os.path.join('xml',
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
