import os

import music21
from music21.key import KeySignatureException
from music21.meter import TimeSignatureException

from DatasetManager.lsdb.LsdbMongo import LsdbMongo
from DatasetManager.lsdb.lsdb_data_helpers import exclude_list_ids, altered_pitches_music21_to_dict, \
    assert_no_time_signature_changes, set_metadata
from DatasetManager.lsdb.lsdb_exceptions import LeadsheetParsingException

import numpy as np


def make_score_dataset():
    """
    Download all LSDB leadsheets, convert them into MusicXML and write them
    in xml folder

    :return:
    """
    if not os.path.exists('xml'):
        os.mkdir('xml')

    # todo add query
    with LsdbMongo() as client:
        db = client.get_db()
        leadsheets = db.leadsheets.find({'_id': {
            '$nin': exclude_list_ids
        }})

        for leadsheet in leadsheets[:10]:
            # discard leadsheet with no title
            if 'title' not in leadsheet:
                continue
            if os.path.exists(os.path.join('xml',
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


def leadsheet_to_music21(leadsheet):
    # must convert b to -
    if 'keySignature' not in leadsheet:
        raise KeySignatureException(f'Leadsheet {leadsheet["title"]} '
                                    f'has no keySignature')
    key_signature = leadsheet['keySignature'].replace('b', '-')
    key_signature = music21.key.Key(key_signature)

    altered_pitches_at_key = altered_pitches_music21_to_dict(
        key_signature.alteredPitches)

    if leadsheet["time"] != '4/4':
        raise TimeSignatureException('Leadsheet ' + leadsheet['title'] + ' ' +
                                     str(leadsheet['_id']) +
                                     ' is not in 4/4')
    if 'changes' not in leadsheet:
        raise LeadsheetParsingException('Leadsheet ' + leadsheet['title'] + ' ' +
                                        str(leadsheet['_id']) +
                                        ' do not contain "changes" attribute')
    assert_no_time_signature_changes(leadsheet)

    chords = []
    notes = []

    score = music21.stream.Score()
    part_notes = music21.stream.Part()
    part_chords = music21.stream.Part()
    for section_index, section in enumerate(leadsheet['changes']):
        for bar_index, bar in enumerate(section['bars']):
            # We consider only 4/4 pieces
            # Chords in bar
            chords_in_bar = chords_in_bar(bar)
            notes_in_bar = notes_in_bar(bar,
                                             altered_pitches_at_key)
            chords.extend(chords_in_bar)
            notes.extend(notes_in_bar)

    # remove FakeNotes
    notes = remove_fake_notes(notes)
    chords = remove_rest_chords(chords)

    # voice_notes = music21.stream.Voice()
    # voice_chords = music21.stream.Voice()
    # todo there might be a cleaner way to do this
    part_notes.append(notes)
    part_chords.append(chords)
    for chord in part_chords.flat.getElementsByClass(
            [music21.harmony.ChordSymbol,
             music21.expressions.TextExpression
             ]):
        # put durations to 0.0 as required for a good rendering
        # handles both textExpression (for N.C.) and ChordSymbols
        if isinstance(chord, music21.harmony.ChordSymbol):
            new_chord = music21.harmony.ChordSymbol(chord.figure)
        elif isinstance(chord, music21.expressions.TextExpression):
            new_chord = music21.expressions.TextExpression(NC)
        else:
            raise ValueError
        part_notes.insert(chord.offset, new_chord)
    # new_chord = music21.harmony.ChordSymbol(chord.figure)
    # part_notes.insert(chord.offset, chord)
    # part_chords.append(chords)
    # voice_notes.append(notes)
    # voice_chords.append(chords)
    # part = music21.stream.Part()
    # part.insert(0, voice_notes)
    # part.insert(0, voice_chords)
    # score.append((part_notes, part_chords))
    # score.append(part)

    part_notes = part_notes.makeMeasures(
        inPlace=False,
        refStreamOrTimeRange=[0.0, part_chords.highestTime])

    # add treble clef and key signature
    part_notes.measure(1).clef = music21.clef.TrebleClef()
    part_notes.measure(1).keySignature = key_signature
    score.append(part_notes)
    set_metadata(score, leadsheet)
    # normally we should use this but it does not look good...
    # score = music21.harmony.realizeChordSymbolDurations(score)

    return score

