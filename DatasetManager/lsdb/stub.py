import music21

if __name__ == '__main__':
    # chord_symbol = music21.harmony.chordSymbolFromChord(chord)
    # print(chord_symbol)
    # <music21.harmony.ChordSymbol Csus>
    # sus4
    # chord = music21.chord.Chord(['G4', 'B-4', 'C5', 'F5'])
    chord = music21.chord.Chord(['C4', 'F4', 'G4'])

    # 7sus4
    chord = music21.chord.Chord(['C4', 'F4', 'G4', 'B-4'])
    chord.root('C4')
    chord.bass('C4')
    # chord = music21.chord.Chord(['D4', 'G4', 'A4', 'C5'])
    # chord.root('D4')
    # chord = music21.chord.Chord(['C4', 'E4', 'G4', 'A4'])

    # minor-major
    # chord = music21.chord.Chord(['C4', 'E-4', 'G4', 'B4'])
    # chord.root('C4')

    # 9th 13th ...
    # chord = music21.chord.Chord(['C4', 'E4', 'G4', 'B4', 'D5'])
    # chord.root('C4')

    # chord = music21.chord.Chord(['C4', 'F4', 'G4', 'B-4', 'D5'])
    # chord = music21.chord.Chord(['D4',  'F#4', 'A4', 'C5', 'E-5'])
    # chord.root('D4')
    chord_symbol = music21.harmony.chordSymbolFromChord(chord)
    print(chord_symbol)
    chord_symbol_figure = music21.harmony.chordSymbolFigureFromChord(chord, True)
    print(chord_symbol_figure)
    print(chord_symbol.pitches)

    interval = music21.interval.Interval('m3')
    chord_symbol_transposed = chord_symbol.transpose(interval)
    print(chord_symbol_transposed.root())
    print(chord_symbol_transposed)
    print(chord_symbol_transposed.pitches)

    MEX = music21.musicxml.m21ToXml.MeasureExporter()

    mxHarmony = MEX.chordSymbolToXml(chord_symbol_transposed)
    MEX.dump(mxHarmony)

    chord_symbol.transpose(interval, inPlace=True)
    print(chord_symbol.root())
    print(chord_symbol)
    print(chord_symbol.pitches)

    mxHarmony = MEX.chordSymbolToXml(chord_symbol)
    MEX.dump(mxHarmony)
    exit()

#   File "music21/harmony.py", line 1826, in _parseFigure
# if int(justints) > 20: # MSC: what is this doing?
# ValueError: invalid literal for int() with base 10: 'hordSym'
