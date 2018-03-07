import music21

if __name__ == '__main__':
	# chord_symbol = music21.harmony.chordSymbolFromChord(chord)
	# print(chord_symbol)
	# <music21.harmony.ChordSymbol Csus>
	# chord = music21.chord.Chord(['G4', 'B-4', 'C5', 'F5'])
	chord = music21.chord.Chord(['C4', 'F4', 'G4'])
	chord = music21.chord.Chord(['C4', 'F4', 'G4', 'B-4'])
	chord.root('C4')
	# chord = music21.chord.Chord(['C4', 'F4', 'G4', 'B-4', 'D5'])
	# chord = music21.chord.Chord(['D4',  'F#4', 'A4', 'C5', 'E-5'])
	# chord.root('D4')
	chord_symbol = music21.harmony.chordSymbolFromChord(chord)
	print(chord_symbol)
	chord_symbol_figure = music21.harmony.chordSymbolFigureFromChord(chord, True)
	print(chord_symbol_figure)
	chord_symbol_transposed = chord_symbol.transpose(1)
	print(chord_symbol_transposed.root())
	print(chord_symbol_transposed)

	#   File "music21/harmony.py", line 1826, in _parseFigure
    # if int(justints) > 20: # MSC: what is this doing?
	# ValueError: invalid literal for int() with base 10: 'hordSym'