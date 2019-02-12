from music21 import corpus

if __name__=='__main__':
    for chorale_id, chorale in enumerate(corpus.chorales.Iterator()):
        print(chorale)