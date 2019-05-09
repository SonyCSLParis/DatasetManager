# Keys must match values of instrument_grouping
# 0 means instrument is not used in the db
def get_instrumentation():
    ordered_instruments = {
        "Piccolo": 1,
        "Flute": 2,
        "Oboe": 2,
        "Clarinet": 2,
        "Bassoon": 2,
        "Horn": 2,
        "Trumpet": 2,
        "Trombone": 2,
        "Tuba": 1,
        "Violin_1": 3,
        "Violin_2": 2,
        "Viola": 2,
        "Violoncello": 2,
        "Contrabass": 2,
        "Remove": 0
    }
    return ordered_instruments
