from arrangement.arrangement_helper import ArrangementIteratorGenerator


def list_instruments_name(database_path):
    score_iterator = ArrangementIteratorGenerator(
        arrangement_path=database_path,
        subsets=['liszt_classical_archives']
    )
    parts = set()
    instruments = set()
    for arrangement_pair in score_iterator():
        for score in arrangement_pair:
            part_names, instru_names = get_names_score(score)
            parts = parts.union(part_names)
            instruments = instruments.union(instru_names)
    return parts, instruments


def get_names_score(score):
    list_instrus = []
    list_parts = []
    for part in score.parts:
        list_instrus.append(part.getInstrument().instrumentName)
        list_parts.append(part.partName)
    return set(list_parts), set(list_instrus)


def write_sets(ss, out_file):
    ll = list(ss)
    ll.sort()
    with open(out_file, 'w') as ff:
        for elem in ll:
            ff.write(f'{elem}\n')
    return


if __name__ == '__main__':
    database_path = '/home/leo/databases/Orchestration/arrangement_mxml/'
    parts, instruments = list_instruments_name(database_path)
    write_sets(parts, 'parts.txt')
    write_sets(instruments, 'instruments.txt')