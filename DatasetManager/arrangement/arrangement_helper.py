import os
import glob
import music21


class ArrangementIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    # todo redo
    def __init__(self, arrangement_path, subsets, num_elements=None):
        self.arrangement_path = arrangement_path  # Root of the database
        self.subsets = subsets
        self.num_elements = num_elements

    def __call__(self, *args, **kwargs):
        it = (
            xml_file
            for xml_file in self.arrangement_generator()
        )
        return it

    def arrangement_generator(self):
        arrangement_paths = []
        for subset in self.subsets:
            # Should return pairs of files
            arrangement_paths += (glob.glob(
                os.path.join(self.arrangement_path, subset, '[0-9]*')))
        if self.num_elements is not None:
            arrangement_paths = arrangement_paths[:self.num_elements]
        for arrangement_path in arrangement_paths:
            try:
                xml_files = glob.glob(arrangement_path + '/*.xml')
                midi_files = glob.glob(arrangement_path + '/*.mid')
                if not((len(xml_files) == 2) != (len(midi_files) == 2)):
                    raise Exception(f'There should be 2 midi or xml files in {arrangement_path}')
                if len(xml_files) == 2:
                    music_files = xml_files
                else:
                    music_files = midi_files
                print(music_files)
                # Here parse files and return as a dict containing matrices for piano and orchestra
                # arrangement_pair = process(xml_files)
                arrangement_pair = music21.converter.parse(music_files[0]), \
                    music21.converter.parse(music_files[1])
                yield (arrangement_pair[0], arrangement_pair[1])
            except Exception as e:
                print(f'{music_files} is not parsable')
                print(e)