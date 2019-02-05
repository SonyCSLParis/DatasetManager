import os
import glob


class ArrangementIteratorGenerator:
    """
    Object that returns a iterator over xml files when called
    :return:
    """

    # todo redo
    def __init__(self, arrangement_path, subsets, num_elements=None):
        self.arrangement_path = arrangement_path
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
                os.path.join(self.arrangement_path, subset, '[0-9]*/*.xml')))
        if self.num_elements is not None:
            arrangement_paths = arrangement_paths[:self.num_elements]
        for arrangement_path in arrangement_paths:
            try:
                if len(arrangement_path) != 2:
                    raise Exception(f'There should be files in {arrangement_path}' )
                print(arrangement_path)
                # Here parse files and return as a dict containing matrices for piano and orchestra
                arrangement_pair = process(arrangement_path)
                yield arrangement_pair
            except Exception as e:
                print(f'{arrangement_path} is not parsable')
                print(e)
