from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.iterators import UnitIterator


def get_mod_index(index):
    return tuple(idx % 3 for idx in index)


class DatasourceManager(object):
    def __init__(self, input_datasource):
        self.repository = dict()
        self.input_datasource = input_datasource
        self.output_datasource_core = None
        self.output_datasource_edge = None
        iterator = UnitIterator()

        # prepopulate
        origin = tuple([1] * len(input_datasource.shape))
        self.repository[origin] = self.create(origin)
        for neighbor in iterator.get_all_neighbors(origin):
            self.repository[neighbor] = self.create(neighbor)

    def create(self, mod_index, *args, **kwargs):
        raise NotImplemented

    def get_datasource(self, index):
        return self.repository[get_mod_index(index)]


class NumpyDatasource(DatasourceManager):
    def create(self, mod_index, *args, **kwargs):
        return GlobalOffsetArray(self.input_datasource.copy(), global_offset=(0, 0))
