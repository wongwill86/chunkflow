from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.iterators import UnitIterator


def get_mod_index(index):
    return tuple(idx % 3 for idx in index)


class DatasourceManager(object):
    def __init__(self, repository):
        self.repository = repository

    def dump_chunk(self, chunk):
        chunk.dump_data(self.repository.get_datasource(chunk.unit_index))

    def load_chunk(self, chunk):
        chunk.load_data(self.repository.get_datasource(chunk.unit_index))

    def upload_overlap(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_overlap, slices)

    def upload_core(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_core, slices)


class DatasourceRepository(object):
    def __init__(self, input_datasource):
        self.repository = dict()
        self.input_datasource = input_datasource
        self.output_datasource_core = None
        self.output_datasource_overlap = None
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


class NumpyDatasource(DatasourceRepository):
    def __init__(self, input_datasource, *args, **kwargs):
        super().__init__(input_datasource, *args, **kwargs)
        self.output_datasource_core = self.create(None)
        self.output_datasource_overlap = self.create(None)

    def create(self, mod_index, *args, **kwargs):
        return GlobalOffsetArray(self.input_datasource.copy(), global_offset=(0, 0))
