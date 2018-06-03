from chunkflow.iterators import UnitIterator


def get_mod_index(index):
    return tuple(idx % 3 for idx in index)


class DatasourceManager(object):
    def __init__(self, repository):
        self.repository = repository

    def download_input(self, chunk):
        chunk.load_data(self.repository.input_datasource)

    def dump_chunk(self, chunk):
        chunk.dump_data(self.repository.get_datasource(chunk.unit_index))

    def load_chunk(self, chunk):
        chunk.load_data(self.repository.get_datasource(chunk.unit_index))

    def upload_output_overlap(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_overlap, slices)

    def upload_output_core(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_core, slices)


class DatasourceRepository(object):
    def __init__(self, input_datasource, output_datasource_core, output_datasource_overlap, index_dimensions,
                 repository=None):
        self.repository = dict()
        self.input_datasource = input_datasource
        self.output_datasource_core = output_datasource_core
        self.output_datasource_overlap = output_datasource_overlap

        if repository is None:
            # prepopulate
            iterator = UnitIterator()
            origin = tuple([1] * index_dimensions)
            self.repository[origin] = self.create(origin)
            for neighbor in iterator.get_all_neighbors(origin):
                self.repository[neighbor] = self.create(neighbor)
        else:
            self.repository = repository

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        return self.repository[get_mod_index(index)]
