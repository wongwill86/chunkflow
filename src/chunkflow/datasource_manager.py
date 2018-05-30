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
        # print('\n\n\noutput core chuunk')
        chunk.dump_data(self.repository.output_datasource_core, slices)


class DatasourceRepository(object):
    def __init__(self, input_datasource, output_datasource_core=None, output_datasource_overlap=None):
        self.repository = dict()
        self.input_datasource = input_datasource

        if output_datasource_core is None:
            output_datasource_core = self.create(None)

        if output_datasource_overlap is None:
            output_datasource_overlap = self.create(None)

        self.output_datasource_core = output_datasource_core
        self.output_datasource_overlap = output_datasource_overlap

        iterator = UnitIterator()

        # prepopulate
        origin = tuple([1] * len(input_datasource.shape))
        self.repository[origin] = self.create(origin)
        for neighbor in iterator.get_all_neighbors(origin):
            self.repository[neighbor] = self.create(neighbor)

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        return self.repository[get_mod_index(index)]
