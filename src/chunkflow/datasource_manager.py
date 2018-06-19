import itertools

from chunkblocks.iterators import UnitIterator


def get_mod_index(index):
    return tuple(abs(idx % 3) for idx in index)


def get_all_mod_index(index):
    return itertools.chain([index], map(get_mod_index, UnitIterator().get_all_neighbors(index)))


class DatasourceManager:
    def __init__(self, repository):
        self.repository = repository

    def download_input(self, chunk):
        chunk.load_data(self.repository.input_datasource)

    def dump_chunk(self, chunk, datasource=None, slices=None):
        if datasource is None:
            datasource = self.repository.get_datasource(chunk.unit_index)
        chunk.dump_data(datasource, slices)

    def load_chunk(self, chunk, datasource=None, slices=None):
        if datasource is None:
            datasource = self.repository.get_datasource(chunk.unit_index)
        chunk.load_data(datasource, slices)

    @property
    def input_datasource(self):
        return self.repository.input_datasource

    @property
    def output_datasource(self):
        return self.repository.output_datasource

    @property
    def output_datasource_overlap(self):
        return self.repository.output_datasource_overlap


class DatasourceRepository:
    def __init__(self, input_datasource, output_datasource, output_datasource_overlap,
                 intermediate_datasources=None):
        self.input_datasource = input_datasource
        self.output_datasource = output_datasource
        self.output_datasource_overlap = output_datasource_overlap
        self.intermediate_datasources = dict()

    def create_intermediate_datasources(self, center_index):
        """
        Create all intermediate datasources.
        :param center_index: this is needed to find out how many dimensions are used to find the neighbors. We are
        unable to use the saved datasources because they maybe have multi dimensional outputs.
        """
        for mod_index in get_all_mod_index(center_index):
            self.get_datasource(mod_index)

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        mod_index = get_mod_index(index)
        if mod_index not in self.intermediate_datasources:
            self.intermediate_datasources[mod_index] = self.create(mod_index)
        return self.intermediate_datasources[mod_index]
