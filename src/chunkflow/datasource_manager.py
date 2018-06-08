def get_mod_index(index):
    return tuple(idx % 3 for idx in index)


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

    def upload_output_overlap(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_overlap, slices)

    def upload_output_core(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_core, slices)

    @property
    def output_datasource_core(self):
        return self.repository.output_datasource_core

    @property
    def output_datasource_overlap(self):
        return self.repository.output_datasource_overlap


class DatasourceRepository:
    def __init__(self, input_datasource, output_datasource_core, output_datasource_overlap,
                 intermediate_datasources=None):
        self.input_datasource = input_datasource
        self.output_datasource_core = output_datasource_core
        self.output_datasource_overlap = output_datasource_overlap
        self.intermediate_datasources = dict()

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        mod_index = get_mod_index(index)
        if mod_index not in self.intermediate_datasources:
            self.intermediate_datasources[mod_index] = self.create(mod_index)

        return self.intermediate_datasources[mod_index]
