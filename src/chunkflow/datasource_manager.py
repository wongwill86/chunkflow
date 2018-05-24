from functools import partial

from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.iterators import UnitIterator

# from rx.Subjects import Subject


def get_mod_index(index):
    return tuple(idx % 3 for idx in index)


class DatasourceManager(object):
    def __init__(self, repository):
        self.repository = repository
        # self.upload = Subject()
        # self.upload.flat_map(
        #     lambda chunk:
        #     Observable.merge(
        #         Observable.combine_latest(
        #             Observable.just(chunk).flat_map(block.overlap_slices),
        #             Observable.just(self.datasource_manager.output_datasource_overlap),
        #             lambda x, y : (x, y)),
        #         Observable.just(chunk).map(block.core_slices).zip(
        #             Observable.just(self.datasource_manager.output_datasource_core),
        #             lambda x, y : (x, y))
        # ).do_action(lambda slice_datasource: chunk_datasource[0].dump_data(chunk_datasource[1]))

    def dump_chunk(self, chunk):
        chunk.dump_data(self.repository.get_datasource(chunk.unit_index))

    def load_chunk(self, chunk):
        chunk.load_data(self.repository.get_datasource(chunk.unit_index))

    def upload_overlap(self, chunk, slices):
        chunk.dump_data(self.repository.output_datasource_overlap, slices)

    def upload_core(self, chunk, slices):
        print('\t\t\t\t\t\t\t\t\t\t\ttrying to write to %s' % (slices,))
        print('\t\t\t\t\tsum of output_datasource before is %s %s' % (chunk.data.sum(),
                                                           self.repository.output_datasource_core.sum()))
        chunk.dump_data(self.repository.output_datasource_core, slices)
        print('\t\t\t\t\tsum of output_datasource after is %s %s' % (chunk.data.sum(),
                                                          self.repository.output_datasource_core.sum()))

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

