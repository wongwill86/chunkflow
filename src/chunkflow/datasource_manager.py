import itertools

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.iterators import UnitIterator


def get_absolute_index(offset, overlap, shape):
    return tuple(o // stride for o, stride in zip(
        offset, tuple(olap - s for olap, s in zip(overlap, shape))
    ))


def get_mod_index(index):
    return tuple(abs(idx % 3) for idx in index)


def get_all_mod_index(index):
    return itertools.chain([index], map(get_mod_index, UnitIterator().get_all_neighbors(index)))


class DatasourceManager:
    def __init__(self, input_datasource, output_datasource, overlap_repository, output_datasource_final=None,
                 buffer=None):
        self.input_datasource = input_datasource
        self.output_datasource = output_datasource
        self.output_datasource_final = output_datasource_final
        self.overlap_repository = overlap_repository
        self.datasource_buffer = dict()

    def download_input(self, chunk, executor=None):
        return self.load_chunk(chunk, datasource=self.input_datasource, executor=executor)

    def dump_chunk(self, chunk, datasource=None, slices=None, executor=None):
        """
        Dump chunk data into target datasource.
        :param chunk source of chunk data to dump from
        :param datasource destination to dump data into. searches for designated chunk datasource if non specified
        :param slices (optional) slices from chunk to dump from. default to chunk's entire bbox if None given
        :param executor (optional) where to schedule dump command. runs on same thread if None specified

        :returns: chunk if no executor is given, otherwise returns the future returned by the executor
        """
        if datasource is None:
            datasource = self.get_datasource(chunk.unit_index)

        # if datasource in datasource_buffer:
        #     datasource = datasource_buffer[datasource]
        # else:
        #     datasource_buffer[datasource] = self.create_buffer(datasource)

        if executor is None:
            return chunk.dump_data(datasource, slices=slices)
        else:
            return executor.submit(chunk.dump_data, datasource, slices)

    def load_chunk(self, chunk, datasource=None, slices=None, executor=None):
        """
        Load chunk with data from datasource
        :param chunk destination chunk to load data into
        :param datasource (optional) source to load data from. searches for designated chunk datasource if non specified
        :param slices (optional) slices from datasource to load from. default to chunk's entire bbox if None given
        :param executor (optional) where to schedule load command. runs on same thread if None specified

        :returns: chunk if no executor is given, otherwise returns the future returned by the executor
        """
        if datasource is None:
            datasource = self.get_datasource(chunk.unit_index)

        if executor is None:
            return chunk.load_data(datasource, slices=slices)
        else:
            return executor.submit(chunk.load_data, datasource, slices)

    def flush(self, chunk, datasource, executor=None):
        try:
            cleared_chunk = datasource.clear(chunk)
            if cleared_chunk is not None:
                # TODO this should be fixed shouldn't use ds.ds
                return self.dump_chunk(cleared_chunk, datasource.datasource, executor=executor)
        except AttributeError:
            pass

        return chunk

    def create_overlap_datasources(self, center_index):
        """
        Create all overlap datasources.
        :param center_index: this is needed to find out how many dimensions are used to find the neighbors. We are
        unable to use the saved datasources because they maybe have multi dimensional outputs.
        """
        for mod_index in get_all_mod_index(center_index):
            self.get_datasource(mod_index)

    @property
    def overlap_datasources(self):
        return self.overlap_repository.overlap_datasources

    def get_datasource(self, index):
        return self.overlap_repository.get_datasource(index)

    def clear(self, chunk):
        return self.overlap_repository.clear(chunk.unit_index)


class OverlapRepository:
    def __init__(self, *args, **kwargs):
        self.overlap_datasources = dict()

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        mod_index = get_mod_index(index)
        if mod_index not in self.overlap_datasources:
            self.overlap_datasources[mod_index] = self.create(mod_index)
        return self.overlap_datasources[mod_index]

    def clear(self, index):
        mod_index = get_mod_index(index)
        return self.overlap_datasources.pop(mod_index)


class SparseOverlapRepository(OverlapRepository):
    def __init__(self, block, channel_dimensions, dtype, *args, **kwargs):
        self.block = block
        self.channel_dimensions = channel_dimensions
        self.dtype = dtype
        super().__init__(*args, **kwargs)

    def get_datasource(self, index):
        if index not in self.overlap_datasources:
            self.overlap_datasources[index] = self.create(index)
        return self.overlap_datasources[index]

    def create(self, index, *args, **kwargs):
        global_offset = (0,) + tuple(s.start for s in self.block.unit_index_to_slices(index))
        return GlobalOffsetArray(
            np.zeros(self.channel_dimensions + self.block.chunk_shape, dtype=self.dtype),
            global_offset=global_offset,
        )

    def clear(self, index):
        del self.overlap_datasources[index]
