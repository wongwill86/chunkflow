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


from concurrent.futures import ProcessPoolExecutor


class DatasourceManager:
    def __init__(self, input_datasource, output_datasource, overlap_repository, output_datasource_final=None,
                 buffer_generator=None, executor=None):
        self.input_datasource = input_datasource
        self.output_datasource = output_datasource
        self.output_datasource_final = output_datasource_final
        self.overlap_repository = overlap_repository
        self.buffer_generator = buffer_generator
        self.datasource_buffers = dict()
        self.executor = executor
        self.other_exec = ProcessPoolExecutor()
        self.ops = []

    def download_input(self, chunk):
        return self.load_chunk(chunk, datasource=self.input_datasource)

    def get_buffer(self, datasource):
        datasource_key = id(datasource)
        if self.buffer_generator is not None:
            if datasource_key not in self.datasource_buffers:
                self.datasource_buffers[datasource_key] = self.buffer_generator(datasource)
            return self.datasource_buffers[datasource_key]
        return datasource

    def get_key(self, tchunk, action, ds, slices):
        if slices is None:
            slices = tchunk.slices
        s_key = tuple((s.start, s.stop) if s is not None else s for s in slices)
        return (action.__func__.__name__, ds, tchunk.unit_index)

    def get_mark_done(self, tchunk, action, ds, slices):
        def inner(something):
            key = self.get_key(tchunk, action, ds.layer_cloudpath, slices)
            self.ops.remove(key)
            print('done exec, remaining:', key, 'remaining now has', len(self.ops), self.ops, self.executor._call_queue.qsize())
        return inner

    def _perform_chunk_action(self, tchunk, chunk_action, datasource, slices=None, executor=None):
        if executor is None:
            return chunk_action(datasource, slices=slices)
        else:
            key = self.get_key(tchunk, chunk_action, datasource.layer_cloudpath, slices)
            self.ops.append(key)
            print('submit exec for', key, 'now contain int:', 'now has', len(self.ops), self.ops,
                  self.executor._call_queue.qsize())
            # return executor.submit(chunk_action, datasource, slices)
            fut = executor.submit(chunk_action, datasource, slices)
            fut.add_done_callback(self.get_mark_done(tchunk, chunk_action, datasource, slices))
            return fut

    def dump_chunk(self, chunk, datasource=None, slices=None):
        """
        Dump chunk data into target datasource.
        :param chunk source of chunk data to dump from
        :param datasource destination to dump data into. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from chunk to dump from. default to chunk's entire bbox if None given
        :param executor (optional) where to schedule dump command. runs on same thread if None specified

        :returns: chunk if no executor is given, otherwise returns the future returned by the executor
        """
        if datasource is None:
            datasource = self.overlap_repository.get_datasource(chunk.unit_index)
        else:
            # TODO maybe allow the cache for overlap repo once BlockChunkBuffer supports setting and clearing
            buffer_datasource = self.get_buffer(datasource)
            if buffer_datasource is not None:
                datasource = buffer_datasource

        return self._perform_chunk_action(chunk, chunk.dump_data, datasource, slices=slices)

    def load_chunk(self, chunk, datasource=None, slices=None):
        """
        Load chunk with data from datasource
        :param chunk destination chunk to load data into
        :param datasource (optional) source to load data from. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from datasource to load from. default to chunk's entire bbox if None given
        :param executor (optional) where to schedule load command. runs on same thread if None specified

        :returns: chunk if no executor is given, otherwise returns the future returned by the executor
        """
        import os
        print('calling load chunk ', chunk.unit_index, 'from', datasource.layer_cloudpath, os.getpid())
        executor = None
        if datasource is None:
            datasource = self.overlap_repository.get_datasource(chunk.unit_index)
        else:
            # TODO maybe allow the cache for overlap repo once BlockChunkBuffer supports setting and clearing
            # Right now don't use cache for downloading data from datasource
            executor = self.executor

        return self._perform_chunk_action(chunk, chunk.load_data, datasource, slices=slices, executor=executor)

    def flush(self, chunk, datasource):
        datasource_buffer = self.get_buffer(datasource)
        try:
            cleared_chunk = datasource_buffer.clear(chunk)
            if cleared_chunk is not None:
                return self._perform_chunk_action(chunk, cleared_chunk.dump_data, datasource, executor=self.executor)
        except AttributeError:
            # Not a buffered datasource, no flush needed
            pass

        return chunk

    def create_overlap_datasources(self, center_index):
        """
        Create all overlap datasources.
        :param center_index: this is needed to find out how many dimensions are used to find the neighbors. We are
        unable to use the saved datasources because they maybe have multi dimensional outputs.
        """
        for mod_index in get_all_mod_index(center_index):
            self.overlap_repository.get_datasource(mod_index)


class OverlapRepository:
    def __init__(self, *args, **kwargs):
        self.datasources = dict()

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def get_datasource(self, index):
        mod_index = get_mod_index(index)
        if mod_index not in self.datasources:
            self.datasources[mod_index] = self.create(mod_index)
        return self.datasources[mod_index]

    def clear(self, index):
        mod_index = get_mod_index(index)
        return self.datasources.pop(mod_index)


class SparseOverlapRepository(OverlapRepository):
    def __init__(self, block, channel_dimensions, dtype, *args, **kwargs):
        self.block = block
        self.channel_dimensions = channel_dimensions
        self.dtype = dtype
        super().__init__(*args, **kwargs)

    def create(self, index, *args, **kwargs):
        global_offset = (0,) + tuple(s.start for s in self.block.unit_index_to_slices(index))
        return GlobalOffsetArray(
            np.zeros(self.channel_dimensions + self.block.chunk_shape, dtype=self.dtype),
            global_offset=global_offset,
        )

    def get_datasource(self, index):
        if index not in self.datasources:
            self.datasources[index] = self.create(index)
        return self.datasources[index]

    def clear(self, index):
        del self.datasources[index]
