import itertools
from concurrent.futures import ProcessPoolExecutor

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
    """
    Class to handle datasources used in chunkflow. Allows user to create buffers for input/output datasources.
    Executors can be specified to speed up data transfer

    WARNING: There is a deadlock I couldn't debug when sharing the same executor across different rx upload/downloads.
    As a heavy handed fix, simply use separate executors for each action (technically each stage) until someone figures
    out why this happens. Oddly enough it works fine when using ThreadPoolExecutor.

    WARNING: dump_executor should not be specified for now when using buffers. BlockChunkBuffer doesn't support setting
    and clearing yet.

    WARNING: Executors are not supported yet for overlap repositories for same reason above
    """
    def __init__(self, input_datasource, output_datasource, overlap_repository, output_datasource_final=None,
                 buffer_generator=None, load_executor=None, dump_executor=None, flush_executor=None):
        self.input_datasource = input_datasource
        self.output_datasource = output_datasource
        self.output_datasource_final = output_datasource_final
        self.overlap_repository = overlap_repository
        self.buffer_generator = buffer_generator
        self.datasource_buffers = dict()
        # TODO remove if someone has found a solution
        assert not (isinstance(load_executor, ProcessPoolExecutor) and load_executor is flush_executor), (
            'Using the same ProcessPoolExecutor for loading and flushing is known to cause deadlocks.'
        )
        # TODO remove when BlockChunkBuffer supports setting and clearing
        assert dump_executor is None or buffer_generator is None, (
            "Dumping data to a the BlockChunkBuffer is not yet supported"
        )
        self.load_executor = load_executor
        self.dump_executor = dump_executor
        self.flush_executor = flush_executor

    def download_input(self, chunk):
        return self.load_chunk(chunk, datasource=self.input_datasource)

    def get_buffer(self, datasource):
        datasource_key = id(datasource)
        if self.buffer_generator is not None:
            if datasource_key not in self.datasource_buffers:
                self.datasource_buffers[datasource_key] = self.buffer_generator(datasource)
            return self.datasource_buffers[datasource_key]
        return None

    def _perform_chunk_action(self, chunk_action, datasource, slices=None, executor=None):
        if executor is None:
            return chunk_action(datasource, slices=slices)
        else:
            return executor.submit(chunk_action, datasource, slices)

    def dump_chunk(self, chunk, datasource=None, slices=None):
        """
        Dump chunk data into target datasource.
        :param chunk source of chunk data to dump from
        :param datasource destination to dump data into. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from chunk to dump from. default to chunk's entire bbox if None given

        :returns: chunk if no executor is configured, otherwise returns the future returned by the executor
        """
        executor = None
        if datasource is None:
            datasource = self.overlap_repository.get_datasource(chunk.unit_index)
        else:
            buffered_datasource = self.get_buffer(datasource)
            if buffered_datasource is not None:
                datasource = buffered_datasource

            # TODO use for all repo when buffers are supported (see class warnings doc)
            executor = self.dump_executor

        return self._perform_chunk_action(chunk.dump_data, datasource, slices=slices, executor=executor)

    def load_chunk(self, chunk, datasource=None, slices=None):
        """
        Load chunk with data from datasource
        :param chunk destination chunk to load data into
        :param datasource (optional) source to load data from. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from datasource to load from. default to chunk's entire bbox if None given

        :returns: chunk if no executor is configured, otherwise returns the future returned by the executor
        """
        executor = None
        if datasource is None:
            datasource = self.overlap_repository.get_datasource(chunk.unit_index)
        else:
            # TODO maybe allow the cache for overlap repo once BlockChunkBuffer supports setting and clearing
            # Right now don't use cache for downloading data from datasource
            executor = self.load_executor

        return self._perform_chunk_action(chunk.load_data, datasource, slices=slices, executor=executor)

    def flush(self, chunk, datasource):
        datasource_buffer = self.get_buffer(datasource)
        try:
            cleared_chunk = datasource_buffer.clear(chunk)
            if cleared_chunk is not None:
                return self._perform_chunk_action(cleared_chunk.dump_data, datasource, executor=self.flush_executor)
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
