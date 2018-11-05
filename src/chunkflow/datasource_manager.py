import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.iterators import UnitIterator
from concurrent.futures import ThreadPoolExecutor, wait
from chunkflow.memory_utils import print_memory
from chunkblocks.global_offset_array import GLOB, GLOB_BASE
import psutil
import objgraph
from memory_profiler import profile
import gc
import weakref


def get_absolute_index(offset, overlap, shape):
    return tuple(o // stride for o, stride in zip(
        offset, tuple(olap - s for olap, s in zip(overlap, shape))
    ))


def get_mod_index(index):
    return tuple(abs(idx % 3) for idx in index)


def get_all_mod_index(index):
    return itertools.chain([index], map(get_mod_index, UnitIterator().get_all_neighbors(index)))


def get_mem_info(name, keys, datas):
    keys = list(keys)
    keys.sort()
    infos = [(x.shape, x.dtype, x.nbytes) for x in datas if hasattr(x, 'nbytes')]
    memory_used = sum(info[2] for info in infos) / 2. ** 30
    if len(set(infos)) > 1:
        print('\n\n\nTHIS SHOULD NOT happen should not have more than one type of data... yet', infos)
    print('%s contains %s/%s (Futures ?: %s),entries of shape %s, total memory: %.3f GiB, entries:%s' % (
        name, len(infos), len(keys), len(keys) - len(infos),
        infos[0][1] if len(infos) else {}, memory_used, []
    ))
    return memory_used


def get_buffer_info(name, chunk_buffer):
    if not chunk_buffer:
        return 0
    if  len(chunk_buffer.local_cache) == 0:
        keys = datas = []
    else:
        keys, chunks = zip(*chunk_buffer.local_cache.items())
        datas = [chunk.data for chunk in chunks if hasattr(chunk, 'data')]
        # keys, datas = zip(*chunk_buffer.local_cache.items())
        # datas = [data for data in datas]
    return get_mem_info(name, keys, datas)


def do_load(chunk, datasource, slices):
    return chunk.load_data(datasource, slices)


def do_dump(chunk, datasource, slices):
    return chunk.dump_data(datasource, slices)


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
        assert dump_executor is None or buffer_generator is None, (
            "Dumping to an in memory buffer using multi-process will not work properly"
        )
        self.load_executor = load_executor
        self.dump_executor = dump_executor
        self.flush_executor = flush_executor
        self.runner = ThreadPoolExecutor(max_workers=12)

    def download_input(self, chunk):
        return self.load_chunk(chunk, datasource=self.input_datasource)

    def get_buffer(self, datasource):
        if self.buffer_generator is not None:
            datasource_key = id(datasource)
            if datasource_key not in self.datasource_buffers:
                self.datasource_buffers[datasource_key] = self.buffer_generator(datasource)
            return self.datasource_buffers[datasource_key]
        return None


    # def _perform_chunk_action(self, action, chunk_action, datasource, slices=None, executor=None):
    def _perform_chunk_action(self, action, chunk, datasource, slices=None, executor=None):
        if True or executor is None:
            return action(chunk, datasource, slices=slices)
        else:
            return executor.submit(action, chunk, datasource, slices)
            # def run_in_executor(executor, chunk_action, datasource, slices):
            def run_in_executor(executor, action, chunk, datasource, slices):
                # thread_future = executor.submit(chunk_action, datasource, slices)
                thread_future = executor.submit(action, chunk, datasource, slices)
                done, not_done = wait([thread_future])
                # objgraph.show_backrefs([thread_future], filename='futs/future%s.png' % id(thread_future))
                # print('showing omost common types for ', id(thread_future))
                # objgraph.show_most_common_types(objects=[thread_future])
                ret = done.pop().result()
                if hasattr(ret, 'data') and ret.data is not None:
                    chunk_copy = ret.block.unit_index_to_chunk(ret.unit_index)
                    if hasattr(chunk_copy, 'block'):
                        del chunk_copy.block
                    chunk_copy.data = ret.data.copy()
                    del ret
                    del thread_future
                    ret = chunk_copy
                return ret

            # return self.runner.submit(run_in_executor, executor, chunk_action, datasource, slices)
            return self.runner.submit(run_in_executor, executor, action, chunk, datasource, slices)


    def dump_chunk(self, chunk, datasource=None, slices=None, use_buffer=True, use_executor=True):
        """
        Dump chunk data into target datasource.
        :param chunk source of chunk data to dump from
        :param datasource destination to dump data into. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from chunk to dump from. default to chunk's entire bbox if None given

        :returns: chunk if no executor is configured, otherwise returns the future returned by the executor
        """
        buffered_datasource = self.get_buffer(datasource) if use_buffer else None
        if buffered_datasource is not None:
            datasource = buffered_datasource

        executor = self.dump_executor if use_executor else None

        return self._perform_chunk_action(do_dump, chunk, datasource, slices=slices, executor=executor)

    def load_chunk(self, chunk, datasource=None, slices=None, use_buffer=True, use_executor=True):
        """
        Load chunk with data from datasource
        :param chunk destination chunk to load data into
        :param datasource (optional) source to load data from. searches for designated chunk datasource if non specified
            if not specified, will search for corresponding overlap datasource (HACK this does NOT use the cache buffer
            at the moment)
        :param slices (optional) slices from datasource to load from. default to chunk's entire bbox if None given

        :returns: chunk if no executor is configured, otherwise returns the future returned by the executor
        """
        buffered_datasource = self.get_buffer(datasource) if use_buffer else None
        if buffered_datasource is not None:
            datasource = buffered_datasource

        executor = self.load_executor if use_executor else None

        return self._perform_chunk_action(do_load, chunk, datasource, slices=slices, executor=executor)

    def clear_buffer(self, datasource, chunk=None):
        datasource_buffer = self.get_buffer(datasource)
        try:
            if chunk is None:
                return datasource_buffer.clear()
            else:
                return [
                    datasource_buffer.clear(datasource_chunk) for datasource_chunk in
                    datasource_buffer.block.slices_to_unit_index(chunk.slices)
                ]
        except AttributeError:
            # Not a buffered datasource, no flush needed
            pass

        return None

    # TODO try reorganize this to delete
    def flush(self, chunk, datasource):
        datasource_buffer = self.get_buffer(datasource)
        try:
            cleared_chunk = datasource_buffer.clear(chunk)
        except (AttributeError, KeyError):
            # Not a buffered datasource, no flush needed
            pass
        else:
            try:
                if cleared_chunk is not None:
                    return self._perform_chunk_action(do_dump, cleared_chunk, datasource, executor=self.flush_executor)
            except AttributeError:
                # cleared chunk doesn't have dump_data must be a list
                return [
                    self._perform_chunk_action(do_dump, chunk, datasource, executor=self.flush_executor) for chunk in
                    cleared_chunk
                ]

        return chunk

    def copy(self, chunk, source, destination, slices=None):
        buffered_datasource = self.get_buffer(destination)
        if buffered_datasource is not None:
            destination = buffered_datasource
        return chunk.copy_data(source=source, destination=destination, slices=slices)

    def create_overlap_datasources(self, center_index):
        """
        Create all overlap datasources.
        :param center_index: this is needed to find out how many dimensions are used to find the neighbors. We are
        unable to use the saved datasources because they maybe have multi dimensional outputs.
        """
        for mod_index in get_all_mod_index(center_index):
            self.overlap_repository.get_datasource(mod_index)

    def get_overlap_datasource(self, chunk):
        datasource = self.overlap_repository.get_datasource(chunk.unit_index, create_missing=False)
        buffer_datasource = self.get_buffer(datasource)
        try:
            return weakref.proxy(buffer_datasource)
        except TypeError:
            try:
                return weakref.proxy(datasource)
            except TypeError:
                return None

    def print_cache_stats(self):
        memory_used = 0
        memory_used += get_buffer_info('input_buffer', self.get_buffer(self.input_datasource))
        memory_used += get_buffer_info('output_buffer', self.get_buffer(self.output_datasource))
        memory_used += get_buffer_info('output_final_buffer', self.get_buffer(self.output_datasource_final))
        memory_used += get_mem_info('SparseOverlapRepository',
                                    self.overlap_repository.datasources.keys(),
                                    self.overlap_repository.datasources.values()
                                    )
        total_memory = print_memory(psutil.Process())
        print('\nActual: %.3f GiB expected: %.3f GiB,  Discrepancy is %.3f GiB' % (
            total_memory, memory_used, total_memory - memory_used))
        print('GlobalOffsets!', len(GLOB), 'summing to', sum(map(lambda x: x.nbytes, GLOB.values())) / 2. ** 30)
        print(list(GLOB.keys()))
        print('GlobalOffsetsBASE!', len(GLOB_BASE), 'summing to', sum(map(lambda x: x.nbytes, GLOB_BASE.values())) / 2. ** 30)
        print(list(GLOB_BASE.keys()))
        return memory_used


class OverlapRepository:
    def __init__(self, *args, **kwargs):
        self.datasources = dict()

    def create(self, mod_index, *args, **kwargs):
        raise NotImplementedError

    def to_datasource_index(self, index):
        return get_mod_index(index)

    def get_datasource(self, index, create_missing=True):
        mod_index = self.to_datasource_index(index)
        if mod_index not in self.datasources:
            if create_missing:
                self.datasources[mod_index] = self.create(mod_index)
            else:
                return None
        return self.datasources[mod_index]

    def clear(self, index):
        mod_index = get_mod_index(index)
        try:
            return self.datasources.pop(mod_index)
        except KeyError:
            return None


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

    def to_datasource_index(self, index):
        return index

    def clear(self, index):
        try:
            del self.datasources[index]
        except KeyError:
            pass
