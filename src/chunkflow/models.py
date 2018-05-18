from datetime import datetime
from functools import lru_cache
from functools import partial
from math import floor
from threading import current_thread

from chunkflow.iterators import UnitBFSIterator


class Chunk(object):
    def __init__(self, block, unit_index):
        self.unit_index = unit_index
        self.slices = block.unit_index_to_slices(unit_index)
        self.data = None
        self.size = block.chunk_size

    def load_data(self, datasource):
        print('VVVVVV %s--%s %s loading into chunk' % (datetime.now(), current_thread().name, self.unit_index))
        self.data = datasource[self.slices]

    def dump_data(self, datasource):
        print('^^^^^^ %s--%s %s dumping from chunk' % (datetime.now(), current_thread().name, self.unit_index))
        datasource[self.slices] = self.data

    def get_core_slices(self):
        pass

    def get_edge_slices(self):
        pass


class Block(object):
    def __init__(self, bounds, chunk_size, overlap=None, base_iterator=None):
        self.bounds = bounds
        self.chunk_size = chunk_size

        if not overlap:
            overlap = tuple([0] * len(chunk_size))

        self.overlap = overlap
        if not base_iterator:
            base_iterator = UnitBFSIterator()
        self.base_iterator = base_iterator

        self.stride = tuple((b_size - olap) for b_size, olap in zip(self.chunk_size, self.overlap))
        self.num_chunks = self._calc_num_chunks()
        self.checkpoints = set()
        self.unit_index_to_chunk = partial(Chunk, self)

    def unit_index_to_slices(self, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + b_size) for b, idx, s, b_size in zip(
            self.bounds, index, self.stride, self.chunk_size))

    def slices_to_unit_index(self, slices):
        return tuple(floor((slice.start - b.start) / s) for b, s, slice in zip(self.bounds, self.stride, slices))

    @lru_cache(maxsize=None)
    def _calc_num_chunks(self):
        data_size = tuple(b.stop - b.start for b in self.bounds)
        num_chunks = tuple(floor(d_size / s) for d_size, s in zip(data_size, self.stride))
        for blocks, b_size, d_size, olap in zip(num_chunks, self.chunk_size, data_size, self.overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    data_size, self.chunk_size, self.overlap))
        return num_chunks

    def checkpoint(self, chunk):
        self.checkpoints.add(chunk.unit_index)

    # def checkpoint_complete(self, indices):
    #     return all(index in checkpoints for index in indices)

    def get_all_neighbors(self, chunk):
        return map(partial(Chunk, self), self.base_iterator.get_all_neighbors(chunk.unit_index, max=self.num_chunks))

    def all_neighbors_checkpointed(self, chunk):
        return all(neighbor.unit_index in self.checkpoints for neighbor in self.get_all_neighbors(chunk))

    def chunk_iterator(self, start):
        yield from map(partial(Chunk, self), self.base_iterator.get(start, self.num_chunks))
