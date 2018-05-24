from datetime import datetime
from functools import lru_cache
from functools import partial
import itertools
from threading import current_thread

from chunkflow.iterators import UnitBFSIterator


class Chunk(object):
    def __init__(self, block, unit_index):
        self.unit_index = unit_index
        self.slices = block.unit_index_to_slices(unit_index)
        self.data = None
        self.size = block.chunk_size
        self.overlap = block.overlap

    def load_data(self, datasource, slices=None):
        print('VVVVVV %s--%s %s loading into chunk' % (datetime.now(), current_thread().name, self.unit_index))

        if slices is None:
            slices = self.slices
        if self.data is None:
            self.data = datasource[slices]
        else:
            self.data[slices] = datasource[slices]
        return self

    def dump_data(self, datasource, slices=None):
        print('^^^^^^ %s--%s %s dumping from chunk' % (datetime.now(), current_thread().name, self.unit_index))
        if slices is None:
            slices = self.slices
        datasource[slices] = self.data[slices]
        return self

    def __eq__(self, other):
        return isinstance(other, Chunk) and self.unit_index == other.unit_index

    def __hash__(self):
        return hash(self.unit_index)

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

        self.data_size = tuple(b.stop - b.start for b in self.bounds)
        self.stride = tuple((c_size - olap) for c_size, olap in zip(self.chunk_size, self.overlap))
        self.num_chunks = tuple((d_size - olap) // s for d_size, olap, s in zip(
            self.data_size, self.overlap, self.stride))

        self.verify_size()

        self.checkpoints = set()
        self.unit_index_to_chunk = partial(Chunk, self)

    def unit_index_to_slices(self, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + c_size) for b, idx, s, c_size in zip(
            self.bounds, index, self.stride, self.chunk_size))

    def slices_to_unit_index(self, slices):
        return tuple((slice.start - b.start) // s for b, s, slice in zip(self.bounds, self.stride, slices))

    @lru_cache(maxsize=None)
    def verify_size(self):
        for chunks, c_size, d_size, olap in zip(self.num_chunks, self.chunk_size, self.data_size, self.overlap):
            if chunks * (c_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    self.data_size, self.chunk_size, self.overlap))

    def checkpoint(self, chunk):
        self.checkpoints.add(chunk.unit_index)

    def get_all_neighbors(self, chunk):
        return map(self.unit_index_to_chunk, self.base_iterator.get_all_neighbors(chunk.unit_index, max=self.num_chunks))

    def all_neighbors_checkpointed(self, chunk):
        return all(neighbor.unit_index in self.checkpoints for neighbor in self.get_all_neighbors(chunk))

    def chunk_iterator(self, start):
        if isinstance(start, Chunk):
            start_index = chunk.unit_index
        else:
            start_index = start
        yield from map(self.unit_index_to_chunk, self.base_iterator.get(start_index, self.num_chunks))

    def core_slices(self, chunk):
        """
        Returns the slices of the core of the block that belongs to the input chunk.
        """
        intersect_slices = []
        for s, b, olap, idx in zip(chunk.slices, self.bounds, self.overlap, range(0, len(chunk.slices))):
            if s.start == b.start:
                intersect_slices.append(slice(s.start + olap, s.stop))
            elif s.stop == b.stop:
                intersect_slices.append(slice(s.start, s.stop - olap))
            else:
                intersect_slices.append(s)

        return tuple(self.remove_chunk_overlap(chunk, intersect_slices))

    def overlap_slices(self, chunk):
        """
        Returns a list of slices for the overlap region of the block that belongs to the input chunk.

        If we have a block:
            dimensions: 7x7
            chunk_size: 3x3
            overlap: 1x1

        This should result in 3x3 chunks. When this function is called with each of these chunks, slices that cover the
        overlap region are returned with no duplicates. Additionally, overlaps across chunks are excluded in a push
        forward fashion, i.e. the slices do not include the portion of the data that had should be accounted for by the
        previous chunk ( previous meaning lesser index ).

        At the non corner chunks, we expect to return a single tuple of slices that
        cover the overlap region, i.e.(not actual format, dictionary used for clarity)
            x: slice(0, 1), y: slice(2, 5)

        For corner chunks, this takes care of overlapping areas so they do not get counted twice.  For example, for the
        chunk at position (0, 0), we should expect to return the tuples of slices:
            x1: slice(0, 3), y1: slice(0, 1)
            x2: slice(0, 1), y2: slice(1, 3)]

        WARNING: not tested for dimensions > 3.

        """
        full_slices = chunk.slices
        num_overlapped = 0

        # determine the common intersect slices within the chunk
        border_dimensions = set()
        intersect_slices = []
        for s, b, olap, idx in zip(full_slices, self.bounds, self.overlap, range(0, len(full_slices))):
            if s.start == b.start:
                intersect_slices.append(slice(s.start, s.start + olap))
                border_dimensions.add(idx)
            elif s.stop == b.stop:
                intersect_slices.append(slice(s.stop - olap, s.stop))
                border_dimensions.add(idx)
            else:
                intersect_slices.append(s)

        # determine the remainder slice that exists with the intersect slices removed
        remainders = tuple(map(lambda x: sub(*x), zip(full_slices, intersect_slices)))

        # generate final list of slices with common intersection removed
        overlapped_slices = []
        processed_dimensions = set()
        for border_dimension in border_dimensions:
            new_slices = list(full_slices)
            overlapped_slices.append(tuple(
                intersect_slices[idx] if idx == border_dimension else
                remainders[idx] if idx in processed_dimensions else
                full_slices[idx]
                for idx in range(0, len(full_slices))
            ))

            processed_dimensions.add(border_dimension)

        return list(map(partial(self.remove_chunk_overlap, chunk), overlapped_slices))

    def remove_chunk_overlap(self, chunk, overlapped_slices):
        """
        remove chunk overlap for slices that
        """
        return tuple(
            slice(o_slice.start + olap, o_slice.stop) if o_slice.start == s.start and o_slice.start != b.start else
            o_slice
            for s, o_slice, olap, b in zip(chunk.slices, overlapped_slices, self.overlap, self.bounds)
        )

def sub(slice_left, slice_right):
    """
    Removes the right slice from the left. Does NOT account for slices on the right that do not touch the border of the
    left
    """
    start = 0
    stop = 0
    if slice_left.start == slice_right.start:
        start = min(slice_left.stop, slice_right.stop)
        stop = max(slice_left.stop, slice_right.stop)
    if slice_left.stop == slice_right.stop:
        start = min(slice_left.start, slice_right.start)
        stop = max(slice_left.start, slice_right.start)

    return slice(start, stop)
