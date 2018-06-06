import itertools
from datetime import datetime
from functools import lru_cache
from functools import partial
from threading import current_thread

from chunkflow.iterators import UnitBFSIterator


@lru_cache(maxsize=None)
def all_borders(dimensions):
    return tuple(itertools.product(range(0, dimensions), (-1, 1)))


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


class Chunk(object):
    def __init__(self, block, unit_index):
        self.unit_index = unit_index
        self.slices = block.unit_index_to_slices(unit_index)
        self.global_offset = [s.start for s in self.slices]
        self.data = None
        self.shape = block.chunk_shape
        self.overlap = block.overlap
        self.all_borders = all_borders(len(self.shape))

    def match_datasource_dimensions(self, datasource, slices):
        return (slice(None),) * (len(datasource.shape) - len(slices)) + slices

    def load_data(self, datasource, slices=None):
        print('VVVVVV %s--%s %s loading into chunk' % (datetime.now(), current_thread().name, self.unit_index))

        if slices is None:
            slices = self.slices

        slices = self.match_datasource_dimensions(datasource, slices)

        if self.data is None:
            self.data = datasource[slices].copy()
        else:
            self.data[slices] = datasource[slices]

        return self

    def dump_data(self, datasource, slices=None):
        print('^^^^^^ %s--%s %s dumping from chunk' % (datetime.now(), current_thread().name, self.unit_index))
        if slices is None:
            slices = self.slices

        slices = self.match_datasource_dimensions(datasource, slices)

        print(datasource.info)
        print(slices)
        print(self.data[slices].shape)
        datasource[slices] = self.data[slices]
        return self

    def __eq__(self, other):
        return isinstance(other, Chunk) and self.unit_index == other.unit_index

    def __hash__(self):
        return hash(self.unit_index)

    def core_slices(self, borders=None):
        """
        Returns a list of non-intersecting slices that is excluded by the requested borders. Borders is a list of
        tuples:
            (dimension index of border, border direction)

        Border direction is specified by -1 to represent the border in the negative index direction and +1 for the
        positive index direction.
        """
        if borders is None:
            borders = self.all_borders

        core_slices = list(self.slices)
        for border, direction in borders:
            core_slice = core_slices[border]
            if direction < 0:
                core_slice = slice(core_slice.start + self.overlap[border], core_slice.stop)
            else:
                core_slice = slice(core_slice.start, core_slice.stop - self.overlap[border])
            core_slices[border] = core_slice

        return tuple(core_slices)

    def border_slices(self, borders=None):
        """
        Returns a list of non-intersecting slices that cover the requested borders. Borders is a list of tuples:
            (dimension index of border, border direction)

        Border direction is specified by -1 to represent the border in the negative index direction and +1 for the
        positive index direction.
        """
        if borders is None:
            borders = self.all_borders

        border_slices = []

        processed_dimensions = set()
        remainders = list(self.slices)

        for border, direction in borders:
            if direction < 0:
                border_slice = slice(self.slices[border].start, self.slices[border].start +
                                     self.overlap[border])
            else:
                border_slice = slice(self.slices[border].stop - self.overlap[border],
                                     self.slices[border].stop)

            new_slices = tuple(
                border_slice if idx == border else
                remainders[idx] if idx in processed_dimensions else
                self.slices[idx]
                for idx in range(0, len(self.slices))
            )
            remainders[border] = sub(remainders[border], new_slices[border])
            border_slices.append(new_slices)
            processed_dimensions.add(border)

        return border_slices


class Block(object):
    def __init__(self, bounds, chunk_shape, overlap=None, base_iterator=None):
        self.bounds = bounds
        self.chunk_shape = chunk_shape

        if not overlap:
            overlap = tuple([0] * len(chunk_shape))

        self.overlap = overlap
        if not base_iterator:
            base_iterator = UnitBFSIterator()
        self.base_iterator = base_iterator

        self.shape = tuple(b.stop - b.start for b in self.bounds)
        self.stride = tuple((c_shape - olap) for c_shape, olap in zip(self.chunk_shape, self.overlap))
        self.num_chunks = tuple((shp - olap) // s for shp, olap, s in zip(
            self.shape, self.overlap, self.stride))

        self.verify_size()

        self.checkpoints = set()
        self.unit_index_to_chunk = partial(Chunk, self)

    def unit_index_to_slices(self, index):
        return tuple(slice(b.start + idx * s, b.start + idx * s + c_shape) for b, idx, s, c_shape in zip(
            self.bounds, index, self.stride, self.chunk_shape))

    def slices_to_unit_index(self, slices):
        return tuple((slice.start - b.start) // s for b, s, slice in zip(self.bounds, self.stride, slices))

    # def to_output_slices(self, slices):
    #     return (slice(None),) * (len(self.output_shape) - len(slices)) + slices

    def verify_size(self):
        for chunks, c_shape, shp, olap in zip(self.num_chunks, self.chunk_shape, self.shape, self.overlap):
            if chunks * (c_shape - olap) + olap != shp:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    self.shape, self.chunk_shape, self.overlap))

    def checkpoint(self, chunk):
        self.checkpoints.add(chunk.unit_index)

    def get_all_neighbors(self, chunk):
        return map(self.unit_index_to_chunk,
                   self.base_iterator.get_all_neighbors(chunk.unit_index, max=self.num_chunks))

    def all_neighbors_checkpointed(self, chunk):
        return all(neighbor.unit_index in self.checkpoints for neighbor in self.get_all_neighbors(chunk))

    def chunk_iterator(self, start):
        if isinstance(start, Chunk):
            start_index = start.unit_index
        else:
            start_index = start
        yield from map(self.unit_index_to_chunk, self.base_iterator.get(start_index, self.num_chunks))

    def core_slices(self, chunk):
        """
        Returns the slices of the chunk that corresponds to the block's core that has no overlap with other blocks.
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

    def overlap_borders(self, chunk):
        """
        Get a list of borders in the chunk that correspond to the block's overlap region.

        Returns list of borders in the form of tuples:
            (dimension index of border, border direction)

        Border direction is specified by -1 to represent the border in the negative index direction and +1 for the
        positive index direction.

        See py:method::overlap_slices(chunk) for usage
        """
        # determine the common intersect slices within the chunk
        borders = []
        for s, b, olap, idx in zip(chunk.slices, self.bounds, self.overlap, range(0, len(chunk.slices))):
            if s.start == b.start:
                borders.append((idx, -1))
            elif s.stop == b.stop:
                borders.append((idx, 1))
        return borders

    def remove_chunk_overlap(self, chunk, overlapped_slices):
        """
        Modify slices to remove the common intersection of the chunks within the block. Common intersections are
        excluded in a push forward fashion, i.e. the slices do not include the portion of the data that had should be
        accounted for by the previous chunk ( previous meaning of a lesser index ).

        See py:method::overlap_slices(chunk) for usage
        """
        return tuple(
            slice(o_slice.start + olap, o_slice.stop) if o_slice.start == s.start and o_slice.start != b.start else
            o_slice
            for s, o_slice, olap, b in zip(chunk.slices, overlapped_slices, self.overlap, self.bounds)
        )

    def overlap_slices(self, chunk):
        """
        Get a list of the slices in the chunk that correspond to the block's overlap region.

        If we have a block:
            dimensions: 7x7
            chunk_shape: 3x3
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
        return [
            self.remove_chunk_overlap(chunk, overlapped_slice) for overlapped_slice in chunk.border_slices(
                self.overlap_borders(chunk))
        ]
