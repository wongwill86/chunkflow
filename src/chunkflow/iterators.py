import itertools
from collections import deque
from math import floor

all_neighbor_offsets = [-1, 0, 1]


class Iterator(object):
    def get_all_neighbors(self, index):
        raise NotImplementedError

    def iterator(self, index):
        raise NotImplementedError


class UnitIterator(Iterator):
    def __init__(self):
        pass

    def get_all_neighbors(self, index):
        """
        Return all neighbors including diagonals
        """
        neighbors = []
        origin = tuple([0]*len(index))
        for offset in itertools.product(*[all_neighbor_offsets] * len(index)):
            neighbor = tuple(offset[dim] + index[dim] for dim in range(0, len(index)))
            if offset != origin:
                neighbors.append(neighbor)
        return neighbors


class UnitBFSIterator(UnitIterator):
    def iterator(self, start, dimensions):
        """
        Get an iterator that traverses dataset in single units using Breadth-First-Search fashion.
        Units start at 0 and end at param: dimensions
        :param start: location to start iteration from
        :param dimensions: list or tuple of units for each dimension
        """
        queue = deque()
        queue.append(start)
        visited = set((start,))

        def inside_bounds(index):
            return not any(idx < 0 or idx >= dimension for idx, dimension in zip(index, dimensions))

        while(len(queue) > 0):
            internal_index = queue.popleft()
            neighbors = [
                neighbor for neighbor in self.get_all_neighbors(internal_index)
                if inside_bounds(neighbor) and neighbor not in visited
            ]
            visited.update(neighbors)
            queue.extend(neighbors)
            yield internal_index


class BlockedIterator(UnitBFSIterator):
    def __init__(self, bounds, overlap, unit_iterator):
        self.bounds = bounds
        self.overlap = overlap
        self.data_size = tuple(b.stop - b.start for b in self.bounds)
        self.stride = tuple((b_size - olap) for b_size, olap in zip(self.block_size, self.overlap))
        self.num_blocks = tuple(floor(d_size / s) for d_size, s in zip(self.data_size, self.stride))
        self.iterator = unit_iterator

    def get_all_neighbors(self, index):
        raise NotImplementedError

    def bfs_iterator(self, bounds, overlap=None):
        """
        Get an iterator that traverses the bounds given an iterator style
        :param bounds: list or tuple of slices fore start and end positions of data to traverse
        :param block_size: list or tuple of the step size to iterate using
        :param iterator: unit step iterator function that fits the function signature:
            func(start, dimensions).  see iterators.bfs_iterator
        :param overlap: (optional) list or tuple of overlap size for each dimension
        """
        if not overlap:
            overlap = tuple([0] * len(bounds))


        for blocks, b_size, d_size, olap in zip(self.num_blocks, self.block_size, self.data_size, self.overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    self.data_size, self.block_size, self.overlap))

        def internal_index_to_slices(index):
            return tuple(slice(b.start + idx * s, b.start + idx * s + b_size) for b, idx, s, b_size in zip(
                self.bounds, self.index, self.stride, self.block_size))

        start = tuple([0] * len(self.bounds))
        for block in iterator(start, self.num_blocks):
            yield internal_index_to_slices(block)
