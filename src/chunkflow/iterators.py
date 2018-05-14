import itertools
from collections import deque
from math import floor

all_neighbor_offsets = [-1, 0, 1]


class Iterator(object):
    def get_all_neighbors(self, index):
        raise NotImplemented

    def iterator(self, index):
        raise NotImplemented


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
            # print('index, is %s, dimensions is %s' % (index, dimensions))
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
            # print(queue)
            # print(visited)
