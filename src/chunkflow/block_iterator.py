import itertools
from collections import deque
from math import floor

class BlockedIterator(object):
    def check_valid(self, data_size, block_size, overlap):
        data_size = tuple(s.stop - s.start for s in data_size)
        stride = tuple((bs - o) for bs, o in zip(block_size, overlap))
        num_blocks = tuple(floor(ds / st) for ds, st in zip(data_size, stride))
        for blocks, b_size, d_size, olap in zip(num_blocks, block_size, data_size, overlap):
            if blocks * (b_size - olap) + olap != d_size:
                raise ValueError('Data size %s divided by %s with overlap %s does not divide evenly' % (
                    data_size, block_size, overlap))

    def _recurse_iterator(self, bounds, block_size, overlap, dimension):
        if dimension == len(bounds):
            yield


    def iterator(self, bounds, block_size, overlap):
        self.check_valid(bounds, block_size, overlap)

        queue = deque()
        queue.append(tuple([0] * len(bounds)))
        visited = set()

        while(len(queue) > 0):
            internal_index = queue.popleft()
            # Neighbor.get_all_neighbors()


face_neighbor_offsets = [1, -1]
all_neighbor_offsets = [-1, 0, 1]
class Neighbor(object):
    @staticmethod
    def get_face_neighbors(index):
        neighbors = []
        for offset in face_neighbor_offsets:
            for offset_dim in range(0, len(index)):
                neighbors.append(
                    tuple(index[dim] if dim != offset_dim else index[dim] + offset for dim in range(0, len(index))))
        return neighbors

    @staticmethod
    def get_all_neighbors(index):
            neighbors = []
            origin = tuple([0]*len(index))
            for offset in itertools.product(*[all_neighbor_offsets] * len(index)):
                neighbor = tuple(offset[dim] + index[dim] for dim in range(0, len(index)))
                if any(neighbor[dim] < 0 or neighbor[dim] > 3 for dim in range(0, len(index))):
                    continue
                if offset != origin:
                    neighbors.append(neighbor)
            return neighbors


