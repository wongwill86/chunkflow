import unittest
# from chunkflow.blocked_iterator import Neighbor
import itertools
from rx import Observable


face_neighbor_offsets = [1, -1]
all_neighbor_offsets = [-1, 0, 1]


lower_bound = 0
upper_bound = 1

def get_face_neighbors(index):
    neighbors = set()
    for offset in face_neighbor_offsets:
        for offset_dim in range(0, len(index)):
            neighbor = tuple(index[dim] if dim != offset_dim else index[dim] + offset for dim in range(0, len(index)))
            if any(idx < lower_bound or idx > upper_bound for idx in neighbor):
                continue
            neighbors.add(neighbor)
    return neighbors

def get_all_neighbors(index):
        neighbors = set()
        origin = tuple([0]*len(index))
        for offset in itertools.product(*[all_neighbor_offsets] * len(index)):
            neighbor = tuple(offset[dim] + index[dim] for dim in range(0, len(index)))
            if any(idx < lower_bound or idx > upper_bound for idx in neighbor):
                continue
            if offset != origin:
                neighbors.add(neighbor)
        return neighbors

parents = set()

def run_inference(index):
    print('running inference at index %s' % (index,))

def blend(index):
    print('blending at %s' % (index,))

def create_blend(index):
    return lambda x: blend(index)

def recurse(index):
    print('recursing at %s ' % (index, ))
    neighbor_stream = (
        Observable.just(index)
        .do_action(run_inference)
        .flat_map(get_face_neighbors)
        .filter(lambda x: x not in parents)
        .do_action(lambda x: parents.add(x))
    )
    return neighbor_stream.flat_map(recurse)

class BlockIteratorTest(unittest.TestCase):
    def test_blah(self):
        parents.clear()
        start = (0,0)
        parents.add(start)
        blah_stream = Observable.just(start).flat_map(recurse)#.publish()
        blah_stream.subscribe(print)
        # blah_stream.connect()
        assert False

