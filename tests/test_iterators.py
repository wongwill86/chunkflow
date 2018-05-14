import unittest
from collections import deque

from chunkflow import iterators


class UnitIteratorTest(unittest.TestCase):
    def test_unit_get_all_neighbors(self):
        iterator = iterators.UnitIterator()
        expected_neighbors = set([
            (0, 1),
            (1, 0),
            (1, 1),
            (0, -1),
            (-1, 0),
            (-1, -1),
            (-1, 1),
            (1, -1),
        ])
        neighbors = iterator.get_all_neighbors((0, 0))
        self.assertEquals(expected_neighbors, set(neighbors))

class UnitBFSIteratorTest(unittest.TestCase):

    def test_bfs_iterator(self):
        expected_bfs_steps = [
            set([(0, 0)]),
            set([(0, 1), (1, 0), (1, 1), ]),
            set([(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), ])
        ]
        iterator = iterators.UnitBFSIterator()
        bfs = deque(iterator.iterator((0, 0), (3, 3)))
        self.assertGreater(len(bfs), 9)
        for expected_bfs in expected_bfs_steps:
            while len(expected_bfs) and len(bfs):
                step = bfs.popleft()
                expected_bfs.remove(step)
            self.assertEquals(0, len(expected_bfs))
        self.assertEquals(0, len(bfs))
