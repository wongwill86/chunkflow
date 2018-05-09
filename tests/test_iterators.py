import unittest
from collections import deque

from chunkflow import iterators


class BlockIteratorTest(unittest.TestCase):
    def test_get_all_neighbors(self):
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
        neighbors = iterators.get_all_neighbors((0, 0))
        self.assertEquals(expected_neighbors, set(neighbors))

    def test_bfs_iterator(self):
        expected_bfs_steps = [
            set([(0, 0)]),
            set([(0, 1), (1, 0), (1, 1), ]),
            set([(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), ])
        ]
        bfs = deque(iterators.bfs_iterator((0, 0), (3, 3)))
        for expected_bfs in expected_bfs_steps:
            while len(expected_bfs) and len(bfs):
                step = bfs.popleft()
                expected_bfs.remove(step)
            self.assertEquals(0, len(expected_bfs))
        self.assertEquals(0, len(bfs))

    def test_blocked_iterator_wrong_size_no_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)

        with self.assertRaises(ValueError):
            for a in iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator):
                print(a)

    def test_blocked_iterator_wrong_size_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)
        overlap = (11, 11)

        with self.assertRaises(ValueError):
            for a in iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator, overlap=overlap):
                print(a)

    def test_blocked_iterator_no_overlap(self):
        expected_bfs_steps = [
            set([str((slice(0, 30), slice(0, 30)))]),
            set([
                str((slice(0, 30), slice(30, 60))),
                str((slice(30, 60), slice(0, 30))),
                str((slice(30, 60), slice(30, 60))),
            ]),
            set([
                str((slice(0, 30), slice(60, 90))),
                str((slice(30, 60), slice(60, 90))),
                str((slice(60, 90), slice(60, 90))),
                str((slice(60, 90), slice(30, 60))),
                str((slice(60, 90), slice(0, 30))),
            ])
        ]
        print(expected_bfs_steps)

        bounds = (slice(0, 90), slice(0, 90))
        block_size = (30, 30)
        bfs = deque(iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator))
        for expected_bfs in expected_bfs_steps:
            while len(expected_bfs) and len(bfs):
                step = bfs.popleft()
                expected_bfs.remove(str(step))
            self.assertEquals(0, len(expected_bfs))

        self.assertEquals(0, len(bfs))

    def test_blocked_iterator_overlap(self):
        expected_bfs_steps = [
            set([str((slice(0, 30), slice(0, 30)))]),
            set([
                str((slice(0, 30), slice(20, 50))),
                str((slice(20, 50), slice(0, 30))),
                str((slice(20, 50), slice(20, 50))),
            ]),
            set([
                str((slice(0, 30), slice(40, 70))),
                str((slice(20, 50), slice(40, 70))),
                str((slice(40, 70), slice(40, 70))),
                str((slice(40, 70), slice(20, 50))),
                str((slice(40, 70), slice(0, 30))),
            ])
        ]
        print(expected_bfs_steps)

        bounds = (slice(0, 70), slice(0, 70))
        block_size = (30, 30)
        overlap = (10, 10)
        bfs = deque(iterators.blocked_iterator(bounds, block_size, iterators.bfs_iterator, overlap=overlap))
        for expected_bfs in expected_bfs_steps:
            while len(expected_bfs) and len(bfs):
                step = bfs.popleft()
                expected_bfs.remove(str(step))
            self.assertEquals(0, len(expected_bfs))

        self.assertEquals(0, len(bfs))
