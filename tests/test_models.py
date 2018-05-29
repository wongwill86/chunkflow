import itertools
import unittest

import numpy as np

from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.iterators import Iterator
from chunkflow.models import Block
from chunkflow.models import Chunk


class IdentityIterator(Iterator):
    def get_all_neighbors(self, index, max=None):
        return index

    def get(self, start, dimensions):
        yield start


class ChunkTest(unittest.TestCase):
    def test_get_border_slices_2d(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        chunk = Chunk(block, (0, 0))

        borders = list(itertools.product(range(0, len(bounds)), [-1, 1]))

        fake_data = np.zeros(chunk.shape)
        for slices in chunk.border_slices(borders):
            fake_data[slices] += 1

        fake_data[chunk.core_slices(borders)] += 1
        self.assertEquals(np.product(fake_data.shape), fake_data.sum())

    def test_get_border_slices_3d(self):
        bounds = (slice(0, 70), slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30, 30)
        overlap = (10, 10, 10)

        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        chunk = Chunk(block, (0, 0, 0))

        borders = list(itertools.product(range(0, len(bounds)), [-1, 1]))

        fake_data = np.zeros(chunk.shape)
        for slices in chunk.border_slices(borders):
            fake_data[slices] += 1

        fake_data[chunk.core_slices(borders)] += 1
        self.assertEquals(np.product(fake_data.shape), fake_data.sum())


class BlockTest(unittest.TestCase):
    def test_init_wrong_size_no_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)

        with self.assertRaises(ValueError):
            block = Block(
                bounds=bounds,
                chunk_shape=chunk_shape,
            )

    def test_init_wrong_size_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (11, 11)

        with self.assertRaises(ValueError):
            block = Block(
                bounds=bounds,
                chunk_shape=chunk_shape,
                overlap=overlap
            )

    def test_index_to_slices(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)


        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        self.assertEquals((slice(0, 30), slice(0, 30)), block.unit_index_to_slices((0, 0)))
        self.assertEquals((slice(0, 30), slice(20, 50)), block.unit_index_to_slices((0, 1)))
        self.assertEquals((slice(20, 50), slice(0, 30)), block.unit_index_to_slices((1, 0)))

    def test_slices_to_index(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)


        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        self.assertEquals(block.slices_to_unit_index((slice(0, 30), slice(0, 30))), (0, 0))
        self.assertEquals(block.slices_to_unit_index((slice(0, 30), slice(20, 50))), (0, 1))
        self.assertEquals(block.slices_to_unit_index((slice(20, 50), slice(0, 30))), (1, 0))
        self.assertEquals(block.slices_to_unit_index((slice(20, 50), slice(20, 50))), (1, 1))

    def test_iterator(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_shape = (30, 30)
        overlap = (10, 10)

        start = (0, 0)
        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap,
            base_iterator=IdentityIterator()
        )

        chunks = list(block.chunk_iterator(start))
        self.assertEqual(1, len(chunks))
        self.assertEqual(start, chunks[0].unit_index)

    def test_get_slices_2d(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)


        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0))
        self.assertEquals((3, 3), block.num_chunks)

        for chunk in block.chunk_iterator((0, 0)):
            for edge_slice in block.overlap_slices(chunk):
                fake_data[edge_slice] += 1
            fake_data[block.core_slices(chunk)] += 1
        self.assertEquals(np.product(fake_data.shape), fake_data.sum())

    def test_overlap_slices_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(
            bounds=bounds,
            chunk_shape=chunk_shape,
            overlap=overlap
        )

        self.assertEquals((3, 3, 3), block.num_chunks)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0, 0))
        for chunk in block.chunk_iterator((1, 0, 1)):
            for edge_slice in block.overlap_slices(chunk):
                fake_data[edge_slice] += 1
            fake_data[block.core_slices(chunk)] += 1
        self.assertEquals(np.product(fake_data.shape), fake_data.sum())
