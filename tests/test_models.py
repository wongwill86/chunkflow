import unittest

from chunkflow.iterators import Iterator
from chunkflow.models import Block


class IdentityIterator(Iterator):
    def get_all_neighbors(self, index, max=None):
        return index

    def get(self, start, dimensions):
        yield start


class BlockTest(unittest.TestCase):
    def test_init_wrong_size_no_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)

        with self.assertRaises(ValueError):
            Block(bounds, chunk_size)

    def test_init_wrong_size_overlap(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)
        overlap = (11, 11)

        with self.assertRaises(ValueError):
            Block(bounds, chunk_size, overlap=overlap)

    def test_index_to_slices(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)
        overlap = (10, 10)

        block = Block(bounds, chunk_size, overlap)

        self.assertEquals((slice(0, 30), slice(0, 30)), block.unit_index_to_slices((0, 0)))
        self.assertEquals((slice(0, 30), slice(20, 50)), block.unit_index_to_slices((0, 1)))
        self.assertEquals((slice(20, 50), slice(0, 30)), block.unit_index_to_slices((1, 0)))

    def test_slices_to_index(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)
        overlap = (10, 10)

        block = Block(bounds, chunk_size, overlap)

        self.assertEquals(block.slices_to_unit_index((slice(0, 30), slice(0, 30))), (0, 0))
        self.assertEquals(block.slices_to_unit_index((slice(0, 30), slice(20, 50))), (0, 1))
        self.assertEquals(block.slices_to_unit_index((slice(20, 50), slice(0, 30))), (1, 0))
        self.assertEquals(block.slices_to_unit_index((slice(20, 50), slice(20, 50))), (1, 1))

    def test_iterator(self):
        bounds = (slice(0, 70), slice(0, 70))
        chunk_size = (30, 30)
        overlap = (10, 10)

        block = Block(bounds, chunk_size, overlap=overlap, base_iterator=IdentityIterator())
        start = (0, 0)
        chunks = list(block.chunk_iterator(start))
        self.assertEqual(1, len(chunks))
        self.assertEqual(start, chunks[0].unit_index)


    def test_overlap_slices(self):
        bounds = (slice(0, 5), slice(0, 5))
        chunk_size = (3, 3)
        overlap = (2, 2)

        block = Block(bounds, chunk_size, overlap=overlap)#, base_iterator=IdentityIterator())
        self.assertEquals((3,3), block.num_chunks)

        edge_slices = dict()
        for chunk in block.chunk_iterator((0,0)):
            edge_slices[chunk] = block.edge_slices(chunk)
            print(chunk.unit_index)
            print(edge_slices[chunk])

        # for chunk, olap in edge_slices.items():

        assert False
