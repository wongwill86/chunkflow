import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray
from chunkblocks.models import Block, Chunk

from chunkflow.chunk_operations.blend_operation import AverageBlend


class TestAverageBlend:
    def test_weight_mapping_2d(self):
        chunk_shape = (2, 4)
        overlap = (1, 2)
        offset = (0, 0)
        num_chunks = (4, 3)

        block = Block(offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0))

        blend_operation = AverageBlend(block)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=offset)
        for chunk in block.chunk_iterator((0, 0)):
            chunk.data = np.zeros(1, dtype=np.float32)
            fake_data[chunk.slices] += blend_operation.get_weight_mapping(chunk)

        assert fake_data.sum() == np.product(fake_data.shape)

    def test_weight_mapping_3d(self):
        chunk_shape = (2, 4, 4)
        overlap = (1, 2, 2)
        offset = (0, 0, 0)
        num_chunks = (3, 3, 5)
        block = Block(offset=offset, num_chunks=num_chunks, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0, 0))

        blend_operation = AverageBlend(block)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=offset)
        for chunk in block.chunk_iterator((0, 0, 0)):
            chunk.data = np.zeros(1, dtype=np.float32)
            fake_data[chunk.slices] += blend_operation.generate_weight_mapping(chunk)

        assert fake_data.sum() == np.product(fake_data.shape)
