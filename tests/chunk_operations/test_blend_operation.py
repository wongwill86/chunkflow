import numpy as np

from chunkflow.chunk_operations.blend_operation import AverageBlend
from chunkflow.global_offset_array import GlobalOffsetArray
from chunkflow.models import Block
from chunkflow.models import Chunk


class TestAverageBlend:
    def test_weight_mapping_2d(self):
        bounds = (slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3)
        overlap = (1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0))

        blend_operation = AverageBlend(block)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0))
        for chunk in block.chunk_iterator((0, 0)):
            fake_data[chunk.slices] += blend_operation.generate_weight_mapping(chunk)

        assert fake_data.sum() == np.product(fake_data.shape)

    def test_weight_mapping_3d(self):
        bounds = (slice(0, 7), slice(0, 7), slice(0, 7))
        chunk_shape = (3, 3, 3)
        overlap = (1, 1, 1)

        block = Block(bounds=bounds, chunk_shape=chunk_shape, overlap=overlap)

        chunk = Chunk(block, (0, 0, 0))

        blend_operation = AverageBlend(block)

        fake_data = GlobalOffsetArray(np.zeros(block.shape), global_offset=(0, 0, 0))
        for chunk in block.chunk_iterator((0, 0, 0)):
            fake_data[chunk.slices] += blend_operation.generate_weight_mapping(chunk)

        assert fake_data.sum() == np.product(fake_data.shape)
