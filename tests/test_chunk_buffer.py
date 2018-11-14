import numpy as np
import pytest
from chunkblocks.models import Block

from chunkflow.chunk_buffer import ChunkBuffer


class TestChunkBuffer:

    def test_clear_all(self, output_cloudvolume):
        chunk_shape = output_cloudvolume.underlying[::-1]
        offset = output_cloudvolume.voxel_offset[::-1]
        size = output_cloudvolume.volume_size[::-1]

        bounds = tuple(slice(o, o + s) for o, s in zip(offset, size))
        block = Block(bounds=bounds, chunk_shape=chunk_shape)

        chunk_buffer = ChunkBuffer(block, output_cloudvolume, (output_cloudvolume.num_channels,))
        slices = (
            slice(offset[0], offset[0] + chunk_shape[0] * 1 + chunk_shape[0] // 2),
            slice(offset[1], offset[1] + chunk_shape[1] * 1 + chunk_shape[0] // 3),
            slice(offset[2], offset[2] + chunk_shape[2] * 1 + chunk_shape[0] // 4),
        )
        item_shape = (output_cloudvolume.num_channels,) + tuple(s.stop - s.start for s in slices)
        chunk_buffer[slices] = np.ones(item_shape)
        assert np.array_equal(output_cloudvolume[slices], np.zeros(item_shape))

        # set again to test cache works properly
        chunk_buffer[slices] = np.ones(item_shape)
        assert np.array_equal(output_cloudvolume[slices], np.zeros(item_shape))
        print(chunk_buffer.local_cache)

        cleared_chunks = chunk_buffer.clear()
        assert len(chunk_buffer.local_cache) == 0
        assert len(cleared_chunks) == 1 << 3

    def test_clear_wrong_chunk(self, output_cloudvolume):
        chunk_shape = output_cloudvolume.underlying[::-1]
        offset = output_cloudvolume.voxel_offset[::-1]
        size = output_cloudvolume.volume_size[::-1]

        bounds = tuple(slice(o, o + s) for o, s in zip(offset, size))
        block = Block(bounds=bounds, chunk_shape=chunk_shape)

        chunk_buffer = ChunkBuffer(block, output_cloudvolume, (output_cloudvolume.num_channels,))

        other_bounds = (slice(200, 203), slice(100, 103), slice(50, 53))
        other_chunk_shape = (3, 3, 3)
        other_overlap = (1, 1, 1)
        other_block = Block(bounds=other_bounds, chunk_shape=other_chunk_shape, overlap=other_overlap)

        with pytest.raises(AssertionError):
            chunk_buffer.clear(next(other_block.chunk_iterator()))

    def test_clear_missing_chunk(self, output_cloudvolume):
        chunk_shape = output_cloudvolume.underlying[::-1]
        offset = output_cloudvolume.voxel_offset[::-1]
        size = output_cloudvolume.volume_size[::-1]

        bounds = tuple(slice(o, o + s) for o, s in zip(offset, size))
        block = Block(bounds=bounds, chunk_shape=chunk_shape)

        chunk_buffer = ChunkBuffer(block, output_cloudvolume, (output_cloudvolume.num_channels,))

        cleared_chunk = chunk_buffer.clear(next(block.chunk_iterator()))

        assert cleared_chunk is None

    def test_clear_single_chunk(self, output_cloudvolume):
        chunk_shape = output_cloudvolume.underlying[::-1]
        offset = output_cloudvolume.voxel_offset[::-1]
        size = output_cloudvolume.volume_size[::-1]

        bounds = tuple(slice(o, o + s) for o, s in zip(offset, size))
        block = Block(bounds=bounds, chunk_shape=chunk_shape)

        chunk_buffer = ChunkBuffer(block, output_cloudvolume, (output_cloudvolume.num_channels,))

        slices = (
            slice(offset[0], offset[0] + chunk_shape[0] * 1 + chunk_shape[0] // 2),
            slice(offset[1], offset[1] + chunk_shape[1] * 1 + chunk_shape[0] // 3),
            slice(offset[2], offset[2] + chunk_shape[2] * 1 + chunk_shape[0] // 4),
        )
        item_shape = (output_cloudvolume.num_channels,) + tuple(s.stop - s.start for s in slices)
        chunk_buffer[slices] = np.ones(item_shape)

        chunk_to_clear = next(block.chunk_iterator())

        cleared_chunk = chunk_buffer.clear(chunk_to_clear)

        assert cleared_chunk.unit_index == chunk_to_clear.unit_index
        assert len(chunk_buffer.local_cache) == (1 << 3) - 1
