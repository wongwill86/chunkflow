import numpy as np
from chunkblocks.models import Block

from chunkflow.buffered_chunk_datasource import BufferedChunkDatasource


class TestBufferedChunkDatasource:

    def test_create(self, output_cloudvolume):
        return

        chunk_shape = output_cloudvolume.underlying[::-1]
        offset = output_cloudvolume.voxel_offset[::-1]
        size = output_cloudvolume.volume_size[::-1]

        print('channels', output_cloudvolume.num_channels)
        bounds = tuple(slice(o, o + s) for o, s in zip(offset, size))
        block = Block(bounds=bounds, chunk_shape=chunk_shape)
        datasource = BufferedChunkDatasource(block, output_cloudvolume)
        slices = (slice(200, 203), slice(100, 122), slice(50, 78))
        item_shape = (output_cloudvolume.num_channels,) + tuple(s.stop - s.start for s in slices)
        datasource[slices] = np.ones(item_shape)
        assert np.array_equal(output_cloudvolume[slices], np.zeros(item_shape))

        # set again to test cache works properly
        datasource[slices] = np.ones(item_shape)
        assert np.array_equal(output_cloudvolume[slices], np.zeros(item_shape))

        datasource.flush()
        assert np.array_equal(output_cloudvolume[slices], np.ones(item_shape))
        assert len(datasource.local_cache) == 0
