import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray


class BufferedChunkDatasource:
    def __init__(self, block, datasource, num_channels):
        self.block = block
        self.local_cache = dict()
        self.datasource = datasource

    def __setitem__(self, slices, item):
        channel_dimensions = len(self.datasource.shape) - len(slices)
        if not isinstance(item, GlobalOffsetArray):
            global_offset = (0,) * channel_dimensions + tuple(
                (0 if s.start is None else s.start) if isinstance(s, slice) else s for s in slices)
            item = GlobalOffsetArray(item, global_offset=global_offset)

        chunk_indices = self.block.slices_to_unit_indices(slices)
        for chunk_index in chunk_indices:
            if chunk_index not in self.local_cache:
                chunk = self.block.unit_index_to_chunk(chunk_index)
                chunk.data = GlobalOffsetArray(
                    np.zeros((self.num_channels,) + self.block.chunk_shape, dtype=self.dtype),
                    global_offset=(0,) + chunk.offset
                )
                self.local_cache[chunk_index] = chunk
            chunk = self.local_cache[chunk_index]
            chunk.load_data(item, slices=slices)

    def __getitem__(self, slices):
        return self.datasource.__getitem__(slices)

    def clear(self, unit_index=None):
        if unit_index is None:
            chunks = self.local_cache.values()
            self.local_cache.clear()
            return chunks
        elif unit_index in self.local_cache:
            obj = self.local_cache.pop(unit_index)
            return obj
        else:
            return None

    def __getattr__(self, attr):
        return getattr(self.datasource, attr)
