import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray


class CacheMiss(Exception):
    def __init__(self, message=None, misses=None):
        self.messsage = message
        self.misses = misses

    def __str__(self):
        return 'Cache miss: %s %s' % (self.message if self.message is not None else '', self.misses)


class BufferedChunkDatasource:
    def __init__(self, block, datasource):
        self.block = block
        self.datasource = datasource
        self.channel_dimensions = (datasource.num_channels,)
        self.local_cache = dict()

    def __setitem__(self, slices, item):
        channel_dimensions = len(self.channel_dimensions) - len(slices)
        if not isinstance(item, GlobalOffsetArray):
            global_offset = (0,) * channel_dimensions + tuple(
                (0 if s.start is None else s.start) if isinstance(s, slice) else s for s in slices)
            item = GlobalOffsetArray(item, global_offset=global_offset)

        chunk_indices = self.block.slices_to_unit_indices(slices)
        for chunk_index in chunk_indices:
            if chunk_index not in self.local_cache:
                chunk = self.block.unit_index_to_chunk(chunk_index)
                chunk.data = GlobalOffsetArray(
                    np.zeros(self.channel_dimensions + self.block.chunk_shape, dtype=self.dtype),
                    global_offset=(0,) + chunk.offset
                )
                self.local_cache[chunk_index] = chunk
            chunk = self.local_cache[chunk_index]
            chunk.load_data(item, slices=slices)

    def __getitem__(self, slices):
        unit_indices = self.block.slices_to_unit_indices(slices)
        missing = list(unit_index for unit_index in unit_indices if unit_index not in self.local_cache)

        if len(missing) > 0:
            raise CacheMiss(misses=missing)
        else:
            offset = tuple(s.start for s in slices)
            size = tuple(s.stop - s.start for s in slices)
            data = GlobalOffsetArray(
                np.zeros(self.channel_dimensions + size, dtype=self.dtype),
                global_offset=(0,) * len(self.channel_dimensions) + offset
            )
            for unit_index in unit_indices:
                chunk = self.local_cache[unit_index]
                # TODO need to fix this
                data[chunk.slices] = chunk[slices]
        return data

    def clear(self, chunk=None):
        assert chunk.block.bounds == self.block.bounds and chunk.block.overlap == self.block.overlap

        if chunk is None:
            chunks = self.local_cache.values()
            self.local_cache.clear()
            return chunks
        elif chunk.unit_index in self.local_cache:
            return self.local_cache.pop(chunk.unit_index)
        else:
            return None

    def __getattr__(self, attr):
        return getattr(self.datasource, attr)
