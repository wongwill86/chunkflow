import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray


class CacheMiss(Exception):
    def __init__(self, message=None, misses=None):
        self.messsage = message
        self.misses = misses

    def __str__(self):
        return 'Cache miss: %s %s' % (self.message if self.message is not None else '', self.misses)


class ChunkBuffer:
    def __init__(self, block, datasource, channel_dimensions):
        self.block = block
        self.datasource = datasource
        self.channel_dimensions = channel_dimensions
        self.local_cache = dict()

    def create(self, offset):
        shape = self.channel_dimensions + self.block.chunk_shape
        # shape = (10, 20, 256, 256)
        print(shape)

        return GlobalOffsetArray(
            np.ones(shape, dtype=self.dtype) - 1,
            global_offset=(0,) * len(self.channel_dimensions) + offset
        )

    def __setitem__(self, slices, item):
        self.setme(slices, item)

    def setme(self, slices, item):
        if not isinstance(item, GlobalOffsetArray):
            global_offset = (0,) * len(self.channel_dimensions) + tuple(
                (0 if s.start is None else s.start) if isinstance(s, slice) else s for s in slices)
            item = GlobalOffsetArray(item, global_offset=global_offset)

        chunk_indices = self.block.slices_to_unit_indices(slices)
        for chunk_index in chunk_indices:
            chunk = self.block.unit_index_to_chunk(chunk_index)
            try:
                chunk = self.local_cache[chunk_index]
                chunk.load_data(item, slices=slices)
            except (KeyError, AttributeError):  # Attribute error in case of retrieving a future from self.local_cache
                chunk = self.block.unit_index_to_chunk(chunk_index)
                chunk.data = self.create(chunk.offset)
                self.local_cache[chunk_index] = chunk
                chunk.load_data(item, slices=slices)

    def __getitem__(self, slices):
        return self.get_item(slices)

    def get_item(self, slices, fill_missing=False):
        unit_indices = list(self.block.slices_to_unit_indices(slices))

        if not fill_missing:
            misses = list(unit_index for unit_index in unit_indices if unit_index not in self.local_cache or
                          not hasattr(self.local_cache[unit_index], 'data'))

            if len(misses) > 0:
                raise CacheMiss(misses=misses)

        channel_slices = self.normalize_channel_slices(slices)
        full_slices = channel_slices + slices[-len(self.block.shape):]

        offset = tuple(s.start for s in full_slices)
        size = tuple(s.stop - s.start for s in full_slices)
        data = GlobalOffsetArray(np.ones(size, dtype=self.dtype) - 1, global_offset=offset)
        slices = full_slices[len(self.channel_dimensions):]

        for unit_index in unit_indices:
            chunk = self.block.unit_index_to_chunk(unit_index)
            try:
                chunk = self.local_cache[unit_index]
            except KeyError:
                # If we had cared about this fill_missing = False should have raised an exception already
                pass
            else:
                normalized_slices = channel_slices + tuple(
                    slice(s1.start if s2.start < s1.start else s2.start, s1.stop if s2.stop > s1.stop else s2.stop)
                    for s1, s2 in zip(chunk.slices, slices))
                data[normalized_slices] = chunk.data[normalized_slices]

        return data

    def normalize_channel_slices(self, slices):
        channel_slices = slices[0:-len(self.block.shape)]
        if len(channel_slices) > 0:
            return tuple(slice(0, c) if s.start is None else s for c, s in zip(self.channel_dimensions,
                                                                               channel_slices))
        else:
            return tuple(slice(0, c) for c in self.channel_dimensions)

    def clear(self, chunk=None):
        if chunk is not None:
            assert chunk.block.bounds == self.block.bounds and chunk.block.overlap == self.block.overlap, (
                'Attempting to clear with chunk of incorrect overlap and bounds'
            )

        if chunk is None:
            chunks = list(self.local_cache.values())
            self.local_cache.clear()
            return chunks
        elif chunk.unit_index in self.local_cache:
            return self.local_cache.pop(chunk.unit_index)
        else:
            return None

    def __getattr__(self, attr):
        return getattr(self.datasource, attr)
