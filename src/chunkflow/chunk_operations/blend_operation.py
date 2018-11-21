import numpy as np

from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityBlend(ChunkOperation):
    def _process(self, chunk):
        pass


class AverageBlend(ChunkOperation):
    """
    This blends by weighting using the average across overlaps.

    :param block: dataset block for computing borders
    :param weight_borders: should the edge block borders be included to be weighted
    """
    def __init__(self, block, weight_borders=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block
        self.weight_cache = {}
        self.weight_borders = weight_borders

    def _process(self, chunk):
        self.run_blend(chunk)

    def remove_offsets(self, chunk, slices, offset):
        return

    def is_border_slice(self, slices):
        """
        Because overlap of inference chunk is the same as across task blocks. we can assume any chunk borders will be
        entirely encapsulated within a task border
        """
        for s, b, olap in zip(slices, self.block.bounds, self.block.overlap):
            if not ((s.start >= b.start + olap and s.start < b.stop - olap) or
                    (s.stop > b.start + olap and s.stop <= b.stop - olap)):
                return True
        return False

    def generate_weight_mapping(self, chunk):
        weight_mapping = np.ones(chunk.shape, dtype=chunk.data.dtype)
        # remove the offset for these slice ranges
        # overlap_slices = [slices for slices in chunk.border_slices(nonintersecting=False) if not self.is_border_slice(
        #     slices)]
        overlap_slices = [slices for slices in chunk.border_slices(nonintersecting=False) if self.weight_borders or not
                          self.is_border_slice(slices)]

        overlap_slices = [
            tuple(slice(s.start - o, s.stop - o) for s, o in zip(
                slices, chunk.offset)) for slices in overlap_slices]

        for slices in overlap_slices:
            weight_mapping[slices] *= 2

        weight_mapping = 1 / weight_mapping
        return weight_mapping

    def get_weight_mapping(self, chunk):
        key = chunk.shape + (chunk.data.dtype,) + tuple(set(self.block.overlap_borders(chunk)))
        if key not in self.weight_cache:
            self.weight_cache[key] = self.generate_weight_mapping(chunk)
        return self.weight_cache[key]

    def run_blend(self, chunk):
        weight_mapping = self.get_weight_mapping(chunk)
        chunk.data *= weight_mapping


class BlendFactory:
    def __init__(self, block):
        self.block = block

    def get_operation(self, framework):
        if framework == 'average':
            return AverageBlend(self.block)
        elif framework == 'identity':
            return IdentityBlend()
        else:
            return AverageBlend(self.block)
