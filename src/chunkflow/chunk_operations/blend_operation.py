import numpy as np

from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityBlend(ChunkOperation):
    def _process(self, chunk):
        pass


class AverageBlend(ChunkOperation):
    """
    This blends by weighting using the average across overlaps.
    """
    def __init__(self, block, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = block

    def _process(self, chunk):
        self.run_blend(chunk)

    def generate_weight_mapping(self, chunk):
        weight_mapping = np.ones(chunk.shape)
        it = np.nditer(weight_mapping, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            multi_index = it.multi_index
            offset_index = [s.start + m_index for s, m_index in zip(chunk.slices, multi_index)]
            for s, b, olap, m_index in zip(chunk.slices, self.block.bounds, chunk.overlap, multi_index):
                offset_index = s.start + m_index
                if (
                    (s.start != b.start and offset_index < s.start + olap) or
                    (s.stop != b.stop and offset_index >= s.stop - olap)
                ):
                    it[0] *= 2
            it.iternext()

        return 1 / weight_mapping

    def run_blend(self, chunk):
        weight_mapping = self.generate_weight_mapping(chunk)
        chunk.data *= weight_mapping
