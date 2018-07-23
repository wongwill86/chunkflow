import numpy as np
from chunkblocks.global_offset_array import GlobalOffsetArray

from chunkflow.datasource_manager import DatasourceRepository


class SparseMatrixDatasourceRepository(DatasourceRepository):
    def __init__(self, block, num_channels, *args, **kwargs):
        self.block = block
        self.num_channels = num_channels
        super().__init__(*args, **kwargs)

    def get_datasource(self, index):
        if index not in self.overlap_datasources:
            self.overlap_datasources[index] = self.create(index)
        return self.overlap_datasources[index]

    def create(self, index, *args, **kwargs):
        global_offset = (0,) + tuple(s.start for s in self.block.unit_index_to_slices(index))
        return GlobalOffsetArray(
            np.zeros((self.num_channels,) + self.block.chunk_shape, dtype=self.output_datasource.dtype),
            global_offset=global_offset,
        )
