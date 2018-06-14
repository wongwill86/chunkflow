import numpy as np

from chunkflow.chunk_operations.chunk_operation import ChunkOperation


class IdentityInference(ChunkOperation):
    def __init__(self, output_channels=1, output_data_type=None):
        self.output_data_type = output_data_type
        self.output_channels = output_channels

    def _process(self, chunk):
        if self.output_data_type is not None:
            chunk.data = chunk.data.astype(self.output_data_type)

        if self.output_channels > 1:
            squeezed_data = chunk.data.squeeze()
            new_data = np.tile(squeezed_data, (self.output_channels,) + (1,) * len(squeezed_data.shape))
            chunk.data = new_data
